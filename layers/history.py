"""History-only conditioning and optional dataset-specific response settings."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from .embedding import sinusoidal_timestep_embedding
from .state_response import StateResponse, StateResponseEDM, correlation


class HistoryCondition(nn.Module):
    def __init__(self, config, mode, period, decay=0.0):
        super().__init__()
        self.mode, self.horizon = mode, config.horizon
        self.decay = decay
        self.period = min(period, config.seq_len)
        self.register_buffer(
            "indices",
            config.seq_len - self.period + torch.arange(config.horizon) % self.period,
        )
        if mode == "linear":
            self.residual = nn.Sequential(
                nn.Linear(config.seq_len, 32), nn.GELU(), nn.Linear(32, config.horizon)
            )
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def forward(self, past):
        exogenous = past[..., 1:]
        if self.mode == "last":
            path = exogenous[:, -1:].expand(-1, self.horizon, -1)
        else:
            path = exogenous[:, self.indices]
        if self.mode == "cycle_mean":
            # Same-phase average of up to three observed cycles; no fitted model.
            cycles = min(3, exogenous.shape[1] // self.period)
            path = torch.stack(
                [exogenous[:, self.indices - i * self.period] for i in range(cycles)]
            ).mean(0)
        if self.mode == "linear":
            mean = exogenous.mean(1, keepdim=True)
            scale = (
                exogenous.var(1, keepdim=True, unbiased=False)
                .add(1e-5)
                .sqrt()
                .clamp_min(0.1)
            )
            residual = self.residual(
                ((exogenous - mean) / scale).transpose(1, 2)
            ).transpose(1, 2)
            path = path + scale * residual.tanh()
        if self.decay:
            lead = torch.arange(
                1, self.horizon + 1, device=past.device, dtype=past.dtype
            )
            confidence = (-lead / self.decay).exp()[None, :, None]
            level = exogenous.mean(1, keepdim=True)
            path = level + confidence * (path - level)
        return path


class HistoricalStateResponseEDM(StateResponseEDM):
    def __init__(
        self,
        *args,
        history_condition="linear",
        history_period=24,
        history_decay=0.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            self.history_conditioner = HistoryCondition(
                self.config, history_condition, history_period, history_decay
            )

    def condition(self, past, future_condition, base_target):
        # Ignore future_condition even if an external caller supplies true values.
        return super().condition(past, self.history_conditioner(past), base_target)


class LagHistoryBank(nn.Module):
    """Lead-dependent retrieval from observed patches at the existing lag grid."""

    def __init__(self, config, lags, period=24, width=24, patch=8):
        super().__init__()
        values = lags.tolist()
        values += [
            v
            for v in (period, 7 * period, config.seq_len // 2)
            if 0 < v <= config.seq_len // 2 and v not in values
        ]
        lags = torch.tensor(list(dict.fromkeys(values)))
        indices = (
            config.seq_len
            - 1
            - lags[:, None]
            - torch.arange(min(patch, config.seq_len)).flip(0)
        )
        self.register_buffer("indices", indices.clamp_min(0))
        self.register_buffer(
            "correlation_indices",
            torch.arange(int(lags.max()), config.seq_len)[:, None] - lags,
        )
        self.lag_bias = nn.Parameter(torch.zeros(config.future_cond_dim, len(lags)))
        self.register_buffer("lead", torch.linspace(0, 1, config.horizon))
        self.channel = nn.Parameter(
            torch.randn(1, config.future_cond_dim, 1, width) * 0.02
        )
        self.encode = nn.Sequential(
            nn.Linear(5, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.query = nn.Sequential(
            nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.token = nn.Linear(width, config.d_model, bias=False)
        self.response = nn.Linear(width, 32, bias=False)
        nn.init.zeros_(self.token.weight)
        nn.init.zeros_(self.response.weight)

    def forward(self, past, transport):
        z = past[..., 1:]
        scale = z.var(1, unbiased=False, keepdim=True).add(1e-5).sqrt().clamp_min(0.1)
        z = (z - z.mean(1, keepdim=True)) / scale
        patches = z[:, self.indices].permute(0, 3, 1, 2)
        lagged = past[..., 1:][:, self.correlation_indices].permute(0, 1, 3, 2)
        target = past[:, self.correlation_indices[:, 0], :1, None]
        y0, z0 = target - target.mean(1, keepdim=True), lagged - lagged.mean(
            1, keepdim=True
        )
        corr = (y0 * z0).sum(1) / (
            y0.square().sum(1) * z0.square().sum(1) + 1e-8
        ).sqrt()
        inherited = torch.nn.functional.pad(
            transport.lag_logits,
            (0, self.lag_bias.shape[1] - transport.lag_logits.shape[1]),
        )
        prior = (
            (inherited + self.lag_bias + 2 * corr.abs()) / transport.temperature
        ).softmax(-1)
        features = torch.stack(
            (
                patches.mean(-1),
                patches[..., -1],
                patches[..., -1] - patches[..., 0],
                patches.var(-1, unbiased=False).add(1e-5).sqrt(),
                corr,
            ),
            -1,
        )
        bank = self.encode(features) + self.channel
        y = past[..., :1]
        ys = y.var(1, unbiased=False).add(1e-5).sqrt().clamp_min(0.1)
        recent = max(2, len(past[0]) // 4)
        trend = (y[:, -1] - y[:, -recent]) / ys
        roughness = y.diff(dim=1).square().mean(1).add(1e-5).sqrt() / ys
        lead = self.lead[None, :, None].expand(len(past), -1, -1)
        query = self.query(
            torch.cat(
                (
                    lead,
                    lead.square(),
                    trend[:, None].expand_as(lead),
                    roughness[:, None].expand_as(lead),
                ),
                -1,
            )
        )
        logits = torch.einsum("bhw,bckw->bhck", query, bank) / math.sqrt(bank.shape[-1])
        weights = (logits + prior.clamp_min(1e-8).log()[:, None]).softmax(-1)
        retrieved = torch.einsum("bhck,bckw->bhcw", weights, bank)
        # Keep a lead-dependent signal in StateResponse, and a channel token in
        # the global conditioner. Both updates are exactly zero at initialization.
        return self.token(retrieved.mean(1)), self.response(retrieved.mean(2))


class HistoryReliableResponse(StateResponse):
    """Bounded learned response gating, not a calibrated confidence estimate."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reliability = nn.Sequential(
            nn.Linear(self.width + 4, self.width), nn.SiLU(), nn.Linear(self.width, 1)
        )
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.zeros_(self.reliability[-1].bias)

    def forward(self, normalized_noisy, noise_level, features):
        original = features[..., : self.width]
        correction = super().forward(normalized_noisy, noise_level, original)
        gate = (
            2
            * self.reliability(
                torch.cat((original, features[..., self.width :]), -1)
            ).sigmoid()
        )
        return gate * correction


class HistoryNativeEDM(StateResponseEDM):
    """Both innovations retained; optional history-native refinements only."""

    def __init__(
        self,
        *args,
        history_condition="provided",
        history_period=24,
        history_decay=0.0,
        native_lag=False,
        reliable_response=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.history_mode = history_condition
        self.native_lag, self.reliable_response = native_lag, reliable_response
        with torch.random.fork_rng(devices=[]):
            if history_condition != "provided":
                self.history_conditioner = HistoryCondition(
                    self.config, history_condition, history_period, history_decay
                )
            if native_lag:
                self.exogenous.history_bank = LagHistoryBank(
                    self.config, self.exogenous.lags, history_period
                )
            if reliable_response:
                old = self.state_response
                self.state_response = HistoryReliableResponse(
                    self.config,
                    width=old.width,
                    gain_scale=old.gain_scale,
                    shift_scale=old.shift_scale,
                    recent_fraction=old.recent_fraction,
                    noise_damping=old.noise_damping,
                )
                self.state_response.load_state_dict(old.state_dict(), strict=False)

    def reliability_features(self, past, path):
        y, z = past[..., :1], past[..., 1:]
        recent = max(2, past.shape[1] // 4)
        agreement = 1 - 0.5 * (
            correlation(y, z) - correlation(y[:, -recent:], z[:, -recent:])
        ).abs().mean(-1)
        _, weights, _, _ = self.exogenous.lag_features(past, path)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(-1).mean(
            -1
        ) / math.log(max(2, weights.shape[-1]))
        scale = z.var(1, unbiased=False).add(1e-5).sqrt().clamp_min(0.1)
        drift = ((z[:, -recent:].mean(1) - z.mean(1)) / scale).abs().mean(-1).tanh()
        lead = torch.linspace(
            0, 1, self.config.horizon, device=past.device, dtype=past.dtype
        )
        state = torch.stack((agreement, entropy, drift), -1)[:, None].expand(
            -1, len(lead), -1
        )
        return torch.cat((state, lead[None, :, None].expand(len(past), -1, -1)), -1)

    def condition(self, past, future_condition, base_target):
        # Future values (including poisoned NaNs) cannot enter either branch.
        path = (
            past.new_zeros(len(past), self.config.horizon, self.config.future_cond_dim)
            if self.history_mode == "provided"
            else self.history_conditioner(past)
        )
        contexts, guide = super().condition(past, path, base_target)
        contexts = list(contexts)
        if self.native_lag:
            tokens, response = self.exogenous.history_bank(past, self.exogenous)
            begin, end = (
                self.config.past_dim,
                self.config.past_dim + self.config.future_cond_dim,
            )
            c = contexts[0]
            contexts[0] = torch.cat(
                (c[:, :begin], c[:, begin:end] + tokens, c[:, end:]), 1
            )
            contexts[-1] = contexts[-1] + response
        if self.reliable_response:
            contexts[-1] = torch.cat(
                (contexts[-1], self.reliability_features(past, path)), -1
            )
        return tuple(contexts), guide

    # Reuse exactly the IID Heun solver, with gradients enabled during refinement.


class HistoryAdaptiveEDM(HistoryNativeEDM):
    def __init__(self, *args, history_trust=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_trust_enabled = history_trust
        if history_trust:
            with torch.random.fork_rng(devices=[]):
                self.history_trust = nn.Sequential(
                    nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 1)
                )
                nn.init.zeros_(self.history_trust[-1].weight)
                nn.init.zeros_(self.history_trust[-1].bias)

    def condition(self, past, future_condition, base_target):
        context, guide = super().condition(past, future_condition, base_target)
        if not self.history_trust_enabled:
            return context, guide
        path = (
            past.new_zeros(len(past), self.config.horizon, self.config.future_cond_dim)
            if self.history_mode == "provided"
            else self.history_conditioner(past)
        )
        gate = 2 * self.history_trust(self.reliability_features(past, path)).sigmoid()
        c = context[0]
        begin, end = (
            self.config.past_dim,
            self.config.past_dim + self.config.future_cond_dim,
        )
        c = torch.cat(
            (c[:, :begin], c[:, begin:end] * gate.mean(1, keepdim=True), c[:, end:]), 1
        )
        return (c, *context[1:], gate), guide

    def denoise(self, noisy, sigma, context):
        if not self.history_trust_enabled:
            return super().denoise(noisy, sigma, context)
        actual, gate = context[:-1], context[-1]
        prediction = super().denoise(noisy, sigma, actual)
        denominator = (sigma.square() + self.sigma_data**2).sqrt()
        correction = self.state_response(
            noisy / denominator, sigma.flatten().log() / 4, actual[-1]
        )
        # Gate=1 is exactly the previous model. Gate near zero ignores the
        # explicit exogenous response; historical target context remains intact.
        return (
            prediction + (gate - 1) * sigma * self.sigma_data / denominator * correction
        )


class MultiScaleLagMemory(nn.Module):
    """Observed lagged patches; retain lead structure in both existing branches."""

    def __init__(self, config, lags, width=24):
        super().__init__()
        self.horizon = config.horizon
        self.scales = tuple(dict.fromkeys(min(n, config.seq_len) for n in (8, 32, 128)))
        for i, size in enumerate(self.scales):
            index = config.seq_len - 1 - lags[:, None] - torch.arange(size).flip(0)
            self.register_buffer("index_" + str(i), index.clamp_min(0))
        self.encode = nn.Sequential(
            nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.query = nn.Sequential(
            nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.token = nn.Linear(3 * width, config.d_model, bias=False)
        self.response = nn.Linear(width, 32, bias=False)
        nn.init.zeros_(self.token.weight)
        nn.init.zeros_(self.response.weight)

    def forward(self, past):
        z = past[..., 1:]
        z = (z - z.mean(1, keepdim=True)) / z.var(1, keepdim=True, unbiased=False).add(
            1e-5
        ).sqrt().clamp_min(0.1)
        banks = []
        for i in range(len(self.scales)):
            p = z[:, getattr(self, "index_" + str(i))].permute(0, 3, 1, 2)
            features = torch.stack(
                (
                    p.mean(-1),
                    p[..., -1],
                    p[..., -1] - p[..., 0],
                    p.var(-1, unbiased=False).add(1e-5).sqrt(),
                ),
                -1,
            )
            banks.append(self.encode(features))
        bank = torch.cat(banks, 2)
        y = past[..., :1]
        scale = y.var(1, unbiased=False).add(1e-5).sqrt().clamp_min(0.1)
        trend = (y[:, -1] - y[:, -max(2, past.shape[1] // 4)]) / scale
        roughness = y.diff(dim=1).square().mean(1).add(1e-5).sqrt() / scale
        lead = torch.linspace(0, 1, self.horizon, device=past.device)[
            None, :, None
        ].expand(len(past), -1, -1)
        query = self.query(
            torch.cat(
                (
                    lead,
                    lead.square(),
                    trend[:, None].expand_as(lead),
                    roughness[:, None].expand_as(lead),
                ),
                -1,
            )
        )
        weights = (
            torch.einsum("bhw,bckw->bhck", query, bank) / math.sqrt(bank.shape[-1])
        ).softmax(-1)
        retrieved = torch.einsum("bhck,bckw->bhcw", weights, bank)
        # Near/middle/far summaries stay separate, unlike a single horizon mean.
        summaries = [part.mean(1) for part in torch.tensor_split(retrieved, 3, dim=1)]
        return self.token(torch.cat(summaries, -1)), self.response(retrieved.mean(2))


class ChannelTrustAttention(nn.Linear):
    """Variable/lead/state-specific trust inside the existing response attention."""

    def __init__(self, width):
        super().__init__(width, 1)
        self.gate = nn.Sequential(nn.Linear(width, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, encoded):
        return (
            super().forward(encoded)
            + (2 * self.gate(encoded).sigmoid()).clamp_min(1e-6).log()
        )


class HistoryExploreEDM(HistoryAdaptiveEDM):
    def __init__(
        self, *args, multiscale_history=False, channel_response=False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.multiscale_history = multiscale_history
        with torch.random.fork_rng(devices=[]):
            if multiscale_history:
                self.exogenous.multiscale = MultiScaleLagMemory(
                    self.config, self.exogenous.lags
                )
            if channel_response:
                old = self.state_response.attention
                self.state_response.attention = ChannelTrustAttention(
                    self.state_response.width
                )
                self.state_response.attention.load_state_dict(
                    old.state_dict(), strict=False
                )

    def condition(self, past, future_condition, base_target):
        context, guide = super().condition(past, future_condition, base_target)
        if not self.multiscale_history:
            return context, guide
        token, response = self.exogenous.multiscale(past)
        context = list(context)
        begin = self.config.past_dim
        end = begin + self.config.future_cond_dim
        context[0] = torch.cat(
            (
                context[0][:, :begin],
                context[0][:, begin:end] + token,
                context[0][:, end:],
            ),
            1,
        )
        index = -2 if self.history_trust_enabled else -1
        features = context[index]
        context[index] = torch.cat(
            (features[..., :32] + response, features[..., 32:]), -1
        )
        return tuple(context), guide


class AccumulatedLagMemory(nn.Module):
    """Past-only accumulated exposures at explicit lag positions; no Z forecast."""

    def __init__(self, config, lags, width=24):
        super().__init__()
        self.horizon = config.horizon
        # Daily hydro windows, bounded by the benchmark lookback.
        self.windows = tuple(min(w, config.seq_len) for w in (3, 7, 30))
        for i, window in enumerate(self.windows):
            end = config.seq_len - lags
            start = (end - window).clamp_min(0)
            self.register_buffer("end_" + str(i), end)
            self.register_buffer("start_" + str(i), start)
        self.encode = nn.Sequential(
            nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.query = nn.Sequential(
            nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.token = nn.Linear(width, config.d_model, bias=False)
        self.response = nn.Linear(width, 32, bias=False)
        nn.init.zeros_(self.token.weight)
        nn.init.zeros_(self.response.weight)

    def forward(self, past):
        z = past[..., 1:]
        cumulative = torch.cat((z.new_zeros(len(z), 1, z.shape[-1]), z.cumsum(1)), 1)
        exposures = []
        for i in range(len(self.windows)):
            end, start = getattr(self, "end_" + str(i)), getattr(
                self, "start_" + str(i)
            )
            # Sum in train-prefix standardized units, scaled for numerical stability.
            exposures.append(
                (cumulative[:, end] - cumulative[:, start])
                / (end - start).clamp_min(1).sqrt()[None, :, None]
            )
        features = torch.stack(exposures, -1).permute(0, 2, 1, 3)
        bank = self.encode(features)
        y = past[..., :1]
        scale = y.var(1, unbiased=False).add(1e-5).sqrt().clamp_min(0.1)
        level = (y[:, -1] - y.mean(1)) / scale
        change = (y[:, -1] - y[:, -min(7, y.shape[1])]) / scale
        lead = torch.linspace(0, 1, self.horizon, device=past.device)[
            None, :, None
        ].expand(len(past), -1, -1)
        query = self.query(
            torch.cat(
                (lead, level[:, None].expand_as(lead), change[:, None].expand_as(lead)),
                -1,
            )
        )
        weights = (
            torch.einsum("bhw,bckw->bhck", query, bank) / math.sqrt(bank.shape[-1])
        ).softmax(-1)
        retrieved = torch.einsum("bhck,bckw->bhcw", weights, bank)
        return self.token(retrieved.mean(1)), self.response(retrieved.mean(2))


class HistoryGapEDM(HistoryExploreEDM):
    def __init__(
        self, *args, target_anchor="original", accumulated_history=False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.target_anchor = target_anchor
        self.accumulated_history = accumulated_history
        if accumulated_history:
            with torch.random.fork_rng(devices=[]):
                self.exogenous.accumulation = AccumulatedLagMemory(
                    self.config, self.exogenous.lags
                )

    def coordinates(self, past):
        mean, scale = super().coordinates(past)
        if self.target_anchor == "last":
            mean = past[:, -1:, : self.config.target_dim]
        elif self.target_anchor == "recent7":
            mean = past[:, -min(7, past.shape[1]) :, : self.config.target_dim].mean(
                1, keepdim=True
            )
        elif self.target_anchor != "original":
            raise ValueError(
                "Unknown history-only target anchor: " + self.target_anchor
            )
        return mean, scale

    def condition(self, past, future_condition, base_target):
        context, guide = super().condition(past, future_condition, base_target)
        if not self.accumulated_history:
            return context, guide
        tokens, response = self.exogenous.accumulation(past)
        context = list(context)
        start = self.config.past_dim
        end = start + self.config.future_cond_dim
        context[0] = torch.cat(
            (
                context[0][:, :start],
                context[0][:, start:end] + tokens,
                context[0][:, end:],
            ),
            1,
        )
        index = -2 if self.history_trust_enabled else -1
        features = context[index]
        context[index] = torch.cat(
            (features[..., :32] + response, features[..., 32:]), -1
        )
        return tuple(context), guide


class RecentLevelCondition(HistoryCondition):
    def forward(self, past):
        path = super().forward(past)
        if self.decay and self.level_window:
            z = past[..., 1:]
            lead = torch.arange(
                1, self.horizon + 1, device=past.device, dtype=past.dtype
            )
            confidence = (-lead / self.decay).exp()[None, :, None]
            shift = z[:, -self.level_window :].mean(1, keepdim=True) - z.mean(
                1, keepdim=True
            )
            path = path + (1 - confidence) * shift
        return path


class HistoryLocalEDM(HistoryGapEDM):
    def __init__(self, *args, anchor_mix=0.0, history_level_window=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor_mix = anchor_mix
        if history_level_window:
            old = self.history_conditioner
            with torch.random.fork_rng(devices=[]):
                current = RecentLevelCondition(
                    self.config, old.mode, old.period, old.decay
                )
            current.level_window = history_level_window
            current.load_state_dict(old.state_dict(), strict=True)
            self.history_conditioner = current

    def coordinates(self, past):
        mean, scale = super().coordinates(past)
        if self.anchor_mix:
            last = past[:, -1:, : self.config.target_dim]
            mean = (1 - self.anchor_mix) * mean + self.anchor_mix * last
        return mean, scale


class ChannelCycleCondition(RecentLevelCondition):
    """Keep repeatable phases; decay transient deviations using observed history."""

    def __init__(self, config, original, mix, decay, week_mix):
        super().__init__(config, original.mode, original.period, original.decay)
        self.level_window = getattr(original, "level_window", 0)
        self.mix, self.transient_decay, self.week_mix = mix, decay, week_mix
        self.load_state_dict(original.state_dict(), strict=True)

    def phase(self, z, period):
        period = min(period, z.shape[1] // 2)
        cycles = min(3, z.shape[1] // period)
        template = (
            z[:, -cycles * period :]
            .reshape(len(z), cycles, period, z.shape[-1])
            .mean(1)
        )
        template = template - template.mean(1, keepdim=True)
        first, second = z[:, period:], z[:, :-period]
        first = first - first.mean(1, keepdim=True)
        second = second - second.mean(1, keepdim=True)
        reliability = (
            (first * second).sum(1)
            / (first.square().sum(1) * second.square().sum(1) + 1e-8).sqrt()
        ).clamp(0, 1)[:, None]
        indices = torch.arange(self.horizon, device=z.device) % period
        return (
            template[:, indices] * reliability,
            template[:, -1:] * reliability,
            reliability,
        )

    def forward(self, past):
        original = super().forward(past)
        if not self.mix:
            return original
        z = past[..., 1:]
        daily, last_daily, reliability = self.phase(z, self.period)
        weekly, last_weekly, _ = self.phase(z, 7 * self.period)
        level = z[:, -min(7 * self.period, z.shape[1]) :].mean(1, keepdim=True)
        periodic = (1 - self.week_mix) * daily + self.week_mix * weekly
        last_periodic = (1 - self.week_mix) * last_daily + self.week_mix * last_weekly
        # Per-channel persistence is a history statistic, not future-Z supervision.
        tau = self.transient_decay * (0.5 + 1.5 * reliability)
        lead = torch.arange(1, self.horizon + 1, device=z.device, dtype=z.dtype)[
            None, :, None
        ]
        transient = (z[:, -1:] - level - last_periodic) * (-lead / tau).exp()
        return (1 - self.mix) * original + self.mix * (level + periodic + transient)


class BoundedStateShift(nn.Module):
    def __init__(self, scale, noise_cutoff, groups=None):
        super().__init__()
        self.scale, self.noise_cutoff = scale, noise_cutoff
        names = ("low_calm", "low_changing", "high_calm", "high_changing")
        self.groups = (
            tuple(names.index(g) for g in groups)
            if groups is not None
            else (0, 1, 2, 3)
        )
        self.net = nn.Sequential(nn.Linear(7, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, sigma):
        shift = self.scale * self.net(features).tanh()
        group = 2 * (features[..., 1] >= 0).long() + (features[..., 3] >= 0.25).long()
        mask = torch.stack([group == g for g in self.groups]).any(0)[..., None]
        shift = shift * mask
        return (
            shift / (1 + (sigma / self.noise_cutoff).square())
            if self.noise_cutoff
            else shift
        )


class GuardedHistoryEDM(HistoryLocalEDM):
    def __init__(
        self,
        *args,
        cycle_mix=0.0,
        cycle_decay=72.0,
        cycle_week_mix=0.0,
        state_shift=False,
        state_shift_scale=0.1,
        state_shift_cutoff=0.0,
        state_shift_groups=None,
        wind_index=1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.state_shift_enabled, self.wind_index = state_shift, wind_index
        with torch.random.fork_rng(devices=[]):
            if cycle_mix:
                self.history_conditioner = ChannelCycleCondition(
                    self.config,
                    self.history_conditioner,
                    cycle_mix,
                    cycle_decay,
                    cycle_week_mix,
                )
            if state_shift:
                self.state_response.history_shift = BoundedStateShift(
                    state_shift_scale, state_shift_cutoff, state_shift_groups
                )

    def condition(self, past, future_condition, base_target):
        context, guide = super().condition(past, future_condition, base_target)
        if not self.state_shift_enabled:
            return context, guide
        n = max(2, past.shape[1] // 8)
        y, z = past[..., :1], past[..., self.wind_index : self.wind_index + 1]
        ys = y.var(1, unbiased=False).add(1e-5).sqrt().clamp_min(0.1)
        zs = z.var(1, unbiased=False).add(1e-5).sqrt().clamp_min(0.1)
        features = torch.cat(
            (
                y[:, -1],
                y[:, -n:].mean(1),
                (y[:, -1] - y[:, -n]) / ys,
                y[:, -n:].diff(dim=1).square().mean(1).add(1e-5).sqrt() / ys,
                z[:, -n:].mean(1),
                (z[:, -1] - z[:, -n]) / zs,
            ),
            -1,
        )
        lead = torch.linspace(
            0, 1, self.config.horizon, device=past.device, dtype=past.dtype
        )
        features = torch.cat(
            (
                features[:, None].expand(-1, len(lead), -1),
                lead[None, :, None].expand(len(past), -1, -1),
            ),
            -1,
        )
        return (*context, features), guide

    def denoise(self, noisy, sigma, context):
        if not self.state_shift_enabled:
            return super().denoise(noisy, sigma, context)
        prediction = super().denoise(noisy, sigma, context[:-1])
        correction = self.state_response.history_shift(context[-1], sigma)
        return (
            prediction
            + sigma
            * self.sigma_data
            / (sigma.square() + self.sigma_data**2).sqrt()
            * correction
        )


class HistoryResponseBasis(nn.Module):
    """Low-rank joint Y/Z history projections; no independent forecasting model."""

    def __init__(self, config, period, rank=16, regimes=1):
        super().__init__()
        self.horizon, self.regimes = config.horizon, regimes
        self.sizes = tuple(
            dict.fromkeys(
                math.ceil(config.seq_len / s) for s in (1, max(2, period // 4), period)
            )
        )
        self.encoders = nn.ModuleList(
            nn.Sequential(nn.Linear(n * config.past_dim, rank), nn.SiLU())
            for n in self.sizes
        )
        self.outputs = nn.ModuleList(
            nn.Linear(rank, config.horizon * regimes) for _ in self.sizes
        )
        self.gate = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, regimes))
        for layer in self.outputs:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, history):
        values = []
        for size, encode, output in zip(self.sizes, self.encoders, self.outputs):
            pooled = F.adaptive_avg_pool1d(history.transpose(1, 2), size).flatten(1)
            values.append(
                output(encode(pooled)).reshape(len(history), self.horizon, self.regimes)
            )
        y = history[..., :1]
        n = max(2, len(history[0]) // 8)
        state = torch.cat(
            (
                y[:, -1],
                y[:, -n:].mean(1),
                y[:, -n:].var(1, unbiased=False).add(1e-5).sqrt(),
                y[:, -1] - y[:, -n],
            ),
            -1,
        )
        weights = self.gate(state).softmax(-1)[:, None]
        return (torch.stack(values).mean(0) * weights).sum(-1, keepdim=True)


class RescueHistoryEDM(GuardedHistoryEDM):
    def __init__(
        self,
        *args,
        response_basis=False,
        basis_rank=16,
        basis_regimes=1,
        basis_scale=0.5,
        basis_cutoff=0.0,
        basis_period=24,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.response_basis_enabled = response_basis
        self.basis_scale, self.basis_cutoff = basis_scale, basis_cutoff
        if response_basis:
            with torch.random.fork_rng(devices=[]):
                self.state_response.history_basis = HistoryResponseBasis(
                    self.config, basis_period, basis_rank, basis_regimes
                )

    def condition(self, past, future_condition, base_target):
        context, guide = super().condition(past, future_condition, base_target)
        if not self.response_basis_enabled:
            return context, guide
        mean, scale = self.coordinates(past)
        history = torch.cat(((past[..., :1] - mean) / scale, past[..., 1:]), -1)
        response = self.basis_scale * self.state_response.history_basis(history).tanh()
        return (*context, response), guide

    def denoise(self, noisy, sigma, context):
        if not self.response_basis_enabled:
            return super().denoise(noisy, sigma, context)
        prediction = super().denoise(noisy, sigma, context[:-1])
        response = context[-1]
        if self.basis_cutoff:
            response = response / (1 + (sigma / self.basis_cutoff).square())
        # This is a denoising response at every solver update, not sample editing.
        return (
            prediction
            + sigma
            * self.sigma_data
            / (sigma.square() + self.sigma_data**2).sqrt()
            * response
        )


class TemporalResponse(nn.Module):
    """Linear smooth/phase responses retain information lost by pooled state stats."""

    def __init__(self, config, period, mode="both", bins=24):
        super().__init__()
        self.horizon, self.period = config.horizon, min(period, config.seq_len)
        self.bins, self.mode = min(bins, config.seq_len), mode
        self.smooth = nn.Linear(self.bins * config.past_dim, min(bins, config.horizon))
        self.phase = nn.Linear(self.period * config.past_dim, self.period)
        for layer in (self.smooth, self.phase):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, history):
        pooled = F.adaptive_avg_pool1d(history.transpose(1, 2), self.bins).flatten(1)
        smooth = F.interpolate(
            self.smooth(pooled)[:, None],
            size=self.horizon,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        cycles = min(4, history.shape[1] // self.period)
        phase = (
            history[:, -cycles * self.period :]
            .reshape(len(history), cycles, self.period, -1)
            .mean(1)
        )
        phase = self.phase(phase.flatten(1))
        phase = phase - phase.mean(1, keepdim=True)
        phase = phase[
            :, torch.arange(self.horizon, device=history.device) % self.period, None
        ]
        if self.mode == "smooth":
            return smooth
        if self.mode == "phase":
            return phase
        return smooth + phase


class RankHistoryEDM(RescueHistoryEDM):
    def __init__(
        self,
        *args,
        temporal_response=False,
        temporal_mode="both",
        temporal_bins=24,
        temporal_scale=1.0,
        temporal_cutoff=0.0,
        temporal_period=24,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.temporal_response_enabled = temporal_response
        self.temporal_scale, self.temporal_cutoff = temporal_scale, temporal_cutoff
        if temporal_response:
            with torch.random.fork_rng(devices=[]):
                self.state_response.temporal = TemporalResponse(
                    self.config, temporal_period, temporal_mode, temporal_bins
                )

    def condition(self, past, future_condition, base_target):
        context, guide = super().condition(past, future_condition, base_target)
        if not self.temporal_response_enabled:
            return context, guide
        mean, scale = self.coordinates(past)
        history = torch.cat(((past[..., :1] - mean) / scale, past[..., 1:]), -1)
        return (
            *context,
            self.state_response.temporal(history) * self.temporal_scale,
        ), guide

    def denoise(self, noisy, sigma, context):
        if not self.temporal_response_enabled:
            return super().denoise(noisy, sigma, context)
        prediction = super().denoise(noisy, sigma, context[:-1])
        response = context[-1]
        if self.temporal_cutoff:
            response = response / (1 + (sigma / self.temporal_cutoff).square())
        return (
            prediction
            + sigma
            * self.sigma_data
            / (sigma.square() + self.sigma_data**2).sqrt()
            * response
        )


class PatchResponse(nn.Module):
    """Retrieve observed Y/Z patches with lead-specific exogenous queries."""

    def __init__(self, config, width=32, patch=12, period=24):
        super().__init__()
        self.patch, self.period, self.width = min(patch, config.seq_len), period, width
        self.encode = nn.Sequential(
            nn.Linear(self.patch * config.past_dim, width),
            nn.SiLU(),
            nn.LayerNorm(width),
        )
        self.query = nn.Linear(config.future_cond_dim + 5, width)
        self.position = nn.Linear(1, width)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3)
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def features(self, history, path):
        # Pad only the earliest historical value; no extrapolated/future observations.
        pad = (-history.shape[1]) % self.patch
        values = F.pad(history.transpose(1, 2), (pad, 0), mode="replicate").transpose(
            1, 2
        )
        bank = self.encode(
            values.reshape(len(values), -1, self.patch * values.shape[-1])
        )
        position = torch.linspace(
            -1, 0, bank.shape[1], device=bank.device, dtype=bank.dtype
        )
        bank = bank + self.position(position[None, :, None])
        lead = torch.arange(1, path.shape[1] + 1, device=path.device, dtype=path.dtype)
        phase = 2 * math.pi * lead / self.period
        timing = torch.stack(
            (
                lead / len(lead),
                phase.sin(),
                phase.cos(),
                (phase / 7).sin(),
                (phase / 7).cos(),
            ),
            -1,
        )
        query = self.query(
            torch.cat((path, timing[None].expand(len(path), -1, -1)), -1)
        )
        weights = (query @ bank.transpose(1, 2) / math.sqrt(self.width)).softmax(-1)
        return self.norm(query + weights @ bank + bank.mean(1, keepdim=True))

    def forward(self, noisy, sigma, features, scale, gain, curve, cutoff):
        noise = sinusoidal_timestep_embedding(sigma.flatten().log() / 4, self.width)[
            :, None
        ]
        shift, slope, skew = self.output(features + noise).tanh().chunk(3, -1)
        normalized = noisy / (sigma.square() + 1).sqrt()
        correction = (
            scale * shift
            + gain * slope * normalized
            + curve * skew * (normalized.square() - 1).tanh()
        )
        if cutoff:
            correction = correction / (1 + (sigma / cutoff).square())
        return sigma / (sigma.square() + 1).sqrt() * correction


class SwitchHistoryEDM(RankHistoryEDM):
    def __init__(
        self,
        *args,
        patch_response=False,
        response_patch=12,
        response_width=32,
        response_period=24,
        patch_scale=0.5,
        patch_gain=0.0,
        patch_curve=0.0,
        patch_cutoff=0.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.patch_response_enabled = patch_response
        self.patch_options = (patch_scale, patch_gain, patch_curve, patch_cutoff)
        if patch_response:
            with torch.random.fork_rng(devices=[]):
                self.state_response.patch_response = PatchResponse(
                    self.config, response_width, response_patch, response_period
                )

    def condition(self, past, future_condition, base_target):
        context, guide = super().condition(past, future_condition, base_target)
        if not self.patch_response_enabled:
            return context, guide
        mean, scale = self.coordinates(past)
        history = torch.cat(((past[..., :1] - mean) / scale, past[..., 1:]), -1)
        path = (
            past.new_zeros(len(past), self.config.horizon, self.config.future_cond_dim)
            if self.history_mode == "provided"
            else self.history_conditioner(past)
        )
        aligned = self.exogenous.transport_path(past, path)
        features = self.state_response.patch_response.features(history, aligned)
        return (*context, features), guide

    def denoise(self, noisy, sigma, context):
        if not self.patch_response_enabled:
            return super().denoise(noisy, sigma, context)
        prediction = super().denoise(noisy, sigma, context[:-1])
        return prediction + self.state_response.patch_response(
            noisy, sigma, context[-1], *self.patch_options
        )


class MultiScaleResponse(nn.Module):
    def __init__(self, config, bins=(6, 24, 72), width=48, period=24, dropout=0.0):
        super().__init__()
        self.bins = tuple(min(b, config.seq_len) for b in bins)
        self.horizon, self.period, self.width = (
            config.horizon,
            min(period, config.seq_len),
            width,
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * b * config.past_dim, width),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(width, 2 * min(b, config.horizon)),
                )
                for b in self.bins
            ]
        )
        self.phase = nn.Sequential(
            nn.Linear(2 * self.period * config.past_dim, width),
            nn.SiLU(),
            nn.Linear(width, 2 * self.period),
        )
        self.context = nn.Linear(
            3 * config.past_dim + 2 * config.future_cond_dim, width
        )
        self.gate = nn.Sequential(nn.SiLU(), nn.Linear(width, len(bins) + 1))
        for head in [*self.heads, self.phase]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def features(self, history, aligned):
        channels = history.transpose(1, 2)
        differences = F.pad(channels.diff(dim=-1), (1, 0))
        responses = []
        for b, head in zip(self.bins, self.heads):
            x = torch.cat(
                (
                    F.adaptive_avg_pool1d(channels, b),
                    F.adaptive_avg_pool1d(differences, b),
                ),
                1,
            ).flatten(1)
            y = head(x).reshape(len(x), 2, -1)
            responses.append(
                F.interpolate(
                    y, size=self.horizon, mode="linear", align_corners=False
                ).transpose(1, 2)
            )
        cycles = min(4, history.shape[1] // self.period)
        phase = (
            history[:, -cycles * self.period :]
            .reshape(len(history), cycles, self.period, -1)
            .mean(1)
        )
        x = torch.cat((phase, history[:, -self.period :] - phase), -1).flatten(1)
        phase = self.phase(x).reshape(len(x), 2, self.period).transpose(1, 2)
        responses.append(
            phase[:, torch.arange(self.horizon, device=history.device) % self.period]
        )
        state = torch.cat(
            (
                history.mean(1),
                history.std(1, unbiased=False),
                history[:, -1],
                aligned.mean(1),
                aligned.std(1, unbiased=False),
            ),
            -1,
        )
        # Keep all context entries as tensors, as required by batched IID sampling.
        return torch.stack(responses, -2), self.context(state)

    def forward(self, noisy, sigma, responses, state, shift, gain, cutoff):
        noise = sinusoidal_timestep_embedding(sigma.flatten().log() / 4, self.width)
        weights = self.gate(state + noise).softmax(-1)[:, None, :, None]
        offset, slope = (responses * weights).sum(-2).tanh().chunk(2, -1)
        correction = shift * offset + gain * slope * noisy / (sigma.square() + 1).sqrt()
        if cutoff:
            correction = correction / (1 + (sigma / cutoff).square())
        return sigma / (sigma.square() + 1).sqrt() * correction


class BroadHistoryEDM(SwitchHistoryEDM):
    def __init__(
        self,
        *args,
        multiscale_response=False,
        multiscale_bins=(6, 24, 72),
        multiscale_width=48,
        multiscale_period=24,
        multiscale_dropout=0.0,
        multiscale_shift=0.5,
        multiscale_gain=0.1,
        multiscale_cutoff=0.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.multiscale_enabled = multiscale_response
        self.multiscale_options = multiscale_shift, multiscale_gain, multiscale_cutoff
        if multiscale_response:
            with torch.random.fork_rng(devices=[]):
                self.state_response.multiscale = MultiScaleResponse(
                    self.config,
                    multiscale_bins,
                    multiscale_width,
                    multiscale_period,
                    multiscale_dropout,
                )

    def condition(self, past, future_condition, base_target):
        context, guide = super().condition(past, future_condition, base_target)
        if not self.multiscale_enabled:
            return context, guide
        mean, scale = self.coordinates(past)
        history = torch.cat(((past[..., :1] - mean) / scale, past[..., 1:]), -1)
        path = (
            past.new_zeros(len(past), self.config.horizon, self.config.future_cond_dim)
            if self.history_mode == "provided"
            else self.history_conditioner(past)
        )
        aligned = self.exogenous.transport_path(past, path)
        responses, state = self.state_response.multiscale.features(history, aligned)
        return (*context, responses, state), guide

    def denoise(self, noisy, sigma, context):
        if not self.multiscale_enabled:
            return super().denoise(noisy, sigma, context)
        prediction = super().denoise(noisy, sigma, context[:-2])
        return prediction + self.state_response.multiscale(
            noisy, sigma, *context[-2:], *self.multiscale_options
        )
