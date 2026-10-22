"""History-only conditional representations; future Z is never read."""
import math
import torch
from torch import nn
from .state_response import StateResponse, StateResponseEDM, correlation

class HistoryCondition(nn.Module):
    def __init__(self, config, mode, period, decay=0.):
        super().__init__()
        self.mode, self.horizon = mode, config.horizon
        self.decay = decay
        self.period = min(period, config.seq_len)
        self.register_buffer("indices", config.seq_len-self.period + torch.arange(config.horizon) % self.period)
        if mode == "linear":
            self.residual = nn.Sequential(nn.Linear(config.seq_len, 32), nn.GELU(), nn.Linear(32, config.horizon))
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
            path = torch.stack([exogenous[:, self.indices - i * self.period]
                                for i in range(cycles)]).mean(0)
        if self.mode == "linear":
            mean = exogenous.mean(1, keepdim=True)
            scale = exogenous.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt().clamp_min(.1)
            residual = self.residual(((exogenous-mean)/scale).transpose(1, 2)).transpose(1, 2)
            path = path + scale * residual.tanh()
        if self.decay:
            lead = torch.arange(1, self.horizon + 1, device=past.device, dtype=past.dtype)
            confidence = (-lead / self.decay).exp()[None, :, None]
            level = exogenous.mean(1, keepdim=True)
            path = level + confidence * (path - level)
        return path


class LagHistoryBank(nn.Module):
    """Lead-dependent retrieval from observed patches at the existing lag grid."""

    def __init__(self, config, lags, period=24, width=24, patch=8):
        super().__init__()
        values = lags.tolist()
        values += [v for v in (period, 7 * period, config.seq_len // 2)
                   if 0 < v <= config.seq_len // 2 and v not in values]
        lags = torch.tensor(list(dict.fromkeys(values)))
        indices = config.seq_len - 1 - lags[:, None] - torch.arange(min(patch, config.seq_len)).flip(0)
        self.register_buffer("indices", indices.clamp_min(0))
        self.register_buffer("correlation_indices", torch.arange(int(lags.max()), config.seq_len)[:, None] - lags)
        self.lag_bias = nn.Parameter(torch.zeros(config.future_cond_dim, len(lags)))
        self.register_buffer("lead", torch.linspace(0, 1, config.horizon))
        self.channel = nn.Parameter(torch.randn(1, config.future_cond_dim, 1, width) * .02)
        self.encode = nn.Sequential(nn.Linear(5, width), nn.SiLU(), nn.Linear(width, width))
        self.query = nn.Sequential(nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width))
        self.token = nn.Linear(width, config.d_model, bias=False)
        self.response = nn.Linear(width, 32, bias=False)
        nn.init.zeros_(self.token.weight)
        nn.init.zeros_(self.response.weight)

    def forward(self, past, transport):
        z = past[..., 1:]
        scale = z.var(1, unbiased=False, keepdim=True).add(1e-5).sqrt().clamp_min(.1)
        z = (z - z.mean(1, keepdim=True)) / scale
        patches = z[:, self.indices].permute(0, 3, 1, 2)
        lagged = past[..., 1:][:, self.correlation_indices].permute(0, 1, 3, 2)
        target = past[:, self.correlation_indices[:, 0], :1, None]
        y0, z0 = target - target.mean(1, keepdim=True), lagged - lagged.mean(1, keepdim=True)
        corr = (y0 * z0).sum(1) / (y0.square().sum(1) * z0.square().sum(1) + 1e-8).sqrt()
        inherited = torch.nn.functional.pad(transport.lag_logits, (0, self.lag_bias.shape[1] - transport.lag_logits.shape[1]))
        prior = ((inherited + self.lag_bias + 2 * corr.abs()) / transport.temperature).softmax(-1)
        features = torch.stack((patches.mean(-1), patches[..., -1],
                                patches[..., -1] - patches[..., 0],
                                patches.var(-1, unbiased=False).add(1e-5).sqrt(), corr), -1)
        bank = self.encode(features) + self.channel
        y = past[..., :1]
        ys = y.var(1, unbiased=False).add(1e-5).sqrt().clamp_min(.1)
        recent = max(2, len(past[0]) // 4)
        trend = (y[:, -1] - y[:, -recent]) / ys
        roughness = y.diff(dim=1).square().mean(1).add(1e-5).sqrt() / ys
        lead = self.lead[None, :, None].expand(len(past), -1, -1)
        query = self.query(torch.cat((lead, lead.square(),
            trend[:, None].expand_as(lead), roughness[:, None].expand_as(lead)), -1))
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
        self.reliability = nn.Sequential(nn.Linear(self.width + 4, self.width), nn.SiLU(), nn.Linear(self.width, 1))
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.zeros_(self.reliability[-1].bias)

    def forward(self, normalized_noisy, noise_level, features):
        original = features[..., :self.width]
        correction = super().forward(normalized_noisy, noise_level, original)
        gate = 2 * self.reliability(torch.cat((original, features[..., self.width:]), -1)).sigmoid()
        return gate * correction


class HistoryNativeEDM(StateResponseEDM):
    """Both innovations retained; optional history-native refinements only."""

    def __init__(self, *args, history_condition="provided", history_period=24,
                 history_decay=0., native_lag=False, reliable_response=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_mode = history_condition
        self.native_lag, self.reliable_response = native_lag, reliable_response
        with torch.random.fork_rng(devices=[]):
            if history_condition != "provided":
                self.history_conditioner = HistoryCondition(self.config, history_condition, history_period, history_decay)
            if native_lag:
                self.exogenous.history_bank = LagHistoryBank(self.config, self.exogenous.lags, history_period)
            if reliable_response:
                old = self.state_response
                self.state_response = HistoryReliableResponse(self.config, width=old.width,
                    gain_scale=old.gain_scale, shift_scale=old.shift_scale,
                    recent_fraction=old.recent_fraction, noise_damping=old.noise_damping)
                self.state_response.load_state_dict(old.state_dict(), strict=False)

    def reliability_features(self, past, path):
        y, z = past[..., :1], past[..., 1:]
        recent = max(2, past.shape[1] // 4)
        agreement = 1 - .5 * (correlation(y, z) - correlation(y[:, -recent:], z[:, -recent:])).abs().mean(-1)
        _, weights, _, _ = self.exogenous.lag_features(past, path)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(-1).mean(-1) / math.log(max(2, weights.shape[-1]))
        scale = z.var(1, unbiased=False).add(1e-5).sqrt().clamp_min(.1)
        drift = ((z[:, -recent:].mean(1) - z.mean(1)) / scale).abs().mean(-1).tanh()
        lead = torch.linspace(0, 1, self.config.horizon, device=past.device, dtype=past.dtype)
        state = torch.stack((agreement, entropy, drift), -1)[:, None].expand(-1, len(lead), -1)
        return torch.cat((state, lead[None, :, None].expand(len(past), -1, -1)), -1)

    def condition(self, past, future_condition, base_target):
        # Future values (including poisoned NaNs) cannot enter either branch.
        path = (past.new_zeros(len(past), self.config.horizon, self.config.future_cond_dim)
                if self.history_mode == "provided" else self.history_conditioner(past))
        contexts, guide = super().condition(past, path, base_target)
        contexts = list(contexts)
        if self.native_lag:
            tokens, response = self.exogenous.history_bank(past, self.exogenous)
            begin, end = self.config.past_dim, self.config.past_dim + self.config.future_cond_dim
            c = contexts[0]
            contexts[0] = torch.cat((c[:, :begin], c[:, begin:end] + tokens, c[:, end:]), 1)
            contexts[-1] = contexts[-1] + response
        if self.reliable_response:
            contexts[-1] = torch.cat((contexts[-1], self.reliability_features(past, path)), -1)
        return tuple(contexts), guide

    # Reuse exactly the IID Heun solver, with gradients enabled during refinement.
