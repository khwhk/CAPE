"""History-guided soft lag alignment of exogenous conditions."""
import torch
from torch import nn

class LagTransport(nn.Module):
    """Align exogenous trajectory tokens, without an extra target-output head."""

    def __init__(self, config, preserve_norm=False, detrend=False, lag_candidates=None):
        super().__init__()
        self.channels = config.future_cond_dim
        self.preserve_norm = preserve_norm
        self.detrend = detrend
        self.temperature = 1.
        self.recent_fraction = 1.
        self.uniform_mix = 0.
        lags = torch.tensor(
            [v for v in (lag_candidates or (0, 1, 3, 6, 12, 24)) if v <= config.seq_len // 2]
        )
        self.register_buffer("lags", lags)
        self.register_buffer(
            "indices", config.seq_len + torch.arange(config.horizon)[:, None] - lags
        )
        self.register_buffer(
            "history_indices",
            torch.arange(int(lags.max()), config.seq_len)[:, None] - lags,
        )
        self.lag_logits = nn.Parameter(torch.zeros(self.channels, len(lags)))
        self.projection = nn.Linear(config.horizon, config.d_model, bias=False)
        nn.init.zeros_(self.projection.weight)

    def lag_features(self, past, future):
        target = past[:, self.history_indices[:, 0], :1, None]
        lagged = past[..., 1:][:, self.history_indices].permute(0, 1, 3, 2)
        if self.detrend:
            target, lagged = target.diff(dim=1), lagged.diff(dim=1)
        def correlate(y, z):
            y, z = y - y.mean(1, keepdim=True), z - z.mean(1, keepdim=True)
            return (y * z).sum(1) / (y.square().sum(1) * z.square().sum(1) + 1e-8).sqrt()
        correlation = correlate(target, lagged)
        if self.recent_fraction < 1:
            recent = max(2, int(target.shape[1] * self.recent_fraction))
            correlation = .5 * (correlation + correlate(target[:, -recent:], lagged[:, -recent:]))
        correlation = correlation.clamp(-1, 1)
        weights = ((self.lag_logits + 2 * correlation.abs()) / self.temperature).softmax(-1)
        if self.uniform_mix:
            weights = (1 - self.uniform_mix) * weights + self.uniform_mix / weights.shape[-1]
        scales = (
            past.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt().clamp_min(0.1)
        )
        path = torch.cat((past[..., 1:], future), 1)
        shifted = path[:, self.indices].permute(0, 1, 3, 2)
        return correlation, weights, scales, shifted

    def transport_path(self, past, future):
        correlation, weights, _, shifted = self.lag_features(past, future)
        aligned = (shifted * weights[:, None]).sum(-1)
        return future + correlation.abs().amax(-1)[:, None] * (aligned - future)

    def forward(self, context, past, future):
        correlation, weights, scales, shifted = self.lag_features(past, future)
        # Add only displacement from the unshifted future path, not a Y forecast.
        displacement = (shifted - future[..., None]) / scales[..., 1:, None]
        displacement = (displacement * weights[:, None]).sum(-1) * scales[..., :1]
        displacement = displacement * correlation.abs().amax(-1)[:, None]
        adjustment = self.projection(displacement.transpose(1, 2))
        original = context[:, -self.channels :]
        transported = original + adjustment
        if self.preserve_norm:
            transported = transported * (
                original.norm(dim=-1, keepdim=True)
                / transported.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            )
        return torch.cat((context[:, : -self.channels], transported), 1)
