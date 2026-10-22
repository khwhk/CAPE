"""State- and noise-conditioned response inside every denoising update."""
import torch
from torch import nn
from .embedding import sinusoidal_timestep_embedding
from .edm import ExtendedForecastEDM

def correlation(target, covariates):
    target = target - target.mean(1, keepdim=True)
    covariates = covariates - covariates.mean(1, keepdim=True)
    return (target * covariates).sum(1) / (
        target.square().sum(1) * covariates.square().sum(1) + 1e-8
    ).sqrt()


class StateResponse(nn.Module):
    def __init__(self, config, width=32, gain_scale=1.0, shift_scale=.5,
                 recent_fraction=.25, noise_damping=0.):
        super().__init__()
        self.width = width
        self.gain_scale = gain_scale
        self.shift_scale = shift_scale
        self.recent_fraction = recent_fraction
        self.noise_damping = noise_damping
        self.channel = nn.Parameter(torch.randn(1, 1, config.future_cond_dim, width) * .02)
        self.exogenous = nn.Sequential(nn.Linear(8, width), nn.SiLU(), nn.Linear(width, width))
        self.state = nn.Sequential(nn.Linear(7, width), nn.SiLU(), nn.Linear(width, 2 * width))
        self.attention = nn.Linear(width, 1)
        self.noise = nn.Sequential(nn.Linear(width, width), nn.SiLU())
        self.response = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))
        nn.init.zeros_(self.response[-1].weight)
        nn.init.zeros_(self.response[-1].bias)

    def features(self, past, future):
        target, exogenous = past[..., :1], past[..., 1:]
        recent = max(2, int(past.shape[1] * self.recent_fraction))
        mean = past.mean(1, keepdim=True)
        scale = past.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt().clamp_min(.1)
        recent_mean = past[:, -recent:].mean(1, keepdim=True)
        recent_scale = past[:, -recent:].var(1, keepdim=True, unbiased=False).add(1e-5).sqrt().clamp_min(.1)
        ym, ys, zm, zs = mean[..., :1], scale[..., :1], mean[..., 1:], scale[..., 1:]
        full_corr = correlation(target, exogenous)[:, None].expand_as(future)
        recent_corr = correlation(target[:, -recent:], exogenous[:, -recent:])[:, None].expand_as(future)
        previous = torch.cat((exogenous[:, -1:], future[:, :-1]), 1)
        lead = torch.linspace(0, 1, future.shape[1], device=past.device, dtype=past.dtype)[None, :, None]
        covariates = torch.stack((
            future, (future - zm) / zs, (future - previous) / zs,
            full_corr, recent_corr,
            ((recent_mean[..., 1:] - zm) / zs).expand_as(future),
            (recent_scale[..., 1:] / zs).log().expand_as(future),
            lead.expand_as(future),
        ), -1)
        state = torch.cat((
            ym, ys.log(), (target[:, -1:] - ym) / ys,
            (recent_mean[..., :1] - ym) / ys,
            (recent_scale[..., :1] / ys).log(),
            (target[:, -1:] - target[:, -recent:-recent + 1]) / ys,
            target.diff(dim=1).square().mean(1, keepdim=True).add(1e-5).sqrt() / ys,
        ), -1)
        gain, shift = self.state(state).chunk(2, -1)
        encoded = self.exogenous(covariates) + self.channel
        encoded = torch.tanh(encoded * (1 + .5 * gain[:, :, None].tanh()) + shift[:, :, None])
        weights = self.attention(encoded).softmax(2)
        return (weights * encoded).sum(2)

    def forward(self, normalized_noisy, noise_level, features):
        noise = self.noise(sinusoidal_timestep_embedding(noise_level, self.width))[:, None]
        shift, gain = self.response(features + noise).chunk(2, -1)
        # Acts before every solver update, never by rescaling completed samples.
        gain_correction = .5 * self.gain_scale * gain.tanh() * normalized_noisy
        if self.noise_damping:
            sigma = (4 * noise_level).exp()[:, None, None]
            gain_correction = gain_correction / (1 + self.noise_damping * sigma)
        return self.shift_scale * shift.tanh() + gain_correction


class StateResponseEDM(ExtendedForecastEDM):
    def __init__(self, *args, response_input="aligned", response_gain_scale=1.0,
                 coordinate_center="baseline", response_shift_scale=.5,
                 response_recent_fraction=.25, response_noise_damping=0.,
                 lag_temperature=1., lag_recent_fraction=1., lag_uniform_mix=0., **kwargs):
        super().__init__(*args, **kwargs)
        if response_input not in ("aligned", "raw", "aligned_detach"):
            raise ValueError("Unknown state-response input")
        self.response_input = response_input
        self.coordinate_center = coordinate_center
        if self.exogenous_mode != "none":
            self.exogenous.temperature = lag_temperature
            self.exogenous.recent_fraction = lag_recent_fraction
            self.exogenous.uniform_mix = lag_uniform_mix
        # Matched seeds keep backbone initialization and subsequent RNG identical.
        with torch.random.fork_rng(devices=[]):
            self.state_response = StateResponse(self.config, gain_scale=response_gain_scale,
                shift_scale=response_shift_scale, recent_fraction=response_recent_fraction,
                noise_damping=response_noise_damping)

    def coordinates(self, past):
        mean, scale = super().coordinates(past)
        if self.coordinate_center == "last":
            mean = past[:, -1:, :self.config.target_dim]
        return mean, scale

    def condition(self, past, future_condition, base_target):
        context, guide = super().condition(past, future_condition, base_target)
        response_path = (self.exogenous.transport_path(past, future_condition)
                         if self.exogenous_mode != "none" and self.response_input != "raw"
                         else future_condition)
        if self.response_input == "aligned_detach":
            response_path = response_path.detach()
        features = self.state_response.features(past, response_path)
        context = context if isinstance(context, tuple) else (context,)
        return (*context, features), guide

    def denoise(self, noisy, sigma, context):
        backbone_context = context[:-1] if self.edm_backbone == "dual" else context[0]
        prediction = super().denoise(noisy, sigma, backbone_context)
        denominator = (sigma.square() + self.sigma_data ** 2).sqrt()
        correction = self.state_response(noisy / denominator, sigma.flatten().log() / 4, context[-1])
        return prediction + sigma * self.sigma_data / denominator * correction
