import torch
from torch import nn
from torch.nn import functional as F
from .embedding import sinusoidal_timestep_embedding

class TrajectoryConditioner(nn.Module):
    """Channel trajectory tokens with history-anchored covariate alignment."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.past = nn.Linear(config.seq_len, config.d_model)
        self.future = nn.Linear(config.horizon, config.d_model)
        self.absolute_future = nn.Linear(config.horizon, config.d_model)
        self.position = nn.Parameter(
            torch.randn(1, config.past_dim + config.future_cond_dim, config.d_model)
            * 0.02
        )

    def forward(self, past, future):
        mean = past.mean(1, keepdim=True)
        scale = (
            past.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt().clamp_min(0.1)
        )
        target_mean, target_scale = mean[..., :1], scale[..., :1]
        aligned_past = (past - mean) / scale * target_scale + target_mean
        aligned_future = (future - mean[..., 1:]) / scale[
            ..., 1:
        ] * target_scale + target_mean
        history = self.past(aligned_past.transpose(1, 2))
        forecast = self.future(aligned_future.transpose(1, 2)) + self.absolute_future(
            future.transpose(1, 2)
        )
        context = torch.cat((history, forecast), 1) + self.position
        return context, past.new_zeros(
            len(past), self.config.horizon, self.config.target_dim
        )


class TrajectoryNoiseTransformer(nn.Module):
    """A noisy target-trajectory token attends to exogenous trajectory tokens."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.target = nn.Linear(config.horizon, config.d_model)
        self.time = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        layer = nn.TransformerEncoderLayer(
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, config.n_layers, norm=nn.LayerNorm(config.d_model)
        )
        self.output = nn.Linear(config.d_model, config.horizon)

    def forward(self, noisy_target, guide, context, timesteps):
        time = self.time(
            sinusoidal_timestep_embedding(timesteps, self.config.d_model)
        ).unsqueeze(1)
        target = self.target(noisy_target.transpose(1, 2)) + time
        encoded = self.encoder(torch.cat((context, target), 1))
        return self.output(encoded[:, -self.config.target_dim :]).transpose(1, 2)

def patchify(value, patch_length):
    padding = (-value.shape[1]) % patch_length
    value = F.pad(value.transpose(1, 2), (0, padding), mode="replicate").transpose(1, 2)
    return value.reshape(len(value), -1, patch_length * value.shape[-1])


def history_statistics(past):
    target = past[..., :1]
    return torch.cat((
        target[:, -1], target.mean(1),
        target.var(1, unbiased=False).add(1e-5).sqrt(),
        target[:, -1] - target[:, 0],
    ), -1)


class DualScaleDenoiser(nn.Module):
    def __init__(self, config, global_denoiser, patch_length):
        super().__init__()
        self.global_denoiser = global_denoiser
        self.config, self.patch_length = config, patch_length
        self.target = nn.Linear(patch_length, config.d_model)
        count = (config.seq_len + patch_length - 1) // patch_length
        count += (config.horizon + patch_length - 1) // patch_length
        self.position = nn.Parameter(torch.randn(1, count, config.d_model) * 0.02)
        self.time = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.SiLU())
        layer = nn.TransformerEncoderLayer(
            config.d_model, config.n_heads, config.d_ff, config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, config.n_layers, norm=nn.LayerNorm(config.d_model)
        )
        self.output = nn.Linear(config.d_model, patch_length)
        # The local correction starts at zero; the global branch remains active.
        self.local_gate = nn.Parameter(torch.zeros(()))

    def forward(self, noisy, guide, context, timesteps):
        global_context, history, future = context
        global_prediction = self.global_denoiser(noisy, guide, global_context, timesteps)
        target = self.target(patchify(noisy, self.patch_length)) + future
        time = self.time(sinusoidal_timestep_embedding(timesteps, self.config.d_model))[:, None]
        tokens = torch.cat((history, target), 1) + self.position + time
        hidden = self.encoder(tokens)[:, history.shape[1]:]
        local = self.output(hidden).reshape(len(noisy), -1, 1)[:, :self.config.horizon]
        return global_prediction + self.local_gate.tanh() * local
