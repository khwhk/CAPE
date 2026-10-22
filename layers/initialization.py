"""Preserve the original seeded initialization before replacing temporary layers.
These temporary modules are discarded; they are not prediction branches.
"""
import torch
from torch import nn
from .embedding import TransformerDiffusionConfig

class _ConditionTransformer(nn.Module):
    def __init__(self, config: TransformerDiffusionConfig) -> None:
        super(_ConditionTransformer, self).__init__()
        self.config = config
        self.past_projection = nn.Linear(config.past_dim, config.d_model)
        self.future_projection = nn.Linear(config.future_cond_dim, config.d_model)
        self.past_position = nn.Parameter(
            torch.zeros(1, config.seq_len, config.d_model)
        )
        self.future_position = nn.Parameter(
            torch.zeros(1, config.horizon, config.d_model)
        )
        self.past_type = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.future_type = nn.Parameter(torch.zeros(1, 1, config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.n_layers, norm=nn.LayerNorm(config.d_model)
        )
        self.guide_head = nn.Linear(config.d_model, config.target_dim)
        self.patch_location = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.past_position, std=0.02)
        nn.init.normal_(self.future_position, std=0.02)
        nn.init.normal_(self.past_type, std=0.02)
        nn.init.normal_(self.future_type, std=0.02)
        if self.patch_location is not None:
            # Begin from the robust PatchTST location path.  The future-
            # covariate fusion correction is learned only when it helps.
            nn.init.zeros_(self.guide_head.weight)
        nn.init.zeros_(self.guide_head.bias)


class _NoiseTransformer(nn.Module):
    def __init__(self, config: TransformerDiffusionConfig) -> None:
        super(_NoiseTransformer, self).__init__()
        self.config = config
        self.noisy_projection = nn.Linear(config.target_dim, config.d_model)
        self.guide_projection = nn.Linear(config.target_dim, config.d_model)
        self.timestep_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.SiLU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.position = nn.Parameter(torch.zeros(1, config.horizon, config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.n_layers, norm=nn.LayerNorm(config.d_model)
        )
        self.output = nn.Linear(config.d_model, config.target_dim)
        nn.init.normal_(self.position, std=0.02)
        nn.init.zeros_(self.output.bias)
