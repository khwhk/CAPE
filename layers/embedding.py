import math
from dataclasses import dataclass
import torch
from torch.nn import functional as F

def make_beta_schedule(
    schedule: str = "cosine",
    num_timesteps: int = 20,
    start: float = 1e-4,
    end: float = 1e-2,
) -> torch.Tensor:
    """Create a beta schedule using the schedule names supported by NsDiff."""

    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")
    normalized = schedule.lower()
    if normalized == "linear":
        betas = torch.linspace(start, end, num_timesteps, dtype=torch.float64)
    elif normalized == "const":
        betas = torch.full((num_timesteps,), end, dtype=torch.float64)
    elif normalized == "quad":
        betas = torch.linspace(
            math.sqrt(start), math.sqrt(end), num_timesteps, dtype=torch.float64
        ).square()
    elif normalized == "sigmoid":
        values = torch.linspace(-6.0, 6.0, num_timesteps, dtype=torch.float64)
        betas = torch.sigmoid(values) * (end - start) + start
    elif normalized in ("cosine", "cosine_reverse"):
        cosine_s = 0.008
        values = []
        for index in range(num_timesteps):
            numerator = (
                math.cos(
                    ((index + 1) / num_timesteps + cosine_s)
                    / (1.0 + cosine_s)
                    * math.pi
                    / 2.0
                )
                ** 2
            )
            denominator = (
                math.cos(
                    (index / num_timesteps + cosine_s)
                    / (1.0 + cosine_s)
                    * math.pi
                    / 2.0
                )
                ** 2
            )
            values.append(min(1.0 - numerator / denominator, 0.999))
        betas = torch.tensor(values, dtype=torch.float64)
        if normalized == "cosine_reverse":
            betas = betas.flip(0)
    else:
        raise ValueError("Unsupported beta schedule: {}".format(schedule))
    if (
        not torch.isfinite(betas).all()
        or torch.any(betas <= 0.0)
        or torch.any(betas >= 1.0)
    ):
        raise ValueError("beta schedule must stay strictly between zero and one")
    return betas.float()


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor, dimension: int
) -> torch.Tensor:
    """Standard sinusoidal embedding for integer diffusion steps."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    half = dimension // 2
    if half == 0:
        return timesteps.float().unsqueeze(-1)
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    if embedding.shape[-1] < dimension:
        embedding = F.pad(embedding, (0, dimension - embedding.shape[-1]))
    return embedding


@dataclass(frozen=True)
class TransformerDiffusionConfig:
    seq_len: int
    horizon: int
    past_dim: int
    future_cond_dim: int
    target_dim: int
    d_model: int = 48
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 96
    dropout: float = 0.1
    diffusion_steps: int = 20
    beta_schedule: str = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 1e-2
    guidance_weight: float = 0.2
    sampler: str = "ddim"
    clip_denoised: float = 5.0
    center_samples: bool = True
    location_model: str = "transformer"
    location_patch_length: int = 16
    location_patch_stride: int = 8

    def validate(self) -> None:
        for name in (
            "seq_len",
            "horizon",
            "past_dim",
            "future_cond_dim",
            "target_dim",
            "d_model",
            "n_heads",
            "n_layers",
            "d_ff",
            "diffusion_steps",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.guidance_weight < 0.0:
            raise ValueError("guidance_weight must be non-negative")
        if self.sampler not in ("ddim", "ddpm"):
            raise ValueError("sampler must be 'ddim' or 'ddpm'")
        if self.clip_denoised < 0.0:
            raise ValueError("clip_denoised must be non-negative")
        if not isinstance(self.center_samples, bool):
            raise ValueError("center_samples must be boolean")
        if self.location_model not in ("transformer", "patchtst"):
            raise ValueError("location_model must be 'transformer' or 'patchtst'")
        for name in ("location_patch_length", "location_patch_stride"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if (
            self.location_model == "patchtst"
            and self.location_patch_length > self.seq_len
        ):
            raise ValueError("location_patch_length cannot exceed seq_len")
        make_beta_schedule(
            self.beta_schedule,
            self.diffusion_steps,
            self.beta_start,
            self.beta_end,
        )
