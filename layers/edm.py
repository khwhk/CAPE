import torch
from torch import nn
from torch.nn import functional as F
from .embedding import make_beta_schedule
from .initialization import _ConditionTransformer, _NoiseTransformer
from .backbone import (
    TrajectoryConditioner,
    TrajectoryNoiseTransformer,
    DualScaleDenoiser,
    patchify,
    history_statistics,
)
from .lag_alignment import LagTransport

EXOGENOUS_MODES = ("none", "transport", "transport_norm")


class TrajectoryModel(nn.Module):
    def __init__(
        self,
        config,
        location="zero",
        fusion="concat",
        distribution="diffusion",
        prediction_type="edm",
        conditioner="trajectory",
        moment_weight=0.0,
        denoise_mae=0.0,
    ):
        super().__init__()
        config.validate()
        if (
            location,
            fusion,
            distribution,
            prediction_type,
            conditioner,
            config.target_dim,
            config.location_model,
        ) != ("zero", "concat", "diffusion", "edm", "trajectory", 1, "transformer"):
            raise ValueError("CAPE uses direct single-target trajectory EDM")
        self.config = config
        # Maintain both checkpoint keys and seeded initialization of the research code.
        self.conditioner = _ConditionTransformer(config)
        self.denoiser = _NoiseTransformer(config)
        betas = make_beta_schedule(
            config.beta_schedule,
            config.diffusion_steps,
            config.beta_start,
            config.beta_end,
        )
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, 0)
        previous = torch.cat((torch.ones(1, dtype=alpha_bars.dtype), alpha_bars[:-1]))
        variance = betas * (1 - previous) / (1 - alpha_bars)
        for name, value in (
            ("betas", betas),
            ("alphas", alphas),
            ("alpha_bars", alpha_bars),
            ("alpha_bars_previous", previous),
            ("posterior_variance", variance.clamp_min(1e-20)),
        ):
            self.register_buffer(name, value)
        self.location, self.distribution = location, distribution
        self.prediction_type, self.moment_weight = prediction_type, moment_weight
        self.denoise_mae = denoise_mae
        self.conditioner = TrajectoryConditioner(config)
        self.denoiser = TrajectoryNoiseTransformer(config)

    def condition(self, past, future_condition, base_target):
        return self.conditioner(past, future_condition)

    def distribution_loss(self, past, future, target, base):
        return self.training_loss(past, future, target, base)[0]


class ForecastEDM(TrajectoryModel):
    def __init__(
        self,
        *args,
        window_norm=False,
        exogenous_mode="none",
        sigma_max=80.0,
        sigma_min=0.002,
        log_sigma_mean=-1.2,
        log_sigma_std=1.2,
        norm_floor=0.1,
        denoise_scale_power=0.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if self.location != "zero" or self.distribution != "diffusion":
            raise ValueError("This EDM experiment uses direct conditional diffusion")
        self.window_norm = window_norm
        # Targets have unit training-prefix variance before optional window norm.
        self.sigma_data = 1.0
        self.sigma_max, self.sigma_min = sigma_max, sigma_min
        self.log_sigma_mean, self.log_sigma_std = log_sigma_mean, log_sigma_std
        self.norm_floor = norm_floor
        self.denoise_scale_power = denoise_scale_power
        self.exogenous_mode = exogenous_mode
        if exogenous_mode != "none":
            if exogenous_mode not in EXOGENOUS_MODES:
                raise ValueError("Unknown exogenous response mode")
            # Keep the baseline initialization and subsequent RNG stream intact.
            with torch.random.fork_rng(devices=[]):
                if exogenous_mode in ("transport", "transport_norm"):
                    if not isinstance(self.conditioner, TrajectoryConditioner):
                        raise ValueError(
                            "Lag transport requires trajectory conditioning"
                        )
                    self.exogenous = LagTransport(
                        self.config, preserve_norm=exogenous_mode == "transport_norm"
                    )

    def condition(self, past, future_condition, base_target):
        context, guide = super().condition(past, future_condition, base_target)
        if self.exogenous_mode in ("transport", "transport_norm"):
            context = self.exogenous(context, past, future_condition)
        return context, guide

    def coordinates(self, past):
        target = past[..., : self.config.target_dim]
        mean = target.mean(1, keepdim=True)
        scale = (
            target.var(1, keepdim=True, unbiased=False)
            .add(1e-5)
            .sqrt()
            .clamp_min(self.norm_floor)
        )
        if not self.window_norm:
            mean, scale = torch.zeros_like(mean), torch.ones_like(scale)
        return mean, scale

    def denoise(self, noisy, sigma, context):
        denominator = (sigma.square() + self.sigma_data**2).sqrt()
        c_skip = self.sigma_data**2 / denominator.square()
        c_out = sigma * self.sigma_data / denominator
        noise_level = sigma.flatten().log() / 4
        prediction = self.denoiser(
            noisy / denominator,
            torch.zeros_like(noisy),
            context,
            noise_level,
        )
        return c_skip * noisy + c_out * prediction

    def training_loss(self, past, future, target, base, timesteps=None):
        context, _ = self.condition(past, future, base)
        mean, scale = self.coordinates(past)
        clean = (target - mean) / scale
        sigma = (
            torch.randn(len(past), 1, 1, device=past.device) * self.log_sigma_std
            + self.log_sigma_mean
        ).exp()
        noisy = clean + sigma * torch.randn_like(clean)
        prediction = self.denoise(noisy, sigma, context)
        weight = (sigma.square() + self.sigma_data**2) / (
            sigma * self.sigma_data
        ).square()
        squared = weight * (prediction - clean).square()
        if self.denoise_scale_power:
            squared = squared * scale.pow(self.denoise_scale_power)
        loss = squared.mean()
        loss = loss + self.denoise_mae * F.l1_loss(prediction * scale + mean, target)
        return loss, {"denoising": loss.detach()}

    @torch.no_grad()
    def sample(
        self,
        past,
        future_condition,
        base_target,
        n_samples=100,
        seed=None,
        sample_batch_size=128,
        noise_batch_size=None,
        **kwargs
    ):
        self.eval()
        contexts, _ = self.condition(past, future_condition, base_target)
        means, scales = self.coordinates(past)
        generator = (
            torch.Generator(device=past.device).manual_seed(seed)
            if seed is not None
            else None
        )
        ramp = torch.linspace(0, 1, self.config.diffusion_steps, device=past.device)
        levels = (
            self.sigma_max ** (1 / 7)
            + ramp * (self.sigma_min ** (1 / 7) - self.sigma_max ** (1 / 7))
        ) ** 7
        levels = torch.cat((levels, levels.new_zeros(1)))
        total = len(past) * n_samples
        # Zero means one parallel solve. Independent noise chunking preserves
        # the legacy random draws when changing the denoising batch size.
        sample_batch_size = sample_batch_size or total
        initial_noise = None
        if noise_batch_size is not None:
            initial_noise = torch.cat(
                [
                    torch.randn(
                        min(noise_batch_size, total - start),
                        self.config.horizon,
                        self.config.target_dim,
                        device=past.device,
                        dtype=past.dtype,
                        generator=generator,
                    )
                    for start in range(0, total, noise_batch_size)
                ]
            )
        outputs = []
        for start in range(0, len(past) * n_samples, sample_batch_size):
            stop = min(start + sample_batch_size, len(past) * n_samples)
            indices = torch.arange(start, stop, device=past.device) // n_samples
            shape = (len(indices), self.config.horizon, self.config.target_dim)
            value = (
                initial_noise[start:stop]
                if initial_noise is not None
                else torch.randn(
                    shape, device=past.device, dtype=past.dtype, generator=generator
                )
            ) * levels[0]
            context = (
                tuple(value[indices] for value in contexts)
                if isinstance(contexts, tuple)
                else contexts[indices]
            )
            for step in range(self.config.diffusion_steps):
                sigma, following = levels[step], levels[step + 1]
                batch_sigma = sigma.expand(len(indices), 1, 1)
                prediction = self.denoise(value, batch_sigma, context)
                derivative = (value - prediction) / sigma
                proposal = value + (following - sigma) * derivative
                if step < self.config.diffusion_steps - 1:
                    corrected = self.denoise(
                        proposal, following.expand_as(batch_sigma), context
                    )
                    derivative_next = (proposal - corrected) / following
                    proposal = (
                        value + (following - sigma) * (derivative + derivative_next) / 2
                    )
                value = proposal
            outputs.append(value * scales[indices] + means[indices])
        return torch.cat(outputs).reshape(len(past), n_samples, *base_target.shape[1:])

    def differentiable_sample(self, past, future, base, n_samples=4, seed=None):
        """Same 20-step Heun path as inference, without disabling gradients.

        Call in eval mode for inference-consistent dropout behavior. Independent
        random draws (not antithetic pairs) are required by the fair objectives.
        """
        contexts, _ = self.condition(past, future, base)
        means, scales = self.coordinates(past)
        indices = torch.arange(len(past), device=past.device).repeat_interleave(
            n_samples
        )
        context = (
            tuple(c[indices] for c in contexts)
            if isinstance(contexts, tuple)
            else contexts[indices]
        )
        ramp = torch.linspace(0, 1, self.config.diffusion_steps, device=past.device)
        levels = (
            self.sigma_max ** (1 / 7)
            + ramp * (self.sigma_min ** (1 / 7) - self.sigma_max ** (1 / 7))
        ) ** 7
        levels = torch.cat((levels, levels.new_zeros(1)))
        generator = (
            torch.Generator(device=past.device).manual_seed(seed)
            if seed is not None
            else None
        )
        shape = (len(indices), self.config.horizon, self.config.target_dim)
        value = (
            torch.randn(
                shape, device=past.device, dtype=past.dtype, generator=generator
            )
            * levels[0]
        )
        for step in range(self.config.diffusion_steps):
            sigma, following = levels[step], levels[step + 1]
            batch_sigma = sigma.expand(len(indices), 1, 1)
            derivative = (value - self.denoise(value, batch_sigma, context)) / sigma
            proposal = value + (following - sigma) * derivative
            if step < self.config.diffusion_steps - 1:
                corrected = self.denoise(
                    proposal, following.expand_as(batch_sigma), context
                )
                derivative_next = (proposal - corrected) / following
                proposal = (
                    value + (following - sigma) * (derivative + derivative_next) / 2
                )
            value = proposal
        value = value * scales[indices] + means[indices]
        return value.reshape(
            len(past), n_samples, self.config.horizon, self.config.target_dim
        )


class ExtendedForecastEDM(ForecastEDM):
    def __init__(
        self,
        *args,
        edm_backbone="global",
        patch_length=12,
        target_stats=False,
        response_memory=False,
        lag_detrend=False,
        lag_candidates=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if self.exogenous_mode not in ("none", "transport", "transport_norm"):
            raise ValueError(
                "Extensions require direct trajectory transport or no transport"
            )
        config = self.config
        self.patch_length = min(patch_length, config.horizon)
        self.edm_backbone, self.target_stats = edm_backbone, target_stats
        self.response_memory = response_memory
        if self.exogenous_mode != "none" and (
            lag_detrend or lag_candidates is not None
        ):
            self.exogenous = LagTransport(
                config,
                preserve_norm=self.exogenous_mode == "transport_norm",
                detrend=lag_detrend,
                lag_candidates=lag_candidates,
            )
        if target_stats:
            self.stats_encoder = nn.Linear(4, config.d_model)
        if response_memory:
            raise ValueError(
                "Response-memory experiments are outside the selected CAPE recipes"
            )
        if edm_backbone == "dual":
            self.local_history = nn.Linear(
                self.patch_length * config.past_dim, config.d_model
            )
            self.local_future = nn.Linear(
                self.patch_length * config.future_cond_dim, config.d_model
            )
            self.denoiser = DualScaleDenoiser(config, self.denoiser, self.patch_length)

    def condition(self, past, future_condition, base_target):
        context, guide = super().condition(past, future_condition, base_target)
        if self.target_stats:
            context = torch.cat(
                (context, self.stats_encoder(history_statistics(past))[:, None]), 1
            )
        aligned = (
            self.exogenous.transport_path(past, future_condition)
            if self.exogenous_mode != "none"
            else future_condition
        )
        memory = self.memory(past, aligned) if self.response_memory else None
        if self.edm_backbone == "global":
            if memory is not None:
                begin = self.config.past_dim
                end = begin + self.config.future_cond_dim
                context = torch.cat(
                    (
                        context[:, :begin],
                        context[:, begin:end] + memory.mean(1, keepdim=True),
                        context[:, end:],
                    ),
                    1,
                )
            return context, guide
        mean, scale = self.coordinates(past)
        local_past = torch.cat(((past[..., :1] - mean) / scale, past[..., 1:]), -1)
        history = self.local_history(patchify(local_past, self.patch_length))
        future = self.local_future(patchify(aligned, self.patch_length))
        if memory is not None:
            future = future + memory
        return (context, history, future), guide
