"""CAPE: Lag-Aligned Conditioning and Condition-Responsive Denoising."""

import importlib

from layers.embedding import TransformerDiffusionConfig
from layers.history import HistoricalStateResponseEDM
from layers.state_response import StateResponseEDM


def build_model(args, protocol, config=None):
    worker = getattr(args, "recipe_worker", "legacy")
    if worker != "legacy":
        factory = {"search": "local", "horizon": "rank"}.get(worker, worker)
        recipe = importlib.import_module("exp.recipes." + factory)
        if worker in ("native", "adaptive"):
            defaults = recipe.parser().parse_args(["--output-dir", args.output_dir])
            vars(defaults).update(vars(args))
        else:
            defaults = recipe.arguments(
                dict(arguments=vars(args)), args.output_dir, args.device
            )
        return recipe.make_model(defaults, protocol, config)
    config = config or TransformerDiffusionConfig(
        seq_len=args.seq_len,
        horizon=args.horizon,
        past_dim=len(protocol["columns"]),
        future_cond_dim=len(protocol["columns"]) - 1,
        target_dim=1,
        d_model=args.d_model,
        n_heads=4,
        n_layers=args.layers,
        d_ff=2 * args.d_model,
        dropout=args.dropout,
        diffusion_steps=args.diffusion_steps,
        guidance_weight=0.0,
        center_samples=False,
    )
    names = (
        "window_norm",
        "exogenous_mode",
        "sigma_max",
        "sigma_min",
        "log_sigma_mean",
        "log_sigma_std",
        "norm_floor",
        "denoise_scale_power",
        "edm_backbone",
        "patch_length",
        "target_stats",
        "lag_detrend",
        "response_input",
        "response_gain_scale",
        "coordinate_center",
        "response_shift_scale",
        "response_recent_fraction",
        "response_noise_damping",
        "lag_temperature",
        "lag_recent_fraction",
        "lag_uniform_mix",
    )
    options = {key: getattr(args, key) for key in names}
    options["lag_candidates"] = protocol.get("lag_candidates")
    model_type = StateResponseEDM
    if args.future_mode == "history" and args.history_condition != "provided":
        model_type = HistoricalStateResponseEDM
        for key in ("history_condition", "history_period", "history_decay"):
            options[key] = getattr(args, key)
    return model_type(
        config,
        "zero",
        "concat",
        "diffusion",
        "edm",
        "trajectory",
        0.0,
        args.denoise_mae,
        **options
    )
