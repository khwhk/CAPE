"""Shared defaults; dataset-specific choices live beside the launch scripts."""
import json
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = dict(
    location='zero', conditioner='trajectory', fusion='concat', distribution='diffusion',
    prediction_type='edm', conditioner_checkpoint=None, moment_weight=0.,
    guide_epochs=0, guide_loss='mixed', point_tolerance=.02,
    validation_fold=2, validation_blocks=1, window_norm=False,
    history_condition='provided', history_period=24, history_decay=0.,
    native_lag=False, reliable_response=False, response_memory=False,
    sample_finetune_epochs=0, lag_grid='fixed', calendar=False,
    norm_floor=.1, denoise_scale_power=0., target_stats=False, lag_detrend=False,
    response_input='aligned', response_gain_scale=1., response_shift_scale=.5,
    response_recent_fraction=.25, response_noise_damping=0.,
    lag_temperature=1., lag_recent_fraction=1., lag_uniform_mix=0.,
    weight_decay=.01, selection_objective='point', coordinate_center='baseline',
    sigma_max=80., sigma_min=.002, log_sigma_mean=-1.2, log_sigma_std=1.2,
    lr_schedule='constant', ema_decay=.99, denoise_mae=1.,
    epochs=50, patience=5, eval_every=1, validation_origins=256, validation_samples=64,
    samples=100, diffusion_steps=20, d_model=48, layers=1, dropout=0.,
    batch_size=64, eval_batch_size=32, sample_batch_size=128, lr=.001,
    edm_backbone='global', patch_length=12, exogenous_mode='transport', threads=2,
)


def load_settings(mode, dataset, length):
    presets = json.loads((ROOT / 'scripts' / mode / 'presets.json').read_text())
    return Namespace(**dict(DEFAULTS, **presets[dataset][length]))
