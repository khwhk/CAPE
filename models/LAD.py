"""Build LAD with lag alignment and state-conditioned diffusion response."""
from layers.embedding import TransformerDiffusionConfig
from layers.history import HistoryNativeEDM
from layers.state_response import StateResponseEDM


def build_model(args, protocol, config=None):
    config = config or TransformerDiffusionConfig(
        seq_len=args.seq_len, horizon=args.horizon,
        past_dim=len(protocol['columns']), future_cond_dim=len(protocol['columns']) - 1,
        target_dim=1, d_model=args.d_model, n_heads=4, n_layers=args.layers,
        d_ff=2 * args.d_model, dropout=args.dropout, diffusion_steps=args.diffusion_steps,
        guidance_weight=0., center_samples=False,
    )
    names = (
        'window_norm', 'exogenous_mode', 'sigma_max', 'sigma_min', 'log_sigma_mean',
        'log_sigma_std', 'norm_floor', 'denoise_scale_power', 'edm_backbone',
        'patch_length', 'target_stats', 'lag_detrend', 'response_input',
        'response_gain_scale', 'coordinate_center', 'response_shift_scale',
        'response_recent_fraction', 'response_noise_damping', 'lag_temperature',
        'lag_recent_fraction', 'lag_uniform_mix',
    )
    options = {key: getattr(args, key) for key in names}
    options['lag_candidates'] = protocol.get('lag_candidates')
    model_type = StateResponseEDM
    if args.future_mode == 'history':
        model_type = HistoryNativeEDM
        for key in ('history_condition', 'history_period', 'history_decay', 'native_lag', 'reliable_response'):
            options[key] = getattr(args, key)
    return model_type(config, 'zero', 'concat', 'diffusion', 'edm', 'trajectory',
                      0., args.denoise_mae, **options)
