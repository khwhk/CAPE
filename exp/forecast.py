"""Standalone training and checkpoint evaluation; no search orchestration."""
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from data_provider.data_loader import load_windows, subset
from layers.embedding import TransformerDiffusionConfig
from models.LAD import build_model
from utils.evaluation import score, write_json
from .training import train


def run(args):
    saved = None
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if saved['args']['seed'] != args.seed or saved['args']['future_mode'] != args.future_mode:
            raise ValueError('Checkpoint seed and information setting must match the requested run')
        runtime = {k: getattr(args, k) for k in (
            'data_path', 'device', 'threads', 'evaluate_test', 'output_dir', 'checkpoint', 'smoke')}
        vars(args).update(saved['args'])
        vars(args).update(runtime)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    datasets, protocol = load_windows(args)
    if saved:
        for name in ('columns', 'length', 'train_end', 'validation_end', 'scaler_mean', 'scaler_scale'):
            if protocol[name] != saved['protocol'][name]:
                raise ValueError('Checkpoint/data mismatch: ' + name)
    if args.smoke:
        datasets = {name: subset(data, 16) for name, data in datasets.items()}
        args.epochs, args.eval_every, args.validation_origins = 1, 1, 16
        args.validation_samples, args.samples = 4, 4
        if not saved:
            args.diffusion_steps = 4
    config = TransformerDiffusionConfig(**saved['model_config']) if saved else None
    model = build_model(args, protocol, config).to(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if saved:
        model.load_state_dict(saved['state_dict'], strict=True)
        history, selection = {'restored_from': args.checkpoint}, saved['selection']
    else:
        history, selection = train(model, datasets, args)
    metadata = dict(args=vars(args), protocol=protocol, model_config=asdict(model.config),
                    parameters=sum(p.numel() for p in model.parameters()),
                    posthoc_sample_centering=False, metric_space='train_prefix_standardized',
                    selection=selection)
    torch.save(dict(state_dict=model.state_dict(), **metadata), output / 'model.pt')
    write_json(output / 'config.json', metadata)
    write_json(output / 'history.json', history)
    metrics = {}
    for split in ('validation', 'test'):
        if split in datasets:
            metrics[split] = score(model, datasets[split], args, args.seed + 2000, args.samples)
            print(split, metrics[split], flush=True)
    result = dict(metadata, metrics=metrics)
    write_json(output / 'summary.json', result)
    return result
