"""IID sample objectives used during refinement."""

import torch


def fair_sample_losses(samples, target):
    """IID ensemble CRPS and unbiased squared error of the distribution mean.

    The variance correction prevents small training ensembles from adding a
    spurious spread penalty. Benchmark evaluation remains empirical CRPS.
    """
    count = samples.shape[1]
    if count < 2:
        raise ValueError("Fair sample objectives require at least two IID draws")
    mean_mse = (
        (samples.mean(1) - target).square() - samples.var(1, unbiased=True) / count
    ).mean()
    ordered = samples.sort(1).values
    weights = 2 * torch.arange(1, count + 1, device=samples.device) - count - 1
    weights = weights.reshape(1, count, *([1] * (samples.ndim - 2)))
    spread = (ordered * weights).sum(1) / (count * (count - 1))
    crps = ((samples - target[:, None]).abs().mean(1) - spread).mean()
    return mean_mse, crps
