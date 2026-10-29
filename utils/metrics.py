"""Sample-based metrics aligned with the NsDiff evaluation protocol.

The public aggregate intentionally contains only MSE, MAE, CRPS and QICE.
MSE/MAE use the sample mean as the point forecast, matching NsDiff's
``ProbMSE`` and ``ProbMAE`` implementations.  QICE uses ten equal-probability
bins by default and reports the mean absolute deviation from ideal occupancy.
"""

from typing import Any, Dict, Tuple

import numpy as np
import torch


def _to_tensor(value: Any, name: str) -> Tuple[torch.Tensor, bool]:
    is_numpy = isinstance(value, np.ndarray)
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        try:
            tensor = torch.as_tensor(np.asarray(value))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "{} must be convertible to a numeric array".format(name)
            ) from exc
    if not tensor.is_floating_point():
        tensor = tensor.float()
    if not torch.isfinite(tensor).all():
        raise ValueError("{} contains NaN or infinite values".format(name))
    return tensor, is_numpy


def _canonical_samples(
    samples: Any, target: Any
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    sample_tensor, samples_were_numpy = _to_tensor(samples, "samples")
    target_tensor, target_was_numpy = _to_tensor(target, "target")
    target_tensor = target_tensor.to(
        device=sample_tensor.device, dtype=sample_tensor.dtype
    )
    if target_tensor.ndim < 2:
        raise ValueError("target must include batch and event dimensions")
    sample_axis_count = sample_tensor.ndim - target_tensor.ndim
    if sample_axis_count < 1:
        raise ValueError(
            "samples must add at least one sample axis after batch; got {} and {}".format(
                tuple(sample_tensor.shape), tuple(target_tensor.shape)
            )
        )
    if sample_tensor.shape[0] != target_tensor.shape[0]:
        raise ValueError("samples and target must share the batch dimension")
    trailing = tuple(sample_tensor.shape[1 + sample_axis_count :])
    if trailing != tuple(target_tensor.shape[1:]):
        raise ValueError(
            "samples event shape {} does not match target event shape {}".format(
                trailing, tuple(target_tensor.shape[1:])
            )
        )
    canonical = sample_tensor.reshape(
        sample_tensor.shape[0], -1, *target_tensor.shape[1:]
    )
    return canonical, target_tensor, samples_were_numpy or target_was_numpy


def _finish(value: torch.Tensor, reduction: str, return_numpy: bool) -> Any:
    if reduction == "mean":
        return float(value.mean().detach().cpu().item())
    if reduction == "none":
        detached = value.detach()
        if return_numpy:
            return detached.cpu().numpy()
        return detached
    raise ValueError("reduction must be 'mean' or 'none'")


def sample_crps(samples: Any, target: Any, reduction: str = "mean") -> Any:
    """Empirical CRPS without constructing an ``S x S`` tensor.

    ``samples`` may be ``[B,S,...]`` or nested ``[B,M,K,...]``. All sample
    axes are flattened while the trailing event axes must match ``target``.
    """

    values, truth, return_numpy = _canonical_samples(samples, target)
    sample_count = values.shape[1]
    first_term = torch.mean(torch.abs(values - truth.unsqueeze(1)), dim=1)
    sorted_values = torch.sort(values, dim=1).values
    indices = torch.arange(
        1, sample_count + 1, device=values.device, dtype=values.dtype
    )
    coefficients = 2.0 * indices - float(sample_count) - 1.0
    coefficient_shape = (1, sample_count) + (1,) * (values.ndim - 2)
    spread = torch.sum(
        sorted_values * coefficients.reshape(coefficient_shape), dim=1
    ) / float(sample_count * sample_count)
    return _finish(first_term - spread, reduction, return_numpy)


def interval_coverage(
    samples: Any, target: Any, level: float = 0.9, reduction: str = "mean"
) -> Any:
    if level <= 0.0 or level >= 1.0:
        raise ValueError("level must be in (0, 1)")
    values, truth, return_numpy = _canonical_samples(samples, target)
    alpha = (1.0 - level) / 2.0
    lower = torch.quantile(values, alpha, dim=1)
    upper = torch.quantile(values, 1.0 - alpha, dim=1)
    covered = ((truth >= lower) & (truth <= upper)).to(values.dtype)
    return _finish(covered, reduction, return_numpy)


def interval_width(
    samples: Any, target: Any, level: float = 0.9, reduction: str = "mean"
) -> Any:
    if level <= 0.0 or level >= 1.0:
        raise ValueError("level must be in (0, 1)")
    values, _, return_numpy = _canonical_samples(samples, target)
    alpha = (1.0 - level) / 2.0
    lower = torch.quantile(values, alpha, dim=1)
    upper = torch.quantile(values, 1.0 - alpha, dim=1)
    return _finish(upper - lower, reduction, return_numpy)


def _point_error(
    samples: Any, target: Any, statistic: str, squared: bool, reduction: str
) -> Any:
    values, truth, return_numpy = _canonical_samples(samples, target)
    normalized = statistic.lower()
    if normalized == "median":
        point = torch.quantile(values, 0.5, dim=1)
    elif normalized == "mean":
        point = values.mean(dim=1)
    else:
        raise ValueError("statistic must be 'mean' or 'median'")
    error = point - truth
    value = error.square() if squared else error.abs()
    return _finish(value, reduction, return_numpy)


def sample_mae(
    samples: Any,
    target: Any,
    statistic: str = "median",
    reduction: str = "mean",
) -> Any:
    return _point_error(samples, target, statistic, False, reduction)


def sample_mse(
    samples: Any,
    target: Any,
    statistic: str = "mean",
    reduction: str = "mean",
) -> Any:
    return _point_error(samples, target, statistic, True, reduction)


def quantile_interval_coverage_error(
    samples: Any,
    target: Any,
    n_bins: int = 10,
    reduction: str = "mean",
) -> Any:
    """Return quantile interval coverage error (QICE).

    For every scalar event, the predictive samples define ``n_bins`` adjacent
    quantile intervals.  Targets below the sample minimum are folded into the
    first interval and targets above the maximum into the last interval, as in
    NsDiff.  QICE is the mean absolute difference between empirical interval
    occupancy and the ideal value ``1 / n_bins``; lower is better.

    ``reduction='none'`` returns the occupancy error for each quantile bin.
    """

    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 2:
        raise ValueError("n_bins must be an integer greater than one")
    values, truth, return_numpy = _canonical_samples(samples, target)
    flattened_samples = values.movedim(1, -1).reshape(-1, values.shape[1])
    flattened_truth = truth.reshape(-1)
    quantile_levels = torch.linspace(
        0.0,
        1.0,
        n_bins + 1,
        device=values.device,
        dtype=values.dtype,
    )
    boundaries = torch.quantile(flattened_samples, quantile_levels, dim=1)
    membership = torch.sum(flattened_truth.unsqueeze(0) > boundaries, dim=0).clamp(
        min=1, max=n_bins
    )
    counts = torch.bincount(membership - 1, minlength=n_bins).to(values.dtype)
    occupancy = counts / float(flattened_truth.numel())
    errors = torch.abs(occupancy - 1.0 / float(n_bins))
    return _finish(errors, reduction, return_numpy)


def evaluate_samples(
    samples: Any,
    target: Any,
    qice_bins: int = 10,
) -> Dict[str, float]:
    """Return exactly the four metrics used by the probability experiments."""

    return {
        "mse": float(sample_mse(samples, target, statistic="mean")),
        "mae": float(sample_mae(samples, target, statistic="mean")),
        "crps": float(sample_crps(samples, target)),
        "qice": float(
            quantile_interval_coverage_error(samples, target, n_bins=qice_bins)
        ),
    }


__all__ = [
    "evaluate_samples",
    "interval_coverage",
    "interval_width",
    "quantile_interval_coverage_error",
    "sample_crps",
    "sample_mae",
    "sample_mse",
]
