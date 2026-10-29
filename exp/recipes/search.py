import argparse

import json

from pathlib import Path

import sys

from exp.recipes import local as local

from exp.recipes.core import write_json


def run(job, output, device="cuda:0", smoke=False):
    module = local.gap.base
    original = module.fair_sample_losses
    weight = job["arguments"].get("refine_spread_weight", 0.0)

    def losses(samples, target):
        mse, crps = original(samples, target)
        if weight:
            mse = mse + weight * samples.var(1, unbiased=True).mean()
        return mse, crps

    module.fair_sample_losses = losses
    try:
        result = local.run(job, output, device, smoke)
        result.update(
            worker="history_search",
            training_spread_penalty=weight,
            probability_note="Spread penalty is an exploratory training regularizer, not a new innovation or posthoc sample transformation.",
        )
        write_json(Path(output) / "summary.json", result)
        return result
    finally:
        module.fair_sample_losses = original
