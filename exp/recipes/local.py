import argparse

import copy

import json

from pathlib import Path

import sys

import torch

from exp.recipes import gap as gap

from exp.recipes.core import write_json

from layers.history import HistoryLocalEDM

original_arguments, original_model = gap.arguments, gap.make_model

original_refine = gap.base.fixed_refine


def arguments(job, output, device="cuda:0"):
    args = original_arguments(job, output, device)
    for k, v in dict(
        anchor_mix=0.0,
        history_level_window=0,
        refine_scope="all",
        refine_weight_decay=0.0,
    ).items():
        vars(args).setdefault(k, v)
    return args


def make_model(args, protocol, config=None):
    original = gap.HistoryGapEDM
    gap.HistoryGapEDM = lambda *a, **kw: HistoryLocalEDM(
        *a,
        anchor_mix=args.anchor_mix,
        history_level_window=args.history_level_window,
        **kw
    )
    try:
        return original_model(args, protocol, config)
    finally:
        gap.HistoryGapEDM = original


def fixed_refine(model, data, args):
    scope = args.refine_scope
    if scope not in ("all", "response_shift", "output_head"):
        raise ValueError("Unknown refinement scope: " + scope)
    names = dict(model.named_parameters())
    selected = (
        set(names)
        if scope == "all"
        else {
            n
            for n in names
            if n.startswith(
                ("state_response.response.2.",)
                if scope == "response_shift"
                else ("denoiser.output.", "denoiser.global_denoiser.output.")
            )
        }
    )
    if not selected:
        raise ValueError("Empty refinement parameter set")
    original_requires_grad = {n: p.requires_grad for n, p in names.items()}
    before = (
        {n: p.detach().cpu().clone() for n, p in names.items()}
        if scope != "all"
        else {}
    )
    handles = []
    for n, p in names.items():
        p.requires_grad_(n in selected)
        if scope == "response_shift" and n in selected:
            # Freeze the gain row as well as every upstream/shared feature.
            handles.append(
                p.register_hook(
                    lambda g: torch.cat((g[:1], torch.zeros_like(g[1:])), 0)
                )
            )
    options = copy.copy(args)
    options.weight_decay = args.refine_weight_decay
    if scope == "response_shift" and options.weight_decay:
        raise ValueError(
            "Shift-only tuning requires zero weight decay to preserve the gain row"
        )
    try:
        history, audit = original_refine(model, data, options)
    finally:
        for handle in handles:
            handle.remove()
        for n, p in names.items():
            p.requires_grad_(original_requires_grad[n])
    changed_frozen = [
        n
        for n, p in names.items()
        if n not in selected and not torch.equal(before[n], p.detach().cpu())
    ]
    gain_unchanged = scope != "response_shift" or all(
        torch.equal(before[n][1:], names[n].detach().cpu()[1:]) for n in selected
    )
    if changed_frozen or not gain_unchanged:
        raise RuntimeError("Refinement modified frozen parameters")
    audit["parameter_scope"] = dict(
        scope=scope,
        names=sorted(selected),
        effective_trainable_elements=sum(names[n].numel() for n in selected)
        // (2 if scope == "response_shift" else 1),
        frozen_parameters_unchanged=not changed_frozen,
        gain_row_unchanged=gain_unchanged,
        weight_decay=options.weight_decay,
    )
    return history, audit


def run(job, output, device="cuda:0", smoke=False):
    previous = gap.arguments, gap.make_model, gap.base.fixed_refine
    gap.arguments, gap.make_model, gap.base.fixed_refine = (
        arguments,
        make_model,
        fixed_refine,
    )
    try:
        result = gap.run(job, output, device, smoke)
        result.update(
            worker="history_local",
            source_legacy_validation_selected=job.get(
                "source_legacy_validation_selected", False
            ),
        )
        write_json(Path(output) / "summary.json", result)
        return result
    finally:
        gap.arguments, gap.make_model, gap.base.fixed_refine = previous
