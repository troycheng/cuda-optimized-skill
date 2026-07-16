#!/usr/bin/env python3
"""Shared benchmark and profiler option handling for the skill scripts."""

from __future__ import annotations

import os


BENCHMARK_DEFAULTS = {
    "backend": "auto",
    "arch": "",
    "gpu": 0,
    "atol": 1e-4,
    "rtol": 1e-3,
    "seed": 42,
    "validation_seeds": "",
    "nvcc_bin": "nvcc",
}

NCU_DEFAULTS = {
    "ncu_bin": "ncu",
    "launch_count": 1,
}


def normalized_benchmark_options(state: dict) -> dict:
    out = dict(BENCHMARK_DEFAULTS)
    out.update(state.get("benchmark_options") or {})
    return out


def normalized_ncu_options(state: dict) -> dict:
    out = dict(NCU_DEFAULTS)
    out.update(state.get("ncu_options") or {})
    return out


def dims_argv(dims: dict) -> list[str]:
    return [f"--{key}={value}" for key, value in sorted((dims or {}).items())]


def benchmark_option_argv(
    state: dict,
    *,
    include_validation: bool = True,
    include_reference: bool = True,
) -> list[str]:
    """Return stable CLI options shared by every benchmark invocation."""
    opts = normalized_benchmark_options(state)
    argv = [
        "--backend", str(opts["backend"]),
        "--gpu", str(opts["gpu"]),
        "--atol", str(opts["atol"]),
        "--rtol", str(opts["rtol"]),
        "--seed", str(opts["seed"]),
        "--nvcc-bin", str(opts["nvcc_bin"]),
    ]
    if opts.get("arch"):
        argv += ["--arch", str(opts["arch"])]
    if include_validation and opts.get("validation_seeds"):
        argv += ["--validation-seeds", str(opts["validation_seeds"])]
    if include_reference and state.get("ref_file"):
        argv += ["--ref", os.path.abspath(state["ref_file"])]
    ptr_size = int(state.get("ptr_size") or 0)
    if ptr_size > 0:
        argv += ["--ptr-size", str(ptr_size)]
    argv += dims_argv(state.get("dims") or {})
    return argv
