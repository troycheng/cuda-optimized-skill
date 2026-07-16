#!/usr/bin/env python3
"""Randomized AB/BA paired timing with telemetry-gated blocks."""

from __future__ import annotations

import copy
import math
import random
import sys
from collections.abc import Mapping
from numbers import Real
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from benchmark import measure_once, prepare_solution, warm_solution  # noqa: E402
from telemetry import read_gpu_telemetry, validate_block  # noqa: E402


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_nonnegative(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative real number")
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite non-negative real number"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name} must be a finite non-negative real number")
    return parsed


def _positive_timing(value, name: str) -> float:
    timing = _finite_nonnegative(value, name)
    if timing <= 0:
        raise ValueError(f"{name} must be positive")
    return timing


def _nonempty_string(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def run_paired(
    baseline_file,
    candidate_file,
    *,
    backend,
    dims,
    ptr_size,
    arch,
    nvcc_bin,
    seed,
    blocks,
    warmup,
    max_temperature_delta_c=5,
    max_clock_delta_pct=5,
    prepare_fn=None,
    warm_fn=None,
    measure_fn=None,
    telemetry_reader=None,
    block_validator=None,
) -> dict:
    """Prepare two solutions once and collect randomized paired observations."""
    _nonempty_string(baseline_file, "baseline_file")
    _nonempty_string(candidate_file, "candidate_file")
    _nonempty_string(backend, "backend")
    if backend not in {"auto", "cuda", "cutlass", "triton"}:
        raise ValueError("backend must be auto, cuda, cutlass, or triton")
    if not isinstance(dims, Mapping):
        raise ValueError("dims must be a mapping")
    _nonnegative_int(ptr_size, "ptr_size")
    _nonempty_string(arch, "arch")
    _nonempty_string(nvcc_bin, "nvcc_bin")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    block_count = _positive_int(blocks, "blocks")
    warmup_count = _nonnegative_int(warmup, "warmup")
    temperature_limit = _finite_nonnegative(
        max_temperature_delta_c, "max_temperature_delta_c"
    )
    clock_limit = _finite_nonnegative(max_clock_delta_pct, "max_clock_delta_pct")

    prepare = prepare_fn or prepare_solution
    warm = warm_fn or warm_solution
    measure = measure_fn or measure_once
    read_telemetry = telemetry_reader or read_gpu_telemetry
    validate = block_validator or validate_block

    common = {
        "backend": backend,
        "ptr_size": ptr_size,
        "arch": arch,
        "nvcc_bin": nvcc_bin,
        "seed": seed,
    }
    baseline_state = prepare(
        baseline_file,
        dims=copy.deepcopy(dict(dims)),
        **common,
    )
    candidate_state = prepare(
        candidate_file,
        dims=copy.deepcopy(dict(dims)),
        **common,
    )
    if baseline_state is candidate_state:
        raise ValueError(
            "prepare_fn must return independent baseline and candidate states"
        )

    warm(baseline_state, warmup_count)
    warm(candidate_state, warmup_count)

    rng = random.Random(seed)
    pairs = []
    for _block_index in range(block_count):
        before = read_telemetry()
        order = rng.choice(("AB", "BA"))
        if order == "AB":
            baseline_timing = _positive_timing(
                measure(baseline_state), "baseline timing"
            )
            candidate_timing = _positive_timing(
                measure(candidate_state), "candidate timing"
            )
        else:
            candidate_timing = _positive_timing(
                measure(candidate_state), "candidate timing"
            )
            baseline_timing = _positive_timing(
                measure(baseline_state), "baseline timing"
            )
        after = read_telemetry()
        validation = validate(
            before,
            after,
            max_temperature_delta_c=temperature_limit,
            max_clock_delta_pct=clock_limit,
        )
        if not isinstance(validation, Mapping):
            raise ValueError("block_validator must return a mapping")
        valid = validation.get("valid")
        if type(valid) is not bool:
            raise ValueError("block_validator result valid must be a bool")
        invalid_reasons = validation.get("invalid_reasons", [])
        if not isinstance(invalid_reasons, list) or not all(
            isinstance(reason, str) for reason in invalid_reasons
        ):
            raise ValueError(
                "block_validator invalid_reasons must be a list of strings"
            )

        pairs.append(
            {
                "order": order,
                "baseline": baseline_timing,
                "candidate": candidate_timing,
                "valid": valid,
                "invalid_reasons": list(invalid_reasons),
                "telemetry": {
                    "before": copy.deepcopy(before),
                    "after": copy.deepcopy(after),
                    "status": validation.get("telemetry_status", "unknown"),
                    "unknown_reasons": list(
                        validation.get("unknown_reasons", [])
                    ),
                    "temperature_delta_c": validation.get(
                        "temperature_delta_c"
                    ),
                    "clock_delta_pct": validation.get("clock_delta_pct"),
                    "clock_delta_capped": validation.get(
                        "clock_delta_capped", False
                    ),
                },
            }
        )

    return {
        "baseline_file": baseline_file,
        "candidate_file": candidate_file,
        "backend": backend,
        "seed": seed,
        "blocks": block_count,
        "warmup": warmup_count,
        "pairs": pairs,
    }
