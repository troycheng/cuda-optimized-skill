#!/usr/bin/env python3
"""Deterministic, position-balanced schedules for paired experiments."""

from __future__ import annotations

import copy
import math
import random
import re
from collections.abc import Mapping
from numbers import Real


_DESIGN_KEYS = {
    "schema_version",
    "formal",
    "schedule",
    "experimental_unit",
    "aggregation",
    "resampling_unit",
    "ci",
    "min_valid_pairs",
    "wins_required",
    "guardrails",
    "exclusion_policy",
    "retry_policy",
}
_PAIR_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def balanced_pair_orders(blocks: int, *, seed: int = 0) -> list[str]:
    """Return a seeded AB/BA schedule whose direction counts differ by at most one.

    Independent random choices do not guarantee position balance and can confound a
    candidate with startup or thermal drift.  This helper freezes the complete
    schedule before measurement while retaining a randomized ordinal order.
    """
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks <= 0:
        raise ValueError("blocks must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    rng = random.Random(seed)
    half = blocks // 2
    orders = ["AB"] * half + ["BA"] * half
    if blocks % 2:
        orders.append(rng.choice(("AB", "BA")))
    rng.shuffle(orders)
    return orders


def _closed_mapping(value, *, field: str, keys: set[str]) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    actual = set(value)
    missing = sorted(keys - actual)
    unknown = sorted(actual - keys)
    if missing or unknown:
        raise ValueError(f"{field} has missing={missing} unknown={unknown}")
    return dict(value)


def _nonempty_string(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _positive_int(value, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _finite_nonnegative(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite non-negative number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return parsed


def _validate_guardrails(value) -> None:
    guardrails = _closed_mapping(
        value, field="guardrails", keys={"relative", "absolute"}
    )
    relative = guardrails["relative"]
    absolute = guardrails["absolute"]
    if not isinstance(relative, list) or not relative:
        raise ValueError("guardrails.relative must be a non-empty array")
    if not isinstance(absolute, list) or not absolute:
        raise ValueError("guardrails.absolute must be a non-empty array")

    relative_keys = {"metric", "comparison", "direction", "limit_pct"}
    for index, item in enumerate(relative):
        row = _closed_mapping(
            item, field=f"guardrails.relative[{index}]", keys=relative_keys
        )
        _nonempty_string(row["metric"], f"guardrails.relative[{index}].metric")
        if row["comparison"] not in {"min_improvement", "max_regression"}:
            raise ValueError("relative guardrail comparison is invalid")
        if row["direction"] not in {"lower", "higher"}:
            raise ValueError("relative guardrail direction is invalid")
        _finite_nonnegative(
            row["limit_pct"], f"guardrails.relative[{index}].limit_pct"
        )

    absolute_keys = {"metric", "operator", "limit"}
    for index, item in enumerate(absolute):
        row = _closed_mapping(
            item, field=f"guardrails.absolute[{index}]", keys=absolute_keys
        )
        _nonempty_string(row["metric"], f"guardrails.absolute[{index}].metric")
        if row["operator"] not in {"<=", ">="}:
            raise ValueError("absolute guardrail operator is invalid")
        if isinstance(row["limit"], bool) or not isinstance(row["limit"], Real):
            raise ValueError("absolute guardrail limit must be finite")
        if not math.isfinite(float(row["limit"])):
            raise ValueError("absolute guardrail limit must be finite")


def validate_frozen_design(value) -> dict:
    """Validate and detach a complete V2.5 formal experiment design."""
    design = _closed_mapping(value, field="experiment_design", keys=_DESIGN_KEYS)
    if design["schema_version"] != "cuda-evidence/experiment-design-v1":
        raise ValueError("experiment_design.schema_version is unsupported")
    if design["formal"] is not True:
        raise ValueError("experiment_design.formal must be true")

    schedule = design["schedule"]
    if not isinstance(schedule, list) or len(schedule) < 2:
        raise ValueError("experiment_design.schedule must contain at least two pairs")
    pair_ids = set()
    orders = []
    for index, item in enumerate(schedule):
        row = _closed_mapping(
            item, field=f"schedule[{index}]", keys={"pair_id", "order"}
        )
        pair_id = _nonempty_string(row["pair_id"], f"schedule[{index}].pair_id")
        if not _PAIR_ID.fullmatch(pair_id) or pair_id in pair_ids:
            raise ValueError("schedule pair_id values must be safe and unique")
        pair_ids.add(pair_id)
        if row["order"] not in {"AB", "BA"}:
            raise ValueError("schedule order must be AB or BA")
        orders.append(row["order"])
    if abs(orders.count("AB") - orders.count("BA")) > 1:
        raise ValueError("formal schedule must be position balanced")

    _nonempty_string(design["experimental_unit"], "experimental_unit")
    if design["aggregation"] not in {
        "median_paired_improvement",
        "mean_paired_improvement",
        "ratio_of_sums",
    }:
        raise ValueError("aggregation is unsupported")
    if design["resampling_unit"] != "pair":
        raise ValueError("formal resampling_unit must be pair")

    ci = _closed_mapping(
        design["ci"],
        field="ci",
        keys={"method", "confidence", "samples", "seed"},
    )
    if ci["method"] != "paired_bootstrap":
        raise ValueError("ci.method must be paired_bootstrap")
    confidence = _finite_nonnegative(ci["confidence"], "ci.confidence")
    if not 0 < confidence < 1:
        raise ValueError("ci.confidence must be between zero and one")
    _positive_int(ci["samples"], "ci.samples")
    if isinstance(ci["seed"], bool) or not isinstance(ci["seed"], int):
        raise ValueError("ci.seed must be an integer")

    minimum = _positive_int(design["min_valid_pairs"], "min_valid_pairs", minimum=2)
    wins = _positive_int(design["wins_required"], "wins_required")
    if minimum > len(schedule) or wins > len(schedule):
        raise ValueError("pair requirements cannot exceed schedule length")
    _validate_guardrails(design["guardrails"])
    if design["exclusion_policy"] != "no_exclusion":
        raise ValueError("formal exclusion_policy must be no_exclusion")

    retry = _closed_mapping(
        design["retry_policy"],
        field="retry_policy",
        keys={"role_retries", "whole_pair_only", "allowed_reasons"},
    )
    if retry["role_retries"] != 0 or isinstance(retry["role_retries"], bool):
        raise ValueError("formal role_retries must be zero")
    if retry["whole_pair_only"] is not True:
        raise ValueError("formal retries must be whole-pair only")
    if retry["allowed_reasons"] != ["pre_measurement_infrastructure_failure"]:
        raise ValueError("formal retry reasons must be pre-measurement only")

    return copy.deepcopy(design)


def schedule_orders(value) -> list[str]:
    """Return the already-frozen pair orders after full design validation."""
    design = validate_frozen_design(value)
    return [row["order"] for row in design["schedule"]]
