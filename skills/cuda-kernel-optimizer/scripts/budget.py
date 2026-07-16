#!/usr/bin/env python3
"""Budget presets and deadline admission for optimizer work."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class BudgetPolicy:
    name: str
    max_seconds: int
    branches: int
    max_rounds: int
    min_pairs: int
    max_pairs: int
    outer_candidates: int
    max_cases: int | None
    sanitizer_mode: str
    reserve_seconds: int = 300


PRESETS = {
    "quick": BudgetPolicy("quick", 2700, 4, 2, 20, 50, 1, 3, "targeted"),
    "balanced": BudgetPolicy("balanced", 10800, 8, 4, 20, 100, 2, 10, "targeted"),
    "thorough": BudgetPolicy("thorough", 36000, 16, 8, 30, 200, 3, None, "full"),
}

_REQUIRED_POSITIVE_FIELDS = (
    "max_seconds",
    "branches",
    "max_rounds",
    "min_pairs",
    "max_pairs",
    "outer_candidates",
)
_OPTIONAL_OVERRIDE_FIELDS = {"max_cases", "sanitizer_mode", "reserve_seconds"}
_VALID_SANITIZER_MODES = {"targeted", "full"}


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_time(value: object, parameter: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{parameter} must be a finite number")
    try:
        numeric = float(value)
    except OverflowError:
        raise ValueError(f"{parameter} must be a finite number") from None
    if not math.isfinite(numeric):
        raise ValueError(f"{parameter} must be a finite number")
    return numeric


def _validate_policy(policy: BudgetPolicy) -> None:
    for field in _REQUIRED_POSITIVE_FIELDS:
        if not _is_positive_int(getattr(policy, field)):
            raise ValueError(f"{field} must be a positive integer")
    if policy.min_pairs > policy.max_pairs:
        raise ValueError("min_pairs must be less than or equal to max_pairs")
    if policy.max_cases is not None and not _is_positive_int(policy.max_cases):
        raise ValueError("max_cases must be a positive integer or None")
    if policy.sanitizer_mode not in _VALID_SANITIZER_MODES:
        raise ValueError(
            "sanitizer_mode must be one of: "
            + ", ".join(sorted(_VALID_SANITIZER_MODES))
        )
    if (
        not isinstance(policy.reserve_seconds, int)
        or isinstance(policy.reserve_seconds, bool)
        or policy.reserve_seconds < 0
    ):
        raise ValueError("reserve_seconds must be a non-negative integer")
    if policy.reserve_seconds >= policy.max_seconds:
        raise ValueError("reserve_seconds must be less than max_seconds")


def resolve_budget(name: str, **overrides: object) -> BudgetPolicy:
    """Return a validated copy of a preset with explicit overrides applied."""
    allowed = set(_REQUIRED_POSITIVE_FIELDS) | _OPTIONAL_OVERRIDE_FIELDS
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise TypeError(f"unknown budget override: {', '.join(unknown)}")

    if name == "custom":
        for field in _REQUIRED_POSITIVE_FIELDS:
            if field in overrides and not _is_positive_int(overrides[field]):
                raise ValueError(f"{field} must be a positive integer")
        missing = [field for field in _REQUIRED_POSITIVE_FIELDS if field not in overrides]
        if missing:
            raise ValueError(f"custom budget requires {', '.join(missing)}")
        policy = BudgetPolicy(
            name="custom",
            max_seconds=overrides["max_seconds"],
            branches=overrides["branches"],
            max_rounds=overrides["max_rounds"],
            min_pairs=overrides["min_pairs"],
            max_pairs=overrides["max_pairs"],
            outer_candidates=overrides["outer_candidates"],
            max_cases=overrides.get("max_cases"),
            sanitizer_mode=overrides.get("sanitizer_mode", "targeted"),
            reserve_seconds=overrides.get("reserve_seconds", 300),
        )
    else:
        if name not in PRESETS:
            raise ValueError(f"unknown budget preset: {name}")
        policy = replace(PRESETS[name], **overrides)

    _validate_policy(policy)
    return policy


@dataclass(frozen=True)
class BudgetClock:
    policy: BudgetPolicy
    started_at: float
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        _validate_time(self.started_at, "started_at")
        elapsed = _validate_time(self.elapsed_seconds, "elapsed_seconds")
        if elapsed < 0.0:
            raise ValueError("elapsed_seconds must be a non-negative finite number")

    def elapsed(self, *, now: float) -> float:
        current = _validate_time(now, "now")
        return self.elapsed_seconds + max(0.0, current - self.started_at)

    def can_start(self, *, now: float, estimated_seconds: float) -> bool:
        estimate = max(
            0.0, _validate_time(estimated_seconds, "estimated_seconds")
        )
        return estimate <= self.execution_seconds_available(now=now)

    def execution_seconds_available(self, *, now: float) -> float:
        return max(
            0.0,
            self.policy.max_seconds
            - self.policy.reserve_seconds
            - self.elapsed(now=now),
        )

    def remaining_seconds(self, *, now: float) -> float:
        return max(0.0, self.policy.max_seconds - self.elapsed(now=now))
