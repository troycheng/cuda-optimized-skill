#!/usr/bin/env python3
"""Paired-effect statistics and bootstrap confidence classification."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping
from numbers import Real


def _finite_real(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite real number")
    return parsed


def _validate_direction(direction: str) -> None:
    if direction not in ("lower", "higher"):
        raise ValueError("direction must be 'lower' or 'higher'")


def _validate_confidence(confidence: float) -> float:
    parsed = _finite_real(confidence, "confidence")
    if not 0.0 < parsed < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return parsed


def _validate_samples(samples: int, name: str = "samples") -> int:
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return samples


def _linear_percentile(sorted_values: list[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return lower + (upper - lower) * fraction


def improvement_pct(baseline: float, candidate: float, direction: str) -> float:
    """Return the candidate's percentage improvement over its paired baseline."""
    baseline_value = _finite_real(baseline, "baseline")
    if baseline_value == 0.0:
        raise ValueError("baseline must be nonzero")
    candidate_value = _finite_real(candidate, "candidate")
    _validate_direction(direction)

    if direction == "lower":
        delta = baseline_value - candidate_value
    else:
        delta = candidate_value - baseline_value
    improvement = delta / abs(baseline_value) * 100.0
    if not math.isfinite(improvement):
        raise ValueError("improvement_pct must be finite")
    return improvement


def bootstrap_median_ci(
    values,
    *,
    confidence: float = 0.95,
    samples: int = 10000,
    seed=0,
) -> tuple[float, float]:
    """Bootstrap a two-sided confidence interval for the sample median."""
    try:
        raw_values = list(values)
    except TypeError as exc:
        raise ValueError("values must be a non-empty numeric sequence") from exc
    if not raw_values:
        raise ValueError("values must be a non-empty numeric sequence")

    clean_values = [_finite_real(value, "values") for value in raw_values]
    confidence_value = _validate_confidence(confidence)
    sample_count = _validate_samples(samples)

    rng = random.Random(seed)
    bootstrap_medians = sorted(
        statistics.median(rng.choices(clean_values, k=len(clean_values)))
        for _ in range(sample_count)
    )
    tail_probability = (1.0 - confidence_value) / 2.0
    low = _linear_percentile(bootstrap_medians, tail_probability)
    high = _linear_percentile(bootstrap_medians, 1.0 - tail_probability)
    return (low, high) if low <= high else (high, low)


def classify_pairs(
    pairs,
    *,
    direction: str,
    min_effect_pct: float,
    confidence: float = 0.95,
    bootstrap_samples: int = 10000,
    min_valid_pairs: int = 2,
    seed=0,
) -> dict:
    """Classify paired observations using their median improvement and CI."""
    _validate_direction(direction)
    min_effect = _finite_real(min_effect_pct, "min_effect_pct")
    if min_effect < 0.0:
        raise ValueError("min_effect_pct must be greater than or equal to zero")
    confidence_value = _validate_confidence(confidence)
    sample_count = _validate_samples(bootstrap_samples, "bootstrap_samples")
    minimum_pairs = _validate_samples(min_valid_pairs, "min_valid_pairs")

    if isinstance(pairs, (str, bytes, bytearray)):
        raise ValueError("pairs must be a non-string iterable of mappings")
    try:
        pair_iterator = iter(pairs)
    except TypeError as exc:
        raise ValueError("pairs must be a non-string iterable of mappings") from exc

    improvements = []
    invalid_pairs = 0
    for index, pair in enumerate(pair_iterator):
        if not isinstance(pair, Mapping):
            raise ValueError(f"pairs[{index}] must be a mapping")
        if "valid" in pair and type(pair["valid"]) is not bool:
            raise ValueError(f"pairs[{index}].valid must be a bool")
        if pair.get("valid", True) is False:
            invalid_pairs += 1
            continue
        if "baseline" not in pair:
            raise ValueError(f"pairs[{index}].baseline is required for a valid pair")
        if "candidate" not in pair:
            raise ValueError(f"pairs[{index}].candidate is required for a valid pair")
        improvements.append(
            improvement_pct(pair["baseline"], pair["candidate"], direction)
        )

    result = {
        "status": "inconclusive",
        "statistic": "median_paired_improvement_pct",
        "direction": direction,
        "min_effect_pct": min_effect,
        "confidence": confidence_value,
        "estimate_pct": None,
        "ci_low_pct": None,
        "ci_high_pct": None,
        "valid_pairs": len(improvements),
        "invalid_pairs": invalid_pairs,
        "improvements_pct": improvements,
    }
    if not improvements:
        return result

    estimate = statistics.median(improvements)
    ci_low, ci_high = bootstrap_median_ci(
        improvements,
        confidence=confidence_value,
        samples=sample_count,
        seed=seed,
    )
    if len(improvements) < minimum_pairs:
        status = "inconclusive"
    elif (min_effect == 0.0 and ci_low > 0.0) or (
        min_effect > 0.0 and ci_low >= min_effect
    ):
        status = "confirmed_win"
    elif (min_effect == 0.0 and ci_high < 0.0) or (
        min_effect > 0.0 and ci_high <= -min_effect
    ):
        status = "confirmed_loss"
    else:
        status = "inconclusive"

    result.update(
        {
            "status": status,
            "estimate_pct": estimate,
            "ci_low_pct": ci_low,
            "ci_high_pct": ci_high,
        }
    )
    return result
