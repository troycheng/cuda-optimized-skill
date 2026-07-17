#!/usr/bin/env python3
"""Paired real-workload measurement and objective evaluation."""

from __future__ import annotations

import copy
import math
import statistics
import sys
import time
from collections.abc import Mapping
from numbers import Real
from pathlib import Path


# Keep sibling imports reliable for direct CLI and importlib file loading.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import paired_stats  # noqa: E402
from experiment_design import balanced_pair_orders  # noqa: E402
from workload_adapter import (  # noqa: E402
    WorkloadSpec,
    run_spec_once,
    validate_objective,
)


DEFAULT_TIMEOUT = None
DEFAULT_BOOTSTRAP_SAMPLES = 10000
_RESULT_FIELDS = {"role", "case", "validation", "benchmark", "objective"}


def _nonnegative_int(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite_real(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite real number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite real number") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite real number")
    return number


def _json_copy(value, field: str = "value"):
    """Return a detached strict JSON value."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} numbers must be finite")
        return value
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} mappings must use string keys")
            normalized[key] = _json_copy(item, f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _json_copy(item, f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{field} must contain only JSON-compatible values")


def _attempt_record(attempt: int, status: str, error=None) -> dict:
    if error is None:
        return {
            "attempt": attempt,
            "status": status,
            "error_type": None,
            "error": None,
        }
    # Exception text may contain credentials or unbounded command output.  The
    # type plus attempt number is enough to diagnose retry behavior safely.
    error_type = type(error).__name__[:128]
    return {
        "attempt": attempt,
        "status": status,
        "error_type": error_type,
        "error": "workload attempt failed",
    }


def _validated_observation(raw, *, role: str) -> tuple[dict, bool | dict]:
    if not isinstance(raw, Mapping):
        raise ValueError("runner result must be a mapping")
    missing = sorted(_RESULT_FIELDS - set(raw))
    if missing:
        raise ValueError(f"runner result missing required field: {missing[0]}")
    if raw["role"] != role:
        raise ValueError("runner result role does not match requested role")
    validation = raw["validation"]
    if type(validation) is bool:
        passed = validation
    elif isinstance(validation, Mapping) and type(validation.get("valid")) is bool:
        passed = validation["valid"]
    else:
        raise ValueError(
            "validation must be literal True or a mapping with literal valid"
        )
    if not passed:
        raise ValueError("workload validation failed")
    if not isinstance(raw["benchmark"], Mapping):
        raise ValueError("benchmark metrics must be a mapping")
    if not isinstance(raw["objective"], Mapping):
        raise ValueError("objective must be a mapping")
    metrics = _json_copy(raw["benchmark"], "benchmark")
    normalized_validation = _json_copy(validation, "validation")
    return metrics, normalized_validation


def measure_candidate(
    workload,
    candidate,
    *,
    role="candidate",
    case=None,
    retries=2,
    timeout=DEFAULT_TIMEOUT,
    deadline_epoch=None,
    runner=None,
) -> dict:
    """Measure one role, retrying ordinary exceptions under one safe contract."""
    retry_count = _nonnegative_int(retries, "retries")
    if not isinstance(role, str) or not role.strip():
        raise ValueError("role must be a non-empty string")
    selected_runner = run_spec_once if runner is None else runner
    if not callable(selected_runner):
        raise ValueError("runner must be callable")
    deadline = (
        None
        if deadline_epoch is None
        else _finite_real(deadline_epoch, "deadline_epoch")
    )

    records = []
    for attempt in range(1, retry_count + 2):
        effective_timeout = timeout
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining <= 0.0:
                error = TimeoutError("workload evaluation deadline expired")
                records.append(_attempt_record(attempt, "failed", error))
                return {
                    "status": "failed",
                    "role": role,
                    "case": _json_copy(case, "case"),
                    "metrics": None,
                    "validation": None,
                    "attempts": attempt,
                    "attempt_records": records,
                    "failure": {
                        "error_type": "TimeoutError",
                        "error": "workload attempt failed",
                    },
                }
            if timeout is not None:
                effective_timeout = min(float(timeout), remaining)
        try:
            raw = selected_runner(
                workload,
                candidate=copy.deepcopy(candidate),
                role=role,
                case=copy.deepcopy(case),
                timeout=effective_timeout,
            )
            metrics, validation = _validated_observation(raw, role=role)
        except Exception as error:
            records.append(_attempt_record(attempt, "failed", error))
            if attempt <= retry_count:
                continue
            final_record = records[-1]
            return {
                "status": "failed",
                "role": role,
                "case": _json_copy(case, "case"),
                "metrics": None,
                "validation": None,
                "attempts": attempt,
                "attempt_records": records,
                "failure": {
                    "error_type": final_record["error_type"],
                    "error": final_record["error"],
                },
            }
        records.append(_attempt_record(attempt, "success"))
        return {
            "status": "measured",
            "role": role,
            "case": _json_copy(case, "case"),
            "metrics": metrics,
            "validation": validation,
            "attempts": attempt,
            "attempt_records": records,
        }

    raise AssertionError("unreachable")


def _failed_evaluation(base: dict, pairs: list[dict], reason: str) -> dict:
    result = dict(base)
    result.update(
        {
            "status": "workload_failed",
            "reason": reason,
            "pairs": pairs,
            "primary": {
                "status": "invalid",
                "statistic": "median_paired_improvement_pct",
                "estimate_pct": None,
                "ci_low_pct": None,
                "ci_high_pct": None,
            },
            "constraints": [],
        }
    )
    for pair in pairs:
        failures = pair.get("failures")
        if not isinstance(failures, Mapping):
            continue
        for role in ("baseline", "candidate"):
            diagnostic = failures.get(role)
            if isinstance(diagnostic, Mapping):
                result["failure"] = {
                    "block": pair.get("block"),
                    "role": role,
                    **copy.deepcopy(dict(diagnostic)),
                }
                return result
    return result


def _metric(metrics: Mapping, name: str, field: str) -> float:
    if name not in metrics:
        raise ValueError(f"{field}.{name} is missing")
    return _finite_real(metrics[name], f"{field}.{name}")


def _error_diagnostic(error: Exception) -> dict:
    return {
        "error_type": type(error).__name__[:128],
        "reason": str(error)[:512],
    }


def _validate_statistical_evidence(value, field: str) -> None:
    if value is None or isinstance(value, str):
        return
    if isinstance(value, bool):
        raise ValueError(f"{field} must not contain boolean numeric evidence")
    if isinstance(value, Real):
        _finite_real(value, field)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_statistical_evidence(item, f"{field}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_statistical_evidence(item, f"{field}[{index}]")
        return
    raise ValueError(f"{field} must contain JSON statistical evidence")


def _failed_statistics(base: dict, pairs: list[dict], error: Exception) -> dict:
    diagnostic = _error_diagnostic(error)
    for pair in pairs:
        pair["valid"] = False
        pair["invalid_reason"] = copy.deepcopy(diagnostic)
    result = _failed_evaluation(
        base, pairs, "statistical aggregation produced invalid evidence"
    )
    result["failure"] = diagnostic
    return result


def evaluate_pairs(
    workload,
    baseline,
    candidate,
    *,
    blocks,
    retries=2,
    seed=0,
    timeout=DEFAULT_TIMEOUT,
    deadline_epoch=None,
    confidence=0.95,
    bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
    runner=None,
) -> dict:
    """Measure position-balanced AB/BA blocks and evaluate the frozen objective."""
    block_count = _positive_int(blocks, "blocks")
    retry_count = _nonnegative_int(retries, "retries")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    confidence_value = _finite_real(confidence, "confidence")
    if not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    bootstrap_count = _positive_int(bootstrap_samples, "bootstrap_samples")
    if not hasattr(workload, "objective") or not hasattr(workload, "cases"):
        raise ValueError("workload must provide objective and cases")
    objective = validate_objective(workload.objective)
    objective_snapshot = _json_copy(objective, "objective")
    cases = tuple(workload.cases)
    pairs = []
    schedule = balanced_pair_orders(block_count, seed=seed)

    for block, order in enumerate(schedule):
        case = None if not cases else _json_copy(cases[block % len(cases)], "case")
        sequence = (
            (("baseline", baseline), ("candidate", candidate))
            if order == "AB"
            else (("candidate", candidate), ("baseline", baseline))
        )
        measurements = {}
        for role, selected_candidate in sequence:
            measurements[role] = measure_candidate(
                workload,
                selected_candidate,
                role=role,
                case=case,
                retries=retry_count,
                timeout=timeout,
                deadline_epoch=deadline_epoch,
                runner=runner,
            )
        baseline_result = measurements["baseline"]
        candidate_result = measurements["candidate"]
        valid = (
            baseline_result["status"] == "measured"
            and candidate_result["status"] == "measured"
        )
        pair = {
            "block": block,
            "order": order,
            "case": copy.deepcopy(case),
            "baseline_metrics": copy.deepcopy(baseline_result["metrics"]),
            "candidate_metrics": copy.deepcopy(candidate_result["metrics"]),
            "valid": valid,
            "attempts": {
                "baseline": baseline_result["attempts"],
                "candidate": candidate_result["attempts"],
            },
            "attempt_records": {
                "baseline": copy.deepcopy(baseline_result["attempt_records"]),
                "candidate": copy.deepcopy(candidate_result["attempt_records"]),
            },
        }
        failures = {
            role: copy.deepcopy(result.get("failure"))
            for role, result in measurements.items()
            if result["status"] == "failed"
        }
        if failures:
            pair["failures"] = failures
        pairs.append(pair)

    classified = classify_recorded_pairs(
        objective_snapshot,
        pairs,
        confidence=confidence_value,
        bootstrap_samples=bootstrap_count,
        seed=seed,
    )
    classified["blocks"] = block_count
    return classified


def classify_recorded_pairs(
    objective,
    pairs,
    *,
    confidence=0.95,
    bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
    seed=0,
) -> dict:
    """Recompute workload statistics from frozen raw paired observations."""
    objective = validate_objective(objective)
    objective_snapshot = _json_copy(objective, "objective")
    confidence_value = _finite_real(confidence, "confidence")
    if not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    bootstrap_count = _positive_int(bootstrap_samples, "bootstrap_samples")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(pairs, (str, bytes, bytearray, Mapping)):
        raise ValueError("pairs must be a sequence")
    try:
        recorded_pairs = _json_copy(list(pairs), "pairs")
    except TypeError as error:
        raise ValueError("pairs must be a sequence") from error
    if not all(isinstance(pair, Mapping) for pair in recorded_pairs):
        raise ValueError("pairs must contain mappings")
    base = {
        "objective": objective_snapshot,
        "confidence": confidence_value,
        "bootstrap_samples": bootstrap_count,
        "seed": seed,
        "pairs": recorded_pairs,
    }
    if any(pair.get("valid") is not True for pair in recorded_pairs):
        return _failed_evaluation(
            base,
            recorded_pairs,
            "one or more workload roles exhausted retries",
        )

    primary = objective["primary_metric"]
    metric_names = [primary["name"]] + [
        constraint["name"] for constraint in objective["constraints"]
    ]
    numeric = []
    metric_failed = False
    for pair in recorded_pairs:
        values = {"baseline": {}, "candidate": {}}
        errors = []
        for metric_name in metric_names:
            for role, key in (
                ("baseline", "baseline_metrics"),
                ("candidate", "candidate_metrics"),
            ):
                try:
                    value = _metric(pair[key], metric_name, key)
                    if role == "baseline" and value == 0.0:
                        raise ValueError(f"{key}.{metric_name} must be nonzero")
                    values[role][metric_name] = value
                except (KeyError, TypeError, ValueError) as error:
                    errors.append(_error_diagnostic(error))
        if errors:
            pair["valid"] = False
            pair["metric_errors"] = errors
            metric_failed = True
        numeric.append(values)
    if metric_failed:
        return _failed_evaluation(base, recorded_pairs, "objective metrics are invalid")

    primary_pairs = []
    constraint_regressions = {
        constraint["name"]: [] for constraint in objective["constraints"]
    }
    derived_failed = False
    for pair, values in zip(recorded_pairs, numeric):
        primary_baseline = values["baseline"][primary["name"]]
        primary_candidate = values["candidate"][primary["name"]]
        primary_pairs.append(
            {"baseline": primary_baseline, "candidate": primary_candidate, "valid": True}
        )
        errors = []
        try:
            paired_stats.improvement_pct(
                primary_baseline, primary_candidate, primary["direction"]
            )
        except ValueError as error:
            errors.append(_error_diagnostic(error))
        for constraint in objective["constraints"]:
            name = constraint["name"]
            baseline_value = values["baseline"][name]
            candidate_value = values["candidate"][name]
            try:
                regression = _finite_real(
                    (candidate_value - baseline_value)
                    / abs(baseline_value)
                    * 100.0,
                    f"constraint {name} regression_pct",
                )
                constraint_regressions[name].append(regression)
            except (ArithmeticError, ValueError) as error:
                errors.append(_error_diagnostic(error))
        if errors:
            pair["valid"] = False
            pair["metric_errors"] = errors
            derived_failed = True
    if derived_failed:
        return _failed_evaluation(
            base, recorded_pairs, "objective metric derivation is invalid"
        )

    try:
        primary_statistics = paired_stats.classify_pairs(
            primary_pairs,
            direction=primary["direction"],
            min_effect_pct=objective["min_effect_pct"],
            confidence=confidence_value,
            bootstrap_samples=bootstrap_count,
            seed=seed,
        )
        constraint_statistics = []
        for index, constraint in enumerate(objective["constraints"]):
            name = constraint["name"]
            cap = constraint["max_regression_pct"]
            regressions = constraint_regressions[name]
            estimate = statistics.median(regressions)
            ci_low, ci_high = paired_stats.bootstrap_median_ci(
                regressions,
                confidence=confidence_value,
                samples=bootstrap_count,
                seed=seed + index + 1,
            )
            status = (
                "passed"
                if ci_high <= cap
                else "failed" if ci_low > cap else "inconclusive"
            )
            constraint_statistics.append(
                {
                    "name": name,
                    "max_regression_pct": cap,
                    "cap_pct": cap,
                    "estimate_pct": estimate,
                    "ci_low_pct": ci_low,
                    "ci_high_pct": ci_high,
                    "status": status,
                    "values_pct": regressions,
                }
            )
        _validate_statistical_evidence(primary_statistics, "primary")
        _validate_statistical_evidence(constraint_statistics, "constraints")
    except (ArithmeticError, ValueError) as error:
        return _failed_statistics(base, recorded_pairs, error)
    return {
        **base,
        "status": "evaluated",
        "primary": primary_statistics,
        "constraints": constraint_statistics,
    }


__all__ = [
    "WorkloadSpec",
    "classify_recorded_pairs",
    "evaluate_pairs",
    "measure_candidate",
]
