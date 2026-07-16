#!/usr/bin/env python3
"""Terminal decision engine for kernel-only and full optimization modes."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path


# Keep sibling imports reliable for direct CLI and importlib file loading.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from workload_adapter import validate_objective  # noqa: E402


TERMINAL_STATUSES = {
    "rejected_compile",
    "rejected_correctness",
    "rejected_constraint",
    "confirmed_loss",
    "inconclusive",
    "kernel_only_win",
    "end_to_end_win",
    "pareto_frontier",
}

_KERNEL_STATUSES = {
    "rejected_compile",
    "rejected_correctness",
    "confirmed_loss",
    "inconclusive",
    "invalid",
    "no_confirmed_kernel_win",
    "confirmed_win",
    "kernel_only_win",
}
_WORKLOAD_STATUSES = {"evaluated", "workload_failed"}
_PRIMARY_STATUSES = {
    "confirmed_win",
    "confirmed_loss",
    "inconclusive",
    "invalid",
}
_CONSTRAINT_STATUSES = {"passed", "failed", "inconclusive"}
_PARETO_SCHEMA = "cuda-kernel-optimizer/pareto-v1"
_STATISTIC_FIELDS = {
    "status",
    "statistic",
    "direction",
    "min_effect_pct",
    "confidence",
    "estimate_pct",
    "ci_low_pct",
    "ci_high_pct",
    "valid_pairs",
    "invalid_pairs",
    "improvements_pct",
}
_PAIRED_STATISTIC = "median_paired_improvement_pct"
_CONSTRAINT_RESULT_FIELDS = {
    "name",
    "max_regression_pct",
    "cap_pct",
    "estimate_pct",
    "ci_low_pct",
    "ci_high_pct",
    "status",
    "values_pct",
}


def _literal_status(value, allowed: set[str], field: str) -> str:
    if type(value) is not str or value not in allowed:
        raise ValueError(f"{field} must be a recognized literal status")
    return value


def _mapping(value, field: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _json_copy(value, field: str, active: set[int] | None = None):
    """Validate strict JSON evidence and return a detached normalized copy."""
    if active is None:
        active = set()
    if value is None or type(value) in {bool, str, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field} numbers must be finite")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{field} must not contain a cycle")
        active.add(identity)
        try:
            result = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise ValueError(f"{field} mappings must use string keys")
                result[key] = _json_copy(item, f"{field}.{key}", active)
            return result
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{field} must not contain a cycle")
        active.add(identity)
        try:
            return [
                _json_copy(item, f"{field}[{index}]", active)
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(identity)
    raise ValueError(f"{field} must contain only strict JSON values")


def _finite_real(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite real number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite real number") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite real number")
    return number


def _literal_integer(value, field: str, *, positive: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be a {qualifier} integer")
    if (positive and value <= 0) or (not positive and value < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be a {qualifier} integer")
    return value


def _paired_status(ci_low: float, ci_high: float, min_effect: float) -> str:
    if (min_effect == 0.0 and ci_low > 0.0) or (
        min_effect > 0.0 and ci_low >= min_effect
    ):
        return "confirmed_win"
    if (min_effect == 0.0 and ci_high < 0.0) or (
        min_effect > 0.0 and ci_high <= -min_effect
    ):
        return "confirmed_loss"
    return "inconclusive"


def _validate_paired_statistics(
    value,
    field: str,
    *,
    expected_direction: str | None = None,
    expected_min_effect: float | None = None,
    required_status: str | None = None,
) -> dict:
    statistics = _mapping(value, field)
    missing = sorted(_STATISTIC_FIELDS - set(statistics))
    if missing:
        raise ValueError(f"{field} missing required field: {missing[0]}")
    unknown = sorted(set(statistics) - _STATISTIC_FIELDS)
    if unknown:
        raise ValueError(f"{field} contains unknown field: {unknown[0]}")
    if statistics["statistic"] != _PAIRED_STATISTIC:
        raise ValueError(f"{field}.statistic must be {_PAIRED_STATISTIC}")

    direction = statistics["direction"]
    if type(direction) is not str or direction not in {"lower", "higher"}:
        raise ValueError(f"{field}.direction must be 'lower' or 'higher'")
    if expected_direction is not None and direction != expected_direction:
        raise ValueError(f"{field}.direction must match objective primary direction")

    min_effect = _finite_real(statistics["min_effect_pct"], f"{field}.min_effect_pct")
    if min_effect < 0.0:
        raise ValueError(f"{field}.min_effect_pct must be non-negative")
    if expected_min_effect is not None and min_effect != expected_min_effect:
        raise ValueError(f"{field}.min_effect_pct must match objective")

    confidence = _finite_real(statistics["confidence"], f"{field}.confidence")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"{field}.confidence must be between zero and one")
    estimate = _finite_real(statistics["estimate_pct"], f"{field}.estimate_pct")
    ci_low = _finite_real(statistics["ci_low_pct"], f"{field}.ci_low_pct")
    ci_high = _finite_real(statistics["ci_high_pct"], f"{field}.ci_high_pct")
    if ci_low > ci_high:
        raise ValueError(f"{field} confidence interval must be ordered")

    valid_pairs = _literal_integer(
        statistics["valid_pairs"], f"{field}.valid_pairs", positive=True
    )
    invalid_pairs = _literal_integer(
        statistics["invalid_pairs"], f"{field}.invalid_pairs", positive=False
    )
    improvements = statistics["improvements_pct"]
    if not isinstance(improvements, list):
        raise ValueError(f"{field}.improvements_pct must be a finite list")
    normalized_improvements = [
        _finite_real(item, f"{field}.improvements_pct[{index}]")
        for index, item in enumerate(improvements)
    ]
    if len(normalized_improvements) < valid_pairs:
        raise ValueError(
            f"{field}.improvements_pct must contain at least valid_pairs values"
        )

    status = _literal_status(
        statistics["status"],
        {"confirmed_win", "confirmed_loss", "inconclusive"},
        f"{field}.status",
    )
    derived_status = _paired_status(ci_low, ci_high, min_effect)
    if status != derived_status:
        raise ValueError(f"{field}.status contradicts its confidence interval")
    if required_status is not None and status != required_status:
        raise ValueError(f"{field}.status must be {required_status}")

    return {
        "status": status,
        "statistic": _PAIRED_STATISTIC,
        "direction": direction,
        "min_effect_pct": min_effect,
        "confidence": confidence,
        "estimate_pct": estimate,
        "ci_low_pct": ci_low,
        "ci_high_pct": ci_high,
        "valid_pairs": valid_pairs,
        "invalid_pairs": invalid_pairs,
        "improvements_pct": normalized_improvements,
    }


def _normalize_mode(mode) -> str:
    if type(mode) is not str or mode not in {"full", "kernel-only"}:
        raise ValueError("mode must be 'full' or 'kernel-only'")
    return mode


def _validate_workload(workload) -> Mapping | None:
    if workload is None:
        return None
    workload = _mapping(workload, "workload")
    status = _literal_status(
        workload.get("status"), _WORKLOAD_STATUSES, "workload.status"
    )
    normalized = dict(workload)
    if status != "evaluated":
        primary = workload.get("primary")
        if primary is not None:
            primary = _mapping(primary, "workload.primary")
            _literal_status(
                primary.get("status"), _PRIMARY_STATUSES, "workload.primary.status"
            )
        return normalized

    if "objective" not in workload:
        raise ValueError("incomplete workload evidence: objective is required")
    try:
        objective = validate_objective(workload["objective"])
    except ValueError as error:
        raise ValueError(f"incomplete workload evidence: {error}") from error
    if "primary" not in workload:
        raise ValueError("incomplete workload evidence: primary is required")
    primary = _validate_paired_statistics(
        workload["primary"],
        "workload.primary",
        expected_direction=objective["primary_metric"]["direction"],
        expected_min_effect=objective["min_effect_pct"],
    )
    if "constraints" not in workload or workload["constraints"] is None:
        raise ValueError("incomplete workload evidence: constraints are required")
    constraints = _validate_workload_constraints(
        workload["constraints"], objective["constraints"]
    )
    normalized["objective"] = objective
    normalized["primary"] = primary
    normalized["constraints"] = constraints
    return normalized


def _validate_constraints(constraints) -> list[dict]:
    if constraints is None:
        raise ValueError("constraints must be a sequence of mappings")
    if isinstance(constraints, (str, bytes, bytearray, Mapping)):
        raise ValueError("constraints must be a sequence of mappings")
    try:
        raw_constraints = list(constraints)
    except TypeError as error:
        raise ValueError("constraints must be a sequence of mappings") from error
    normalized = []
    seen = set()
    for index, constraint in enumerate(raw_constraints):
        constraint = _mapping(constraint, f"constraints[{index}]")
        name = constraint.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"constraints[{index}].name must be non-empty")
        name = name.strip()
        if name in seen:
            raise ValueError(f"constraints contains duplicate name: {name}")
        seen.add(name)
        _literal_status(
            constraint.get("status"),
            _CONSTRAINT_STATUSES,
            f"constraints[{index}].status",
        )
        item = dict(constraint)
        item["name"] = name
        normalized.append(item)
    return normalized


def _validate_workload_constraints(constraints, declared_constraints) -> list[dict]:
    results = _validate_constraints(constraints)
    declared = {constraint["name"]: constraint for constraint in declared_constraints}
    result_names = {result["name"] for result in results}
    if result_names != set(declared):
        raise ValueError(
            "incomplete workload evidence: objective and result constraints must match"
        )

    normalized = []
    for index, result in enumerate(results):
        field = f"workload.constraints[{index}]"
        missing = sorted(_CONSTRAINT_RESULT_FIELDS - set(result))
        if missing:
            raise ValueError(f"{field} missing required field: {missing[0]}")
        cap = declared[result["name"]]["max_regression_pct"]
        declared_cap = _finite_real(
            result["max_regression_pct"], f"{field}.max_regression_pct"
        )
        result_cap = _finite_real(result["cap_pct"], f"{field}.cap_pct")
        if declared_cap != cap or result_cap != cap:
            raise ValueError(f"{field} cap must match its objective constraint")
        estimate = _finite_real(result["estimate_pct"], f"{field}.estimate_pct")
        ci_low = _finite_real(result["ci_low_pct"], f"{field}.ci_low_pct")
        ci_high = _finite_real(result["ci_high_pct"], f"{field}.ci_high_pct")
        if ci_low > ci_high:
            raise ValueError(f"{field} confidence interval must be ordered")
        values = result["values_pct"]
        if not isinstance(values, list):
            raise ValueError(f"{field}.values_pct must be a finite list")
        normalized_values = [
            _finite_real(item, f"{field}.values_pct[{value_index}]")
            for value_index, item in enumerate(values)
        ]
        if ci_high <= cap:
            derived_status = "passed"
        elif ci_low > cap:
            derived_status = "failed"
        else:
            derived_status = "inconclusive"
        if result["status"] != derived_status:
            raise ValueError(f"{field}.status contradicts its confidence interval")

        item = dict(result)
        item.update(
            {
                "max_regression_pct": cap,
                "cap_pct": cap,
                "estimate_pct": estimate,
                "ci_low_pct": ci_low,
                "ci_high_pct": ci_high,
                "values_pct": normalized_values,
            }
        )
        normalized.append(item)
    return normalized


def _validate_pareto(pareto) -> dict | None:
    if pareto is None:
        return None
    pareto = _mapping(pareto, "pareto")
    if set(pareto) != {"schema", "status", "objectives"}:
        raise ValueError("pareto must contain only schema, status, and objectives")
    if pareto["schema"] != _PARETO_SCHEMA:
        raise ValueError(f"pareto.schema must be {_PARETO_SCHEMA}")
    if pareto["status"] != "non_dominated":
        raise ValueError("pareto.status must be non_dominated")
    objectives = pareto["objectives"]
    if isinstance(objectives, (str, bytes, bytearray, Mapping)):
        raise ValueError("pareto.objectives must be a sequence")
    try:
        objectives = list(objectives)
    except TypeError as error:
        raise ValueError("pareto.objectives must be a sequence") from error
    if len(objectives) < 2:
        raise ValueError("pareto.objectives must contain at least two objectives")
    normalized_objectives = []
    names = set()
    outcomes = set()
    for index, objective in enumerate(objectives):
        objective = _mapping(objective, f"pareto.objectives[{index}]")
        if set(objective) != {"name", "outcome"}:
            raise ValueError("pareto objectives contain only name and outcome")
        name = objective.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"pareto.objectives[{index}].name must be non-empty")
        name = name.strip()
        if name in names:
            raise ValueError(f"pareto contains duplicate objective: {name}")
        names.add(name)
        outcome = objective.get("outcome")
        if type(outcome) is not str or outcome not in {
            "improved",
            "regressed",
            "unchanged",
        }:
            raise ValueError(
                f"pareto.objectives[{index}].outcome must be a literal outcome"
            )
        outcomes.add(outcome)
        normalized_objectives.append({"name": name, "outcome": outcome})
    if not {"improved", "regressed"}.issubset(outcomes):
        raise ValueError(
            "non_dominated pareto evidence requires an explicit tradeoff"
        )
    return {
        "schema": _PARETO_SCHEMA,
        "status": "non_dominated",
        "objectives": normalized_objectives,
    }


def _kernel_statistics(kernel: Mapping) -> dict:
    if "statistics" not in kernel:
        raise ValueError("kernel.statistics is required for an inner win")
    return _validate_paired_statistics(
        kernel["statistics"],
        "kernel.statistics",
        required_status="confirmed_win",
    )


def _result(status: str, reason: str, mode: str, evidence: dict, **fields) -> dict:
    if status not in TERMINAL_STATUSES:
        raise AssertionError("decision engine produced a non-terminal status")
    result = {
        "status": status,
        "reason": reason,
        "mode": mode,
        "evidence": evidence,
        **fields,
    }
    normalized = _json_copy(result, "decision")
    try:
        json.dumps(normalized, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"decision result must be strict JSON: {error}") from error
    return normalized


def decide(*, mode, kernel, workload=None, constraints=None, pareto=None) -> dict:
    """Return one terminal decision without mutating or weighting evidence."""
    normalized_mode = _normalize_mode(mode)
    kernel = _json_copy(kernel, "kernel")
    if workload is not None:
        workload = _json_copy(workload, "workload")
    if constraints is not None:
        constraints = _json_copy(constraints, "constraints")
    if pareto is not None:
        pareto = _json_copy(pareto, "pareto")
    kernel = _mapping(kernel, "kernel")
    kernel_status = _literal_status(
        kernel.get("status"), _KERNEL_STATUSES, "kernel.status"
    )
    workload = _validate_workload(workload)
    workload_has_constraints = workload is not None and "constraints" in workload
    workload_constraints = (
        _validate_constraints(workload["constraints"])
        if workload_has_constraints
        else None
    )
    explicit_constraints = (
        _validate_constraints(constraints) if constraints is not None else None
    )
    if (
        workload_constraints is not None
        and explicit_constraints is not None
        and workload_constraints != explicit_constraints
    ):
        raise ValueError("conflicting constraints between workload and caller")
    if workload_constraints is not None:
        normalized_constraints = workload_constraints
    elif explicit_constraints is not None:
        normalized_constraints = explicit_constraints
    else:
        normalized_constraints = []
    normalized_pareto = _validate_pareto(pareto)
    evidence = {
        "kernel": dict(kernel),
        "workload": None if workload is None else dict(workload),
        "constraints": normalized_constraints,
        "pareto": normalized_pareto,
    }

    if kernel_status in {"rejected_compile", "rejected_correctness"}:
        return _result(
            kernel_status,
            f"kernel evaluation ended with {kernel_status}",
            normalized_mode,
            evidence,
        )
    if kernel_status == "confirmed_loss":
        return _result(
            "confirmed_loss",
            "paired kernel evidence confirms a loss",
            normalized_mode,
            evidence,
        )
    if kernel_status in {"inconclusive", "invalid", "no_confirmed_kernel_win"}:
        return _result(
            "inconclusive",
            "kernel evidence does not contain a confirmed inner-loop win",
            normalized_mode,
            evidence,
        )

    kernel_statistics = _kernel_statistics(kernel)
    if normalized_mode == "kernel-only":
        return _result(
            "kernel_only_win",
            "kernel-only mode confirms only the inner-loop result",
            normalized_mode,
            evidence,
            statistics=kernel_statistics,
        )

    if any(item["status"] == "failed" for item in normalized_constraints):
        return _result(
            "rejected_constraint",
            "at least one hard constraint is confirmed failed",
            normalized_mode,
            evidence,
            statistics=kernel_statistics,
            constraints=normalized_constraints,
        )
    if any(item["status"] != "passed" for item in normalized_constraints):
        return _result(
            "kernel_only_win",
            "hard-constraint evidence is inconclusive",
            normalized_mode,
            evidence,
            statistics=kernel_statistics,
            constraints=normalized_constraints,
        )
    if workload is None:
        return _result(
            "kernel_only_win",
            "no real-workload evaluation was supplied",
            normalized_mode,
            evidence,
            statistics=kernel_statistics,
        )
    if workload["status"] == "workload_failed":
        return _result(
            "kernel_only_win",
            "real-workload collection failed",
            normalized_mode,
            evidence,
            statistics=kernel_statistics,
        )
    primary = workload.get("primary")
    if primary is None or primary["status"] != "confirmed_win":
        return _result(
            "kernel_only_win",
            "real-workload primary metric is not a confirmed win",
            normalized_mode,
            evidence,
            statistics=kernel_statistics,
        )
    workload_statistics = _validate_paired_statistics(
        primary,
        "workload.primary",
        expected_direction=workload["objective"]["primary_metric"]["direction"],
        expected_min_effect=workload["objective"]["min_effect_pct"],
        required_status="confirmed_win",
    )
    if normalized_pareto is not None:
        return _result(
            "pareto_frontier",
            "explicit non-dominated multi-objective evidence shows a tradeoff",
            normalized_mode,
            evidence,
            statistics=kernel_statistics,
            workload_statistics=workload_statistics,
            pareto=normalized_pareto,
        )
    return _result(
        "end_to_end_win",
        "kernel and real-workload evidence confirm the candidate",
        normalized_mode,
        evidence,
        statistics=kernel_statistics,
        workload_statistics=workload_statistics,
        constraints=normalized_constraints,
    )


__all__ = ["TERMINAL_STATUSES", "decide"]
