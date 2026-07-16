#!/usr/bin/env python3
"""Terminal decision engine for kernel-only and full optimization modes."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping


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
    "statistic",
    "estimate_pct",
    "ci_low_pct",
    "ci_high_pct",
    "status",
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


def _validate_win_statistics(value, field: str) -> dict:
    statistics = _mapping(value, field)
    missing = sorted(_STATISTIC_FIELDS - set(statistics))
    if missing:
        raise ValueError(f"{field} missing required field: {missing[0]}")
    statistic = statistics["statistic"]
    if not isinstance(statistic, str) or not statistic.strip():
        raise ValueError(f"{field}.statistic must be a non-empty string")
    _literal_status(statistics["status"], {"confirmed_win"}, f"{field}.status")
    for name in ("estimate_pct", "ci_low_pct", "ci_high_pct"):
        number = statistics[name]
        try:
            finite = math.isfinite(float(number))
        except (OverflowError, TypeError, ValueError):
            finite = False
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not finite
        ):
            raise ValueError(f"{field}.{name} must be a finite real number")
    normalized = dict(statistics)
    normalized["statistic"] = statistic.strip()
    return normalized


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
    primary = workload.get("primary")
    if primary is not None:
        primary = _mapping(primary, "workload.primary")
        primary_status = _literal_status(
            primary.get("status"), _PRIMARY_STATUSES, "workload.primary.status"
        )
        if primary_status == "confirmed_win":
            _validate_win_statistics(primary, "workload.primary")
            _validate_winning_workload_schema(workload)
    if status == "evaluated" and "constraints" in workload:
        _validate_constraints(workload["constraints"])
    return workload


def _validate_winning_workload_schema(workload: Mapping) -> None:
    objective = workload.get("objective")
    if not isinstance(objective, Mapping):
        raise ValueError("incomplete workload evidence: objective is required")
    if "primary_metric" not in objective or "constraints" not in objective:
        raise ValueError(
            "incomplete workload evidence: objective primary_metric and constraints "
            "are required"
        )
    primary_metric = objective["primary_metric"]
    if not isinstance(primary_metric, Mapping):
        raise ValueError("incomplete workload evidence: primary_metric must be a mapping")
    primary_name = primary_metric.get("name")
    if not isinstance(primary_name, str) or not primary_name.strip():
        raise ValueError("incomplete workload evidence: primary_metric.name is required")
    if primary_metric.get("direction") not in {"lower", "higher"}:
        raise ValueError(
            "incomplete workload evidence: primary_metric.direction is invalid"
        )
    declared = objective["constraints"]
    if not isinstance(declared, list):
        raise ValueError(
            "incomplete workload evidence: objective.constraints must be a sequence"
        )
    declared_names = []
    declared_seen = set()
    for index, constraint in enumerate(declared):
        constraint = _mapping(
            constraint, f"workload.objective.constraints[{index}]"
        )
        name = constraint.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"workload.objective.constraints[{index}].name must be non-empty"
            )
        name = name.strip()
        if name in declared_seen:
            raise ValueError(f"workload objective contains duplicate constraint: {name}")
        declared_seen.add(name)
        declared_names.append(name)
    if "constraints" not in workload or workload["constraints"] is None:
        raise ValueError("incomplete workload evidence: constraints are required")
    results = _validate_constraints(workload["constraints"])
    result_names = [constraint["name"] for constraint in results]
    if set(result_names) != set(declared_names):
        raise ValueError(
            "incomplete workload evidence: objective and result constraints must match"
        )


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
    return _validate_win_statistics(kernel["statistics"], "kernel.statistics")


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
    workload_statistics = _validate_win_statistics(primary, "workload.primary")
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
