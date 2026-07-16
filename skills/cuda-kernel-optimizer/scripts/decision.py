#!/usr/bin/env python3
"""Terminal decision engine for kernel-only and full optimization modes."""

from __future__ import annotations

import copy
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


def _literal_status(value, allowed: set[str], field: str) -> str:
    if type(value) is not str or value not in allowed:
        raise ValueError(f"{field} must be a recognized literal status")
    return value


def _mapping(value, field: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _normalize_mode(mode) -> str:
    if mode == "kernel_only":
        mode = "kernel-only"
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
        _literal_status(
            primary.get("status"), _PRIMARY_STATUSES, "workload.primary.status"
        )
    if status == "evaluated" and "constraints" in workload:
        _validate_constraints(workload["constraints"])
    return workload


def _validate_constraints(constraints) -> list[dict]:
    if constraints is None:
        return []
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
        item = copy.deepcopy(dict(constraint))
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
    statistics = kernel.get("statistics")
    if statistics is not None:
        statistics = _mapping(statistics, "kernel.statistics")
        _literal_status(
            statistics.get("status"), {"confirmed_win"}, "kernel.statistics.status"
        )
        return copy.deepcopy(dict(statistics))
    if kernel["status"] == "confirmed_win":
        return copy.deepcopy(dict(kernel))
    return {"status": "confirmed_win", "source_status": "kernel_only_win"}


def _result(status: str, reason: str, mode: str, evidence: dict, **fields) -> dict:
    if status not in TERMINAL_STATUSES:
        raise AssertionError("decision engine produced a non-terminal status")
    return {
        "status": status,
        "reason": reason,
        "mode": mode,
        "evidence": evidence,
        **fields,
    }


def decide(*, mode, kernel, workload=None, constraints=None, pareto=None) -> dict:
    """Return one terminal decision without mutating or weighting evidence."""
    normalized_mode = _normalize_mode(mode)
    kernel = _mapping(kernel, "kernel")
    kernel_status = _literal_status(
        kernel.get("status"), _KERNEL_STATUSES, "kernel.status"
    )
    workload = _validate_workload(workload)
    constraint_source = constraints
    if constraint_source is None and workload is not None:
        constraint_source = workload.get("constraints", [])
    normalized_constraints = _validate_constraints(constraint_source)
    normalized_pareto = _validate_pareto(pareto)
    evidence = {
        "kernel": copy.deepcopy(dict(kernel)),
        "workload": None if workload is None else copy.deepcopy(dict(workload)),
        "constraints": copy.deepcopy(normalized_constraints),
        "pareto": copy.deepcopy(normalized_pareto),
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
            constraints=copy.deepcopy(normalized_constraints),
        )
    if normalized_pareto is not None:
        return _result(
            "pareto_frontier",
            "explicit non-dominated multi-objective evidence shows a tradeoff",
            normalized_mode,
            evidence,
            statistics=kernel_statistics,
            pareto=copy.deepcopy(normalized_pareto),
        )
    if any(item["status"] != "passed" for item in normalized_constraints):
        return _result(
            "kernel_only_win",
            "hard-constraint evidence is inconclusive",
            normalized_mode,
            evidence,
            statistics=kernel_statistics,
            constraints=copy.deepcopy(normalized_constraints),
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
    workload_statistics = copy.deepcopy(dict(primary))
    return _result(
        "end_to_end_win",
        "kernel and real-workload evidence confirm the candidate",
        normalized_mode,
        evidence,
        statistics=kernel_statistics,
        workload_statistics=workload_statistics,
        constraints=copy.deepcopy(normalized_constraints),
    )


__all__ = ["TERMINAL_STATUSES", "decide"]
