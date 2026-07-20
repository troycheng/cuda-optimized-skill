#!/usr/bin/env python3
"""Admit Planner candidates only from current Controller-verified evidence."""

from __future__ import annotations

import importlib.util
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ADMISSION_SCHEMA = "cuda-optimizer/planner-admission-v1"
_DIAGNOSTIC_KINDS = {"nsys_timeline", "pytorch_profile"}
_KNOWLEDGE_ALLOWED = {"experimental", "verified"}
_POLICY_FIELDS = {
    "arch",
    "task",
    "layer",
    "framework_versions",
    "as_of",
    "max_review_age_days",
    "context_budget_bytes",
    "limit",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ValidationError(ValueError):
    pass


def _sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"cuda_planner_{name}", path)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise ValidationError(f"cannot load {name}.py")
    spec.loader.exec_module(module)
    return module


_SUMMARY = _sibling("evidence_summary")
_CAPABILITY_QUERY = _sibling("capability_query")
_RUN_CONTROL = _sibling("run_control")
_ADMISSION = _sibling("planner_admission")


def _closed(value: Any, fields: set[str], label: str) -> dict:
    if type(value) is not dict:
        raise ValidationError(f"{label} must be an object")
    missing = fields - set(value)
    extra = set(value) - fields
    if missing or extra:
        raise ValidationError(
            f"{label} must be closed; missing={sorted(missing)} extra={sorted(extra)}"
        )
    return value


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be lowercase SHA-256")
    return value


def _replay_query(policy: Mapping[str, Any], *, signals: set[str], evidence: set[str]) -> dict:
    clean = _closed(dict(policy), _POLICY_FIELDS, "capability_policy")
    try:
        return _CAPABILITY_QUERY.query(
            arch=clean["arch"],
            task=clean["task"],
            layer=clean["layer"],
            signals=sorted(signals),
            available_evidence=sorted(evidence),
            framework_versions=clean["framework_versions"],
            context_budget_bytes=clean["context_budget_bytes"],
            limit=clean["limit"],
            as_of=clean["as_of"],
            max_review_age_days=clean["max_review_age_days"],
        )
    except ValueError as exc:
        raise ValidationError(f"capability query policy is invalid: {exc}") from exc


def validate_candidate_admission(
    proposal: Mapping[str, Any],
    *,
    capability_query: Mapping[str, Any],
    observation_summary: Mapping[str, Any],
    capability_policy: Mapping[str, Any],
    ledger_path: str | os.PathLike,
    artifact_root: str | os.PathLike,
    controller_seal_key: bytes,
    expected_run_id: str,
    expected_ledger_id: str,
    expected_contract_sha256: str,
    expected_environment_sha256: str,
    expected_reference_sha256: str,
    expected_target_sha256: str,
    expected_workload_sha256: str,
    current_as_of: float,
    max_age_seconds: float,
) -> dict:
    """Return a hash-bound admission record or fail closed.

    The caller supplies Controller-owned identities and capability policy. The
    Planner may propose a candidate, but cannot choose its observed signals,
    available evidence, registry result, or pre-execution gate outcome.
    """
    try:
        candidate = _RUN_CONTROL.validate_candidate_proposal(dict(proposal))
        persisted = _SUMMARY.verify_summary(
            observation_summary,
            ledger_path=ledger_path,
            artifact_root=artifact_root,
            controller_seal_key=controller_seal_key,
        )
    except ValueError as exc:
        raise ValidationError(f"candidate or observation summary is invalid: {exc}") from exc
    identities = {
        "run_id": expected_run_id,
        "ledger_id": expected_ledger_id,
        "contract_sha256": _sha(expected_contract_sha256, "expected_contract_sha256"),
        "environment_sha256": _sha(
            expected_environment_sha256, "expected_environment_sha256"
        ),
    }
    for field, expected in identities.items():
        if persisted[field] != expected:
            raise ValidationError(f"observation summary {field} does not match Controller")
    if persisted["as_of"] != current_as_of:
        raise ValidationError("observation summary is not the current Controller snapshot")
    if persisted["max_age_seconds"] != max_age_seconds:
        raise ValidationError("observation summary freshness policy changed")
    if candidate["observation_summary_sha256"] != persisted["summary_sha256"]:
        raise ValidationError("candidate observation summary hash does not match")

    target_sha = _sha(expected_target_sha256, "expected_target_sha256")
    relevant = [
        item
        for item in persisted["observations"]
        if item["kind"] in _DIAGNOSTIC_KINDS
        and item["freshness"] == "current"
        and item.get("subject", {}).get("target_sha256") == target_sha
    ]
    if not relevant:
        raise ValidationError("no current diagnostic evidence matches the target")
    evidence = {item["kind"] for item in relevant}
    signals = {signal for item in relevant for signal in item["signals"]}
    replayed_query = _replay_query(
        capability_policy, signals=signals, evidence=evidence
    )
    try:
        supplied_query = _CAPABILITY_QUERY.validate_query_result(dict(capability_query))
    except ValueError as exc:
        raise ValidationError(f"capability query replay failed: {exc}") from exc
    if supplied_query != replayed_query:
        raise ValidationError(
            "capability query signals, evidence, or Controller policy do not match"
        )
    if candidate["capability_query_sha256"] != replayed_query["query_sha256"]:
        raise ValidationError("candidate capability query hash does not match")
    if not candidate["capability_ids"]:
        raise ValidationError("candidate admission requires at least one capability")
    selected = {item["id"]: item for item in replayed_query["capabilities"]}
    if not set(candidate["capability_ids"]).issubset(selected):
        raise ValidationError("candidate cites an unselected capability")
    if candidate["mechanism_id"] not in candidate["capability_ids"]:
        raise ValidationError(
            "candidate mechanism_id must be a selected capability identity"
        )
    chosen = [selected[item] for item in candidate["capability_ids"]]
    for item in chosen:
        if item["retrieval_status"] != "ready":
            raise ValidationError(f"capability {item['id']} lacks required evidence")
        if item["knowledge_status"] not in _KNOWLEDGE_ALLOWED:
            raise ValidationError(f"capability {item['id']} knowledge is not admissible")
        if item["contract_binding_required"] is not True:
            raise ValidationError(f"capability {item['id']} is not contract bound")
        conflicts = set(item["conflicts"]) & set(candidate["capability_ids"])
        if conflicts:
            raise ValidationError(
                f"capability {item['id']} conflicts with {sorted(conflicts)}"
            )

    focal = next(
        (
            item
            for item in relevant
            if item["observation_id"] == candidate["observation_id"]
        ),
        None,
    )
    if focal is None:
        raise ValidationError("candidate focal observation is not current diagnostic evidence")
    matched_signals = {
        signal for item in chosen for signal in item["matched_signals"]
    }
    if not set(focal["signals"]) & matched_signals:
        raise ValidationError("candidate focal observation does not support its capability")

    gate_requirements = chosen[0]["gate_requirements"]
    if any(item["gate_requirements"] != gate_requirements for item in chosen[1:]):
        raise ValidationError("candidate capabilities require incompatible phase gates")
    try:
        gates = _SUMMARY.resolve_gate_requirements(
            persisted,
            gate_requirements,
            ledger_path=ledger_path,
            artifact_root=artifact_root,
            expected_run_id=expected_run_id,
            expected_ledger_id=expected_ledger_id,
            expected_contract_sha256=expected_contract_sha256,
            expected_environment_sha256=expected_environment_sha256,
            current_as_of=current_as_of,
            max_age_seconds=max_age_seconds,
            expected_ledger_tail_sha256=persisted["ledger_tail_sha256"],
            expected_reference_sha256=expected_reference_sha256,
            expected_target_sha256=expected_target_sha256,
            expected_workload_sha256=expected_workload_sha256,
            expected_arch=capability_policy["arch"],
            controller_seal_key=controller_seal_key,
        )
    except ValueError as exc:
        raise ValidationError(f"pre-execution gate resolution failed: {exc}") from exc
    if not gates["pre_execution"]["satisfied"]:
        raise ValidationError(
            f"pre-execution gates are missing: {gates['pre_execution']['missing_gates']}"
        )

    used_observation_ids = {candidate["observation_id"]}
    for gate in gates["pre_execution"]["gates"]:
        used_observation_ids.update(gate["observation_ids"])
    used_ages = [
        item["age_seconds"]
        for item in persisted["observations"]
        if item["observation_id"] in used_observation_ids
    ]
    if len(used_ages) != len(used_observation_ids):
        raise ValidationError("admission evidence set is incomplete")

    admission = {
        "schema_version": ADMISSION_SCHEMA,
        "status": "ADMITTED",
        "run_id": expected_run_id,
        "ledger_id": expected_ledger_id,
        "contract_sha256": expected_contract_sha256,
        "environment_sha256": expected_environment_sha256,
        "candidate_id": candidate["candidate_id"],
        "mechanism_id": candidate["mechanism_id"],
        "observation_id": candidate["observation_id"],
        "observation_summary_sha256": persisted["summary_sha256"],
        "capability_query_sha256": replayed_query["query_sha256"],
        "capability_ids": list(candidate["capability_ids"]),
        "admitted_at": float(current_as_of),
        "evidence_age_seconds": max(used_ages),
        "pre_execution": gates["pre_execution"],
    }
    return _ADMISSION.seal_admission(
        admission, controller_seal_key=controller_seal_key
    )


def register_candidate(
    state: Mapping[str, Any],
    proposal: Mapping[str, Any],
    *,
    now: float,
    **admission_inputs: Any,
) -> dict:
    """Validate the full Planner boundary, then register through run control."""
    if type(now) not in {int, float} or float(now) != float(
        admission_inputs.get("current_as_of", -1)
    ):
        raise ValidationError(
            "registration time must equal the current Controller snapshot time"
        )
    admission = validate_candidate_admission(proposal, **admission_inputs)
    try:
        updated = _RUN_CONTROL.register_candidate(
            state,
            proposal,
            admission=admission,
            controller_seal_key=admission_inputs["controller_seal_key"],
            now=now,
        )
    except ValueError as exc:
        raise ValidationError(f"run control rejected admitted candidate: {exc}") from exc
    return {"state": updated, "admission": admission}


def register_run_candidate(
    contract_path: str | os.PathLike,
    run_dir: str | os.PathLike,
    proposal: Mapping[str, Any],
    *,
    now: float,
    expected_tail_sha256: str | None = None,
    **admission_inputs: Any,
) -> dict:
    """Persist one admitted candidate in the hash-chained run ledger."""
    if type(now) not in {int, float} or float(now) != float(
        admission_inputs.get("current_as_of", -1)
    ):
        raise ValidationError(
            "registration time must equal the current Controller snapshot time"
        )
    admission = validate_candidate_admission(proposal, **admission_inputs)
    try:
        persisted = _RUN_CONTROL.register_run_candidate(
            contract_path,
            run_dir,
            proposal,
            admission=admission,
            controller_seal_key=admission_inputs["controller_seal_key"],
            now=now,
            expected_tail_sha256=expected_tail_sha256,
        )
    except ValueError as exc:
        raise ValidationError(f"run control rejected admitted candidate: {exc}") from exc
    return {**persisted, "admission": admission}
