#!/usr/bin/env python3
"""Deterministic state and candidate gates for long-running optimization."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any


STATE_SCHEMA = "cuda-optimizer/run-control-v1"
PROPOSAL_SCHEMA = "cuda-optimizer/candidate-proposal-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PHASES = {
    "INIT",
    "FROZEN",
    "CALIBRATING",
    "EXPLORING",
    "AUDITING",
    "DRIFTED",
    "CONVERGING",
    "STOPPED",
}
_OUTCOMES = {"PASS", "KILL", "INCONCLUSIVE", "DEFERRED"}
_PROPOSAL_FIELDS = {
    "schema_version",
    "candidate_id",
    "mechanism_id",
    "observation_id",
    "observation_summary_sha256",
    "capability_query_sha256",
    "hypothesis",
    "expected_metric",
    "expected_effect_pct",
    "kill_gate",
    "estimated_cost_seconds",
    "capability_ids",
    "paths",
}
_LEGACY_PROPOSAL_FIELDS = _PROPOSAL_FIELDS - {"mechanism_id"}
_TRANSITIONS = {
    ("INIT", "freeze"): "FROZEN",
    ("FROZEN", "calibrate"): "CALIBRATING",
    ("CALIBRATING", "start_exploration"): "EXPLORING",
    ("EXPLORING", "audit"): "AUDITING",
    ("AUDITING", "audit_pass"): "EXPLORING",
    ("EXPLORING", "converge"): "CONVERGING",
    ("CONVERGING", "stop"): "STOPPED",
    ("CALIBRATING", "stop"): "STOPPED",
    ("EXPLORING", "stop"): "STOPPED",
    ("AUDITING", "stop"): "STOPPED",
    ("DRIFTED", "stop"): "STOPPED",
    ("FROZEN", "stop"): "STOPPED",
    ("CALIBRATING", "environment_yellow"): "CALIBRATING",
    ("EXPLORING", "environment_yellow"): "AUDITING",
    ("AUDITING", "environment_yellow"): "AUDITING",
}
for _phase in _PHASES - {"INIT", "STOPPED", "DRIFTED"}:
    _TRANSITIONS[(_phase, "drift")] = "DRIFTED"
    _TRANSITIONS[(_phase, "environment_red")] = "STOPPED"


class ValidationError(ValueError):
    pass


def _closed(value: Mapping, fields: set[str], field: str) -> None:
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ValidationError(f"{field} contains unknown fields: {', '.join(unknown)}")
    missing = sorted(fields - set(value))
    if missing:
        raise ValidationError(f"{field} is missing required fields: {', '.join(missing)}")


def _identifier(value: Any, field: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValidationError(f"{field} must be a safe identifier")
    return value


def _string(value: Any, field: str, *, max_length: int = 4096) -> str:
    if type(value) is not str or not value.strip():
        raise ValidationError(f"{field} must be a non-empty string")
    if len(value) > max_length:
        raise ValidationError(f"{field} exceeds {max_length} characters")
    return value


def _sha(value: Any, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{field} must be lowercase SHA-256")
    return value


def _finite(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{field} must be a finite number")
    if number < minimum:
        raise ValidationError(f"{field} must be at least {minimum}")
    return number


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{field} must be a positive integer")
    return value


def _json_copy(value: Any, field: str = "value") -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError(f"{field} must contain finite JSON values") from error


def _string_array(value: Any, field: str, *, allow_empty: bool = False) -> list[str]:
    if type(value) is not list or (not value and not allow_empty):
        raise ValidationError(f"{field} must be {'an' if allow_empty else 'a non-empty'} array")
    values = [_identifier(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if len(values) != len(set(values)):
        raise ValidationError(f"{field} must not contain duplicates")
    return values


def _paths(value: Any) -> list[str]:
    if type(value) is not list or not value:
        raise ValidationError("paths must be a non-empty array")
    paths = []
    for index, item in enumerate(value):
        text = _string(item, f"paths[{index}]")
        path = Path(text)
        if path.is_absolute() or text in {".", ".."} or ".." in path.parts:
            raise ValidationError(f"paths[{index}] must be a contained relative path")
        normalized = os.path.normpath(text)
        if normalized in {"", ".", ".."}:
            raise ValidationError(f"paths[{index}] must name a project path")
        paths.append(normalized)
    if len(paths) != len(set(paths)):
        raise ValidationError("paths must not contain duplicates")
    return paths


def _candidate_path_has_no_symlink(project_root: str, relative: str) -> None:
    current = Path(project_root)
    for component in Path(relative).parts:
        current = current / component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            return
        except OSError as error:
            raise ValidationError(f"candidate mutation path is unsafe: {relative}") from error
        if stat.S_ISLNK(mode):
            raise ValidationError(
                f"candidate mutation path contains a symlink: {relative}"
            )


def validate_candidate_proposal(
    value: Mapping[str, Any], *, allow_legacy: bool = False
) -> dict:
    if type(value) is not dict:
        raise ValidationError("candidate proposal must be an object")
    fields = set(value)
    legacy = fields == _LEGACY_PROPOSAL_FIELDS
    if not (legacy and allow_legacy):
        _closed(value, _PROPOSAL_FIELDS, "candidate proposal")
    if value["schema_version"] != PROPOSAL_SCHEMA:
        raise ValidationError(f"schema_version must be {PROPOSAL_SCHEMA}")
    _identifier(value["candidate_id"], "candidate_id")
    if not legacy:
        _identifier(value["mechanism_id"], "mechanism_id")
    _identifier(value["observation_id"], "observation_id")
    _sha(value["observation_summary_sha256"], "observation_summary_sha256")
    _sha(value["capability_query_sha256"], "capability_query_sha256")
    _string(value["hypothesis"], "hypothesis")
    metric = value["expected_metric"]
    if type(metric) is not dict:
        raise ValidationError("expected_metric must be an object")
    _closed(metric, {"name", "direction"}, "expected_metric")
    _identifier(metric["name"], "expected_metric.name")
    if metric["direction"] not in {"lower", "higher"}:
        raise ValidationError("expected_metric.direction must be lower or higher")
    _finite(value["expected_effect_pct"], "expected_effect_pct")
    _string(value["kill_gate"], "kill_gate")
    if _finite(value["estimated_cost_seconds"], "estimated_cost_seconds") <= 0:
        raise ValidationError("estimated_cost_seconds must be positive")
    _string_array(value["capability_ids"], "capability_ids", allow_empty=True)
    _paths(value["paths"])
    return _json_copy(value, "candidate proposal")


def mechanism_fingerprint(proposal: Mapping[str, Any]) -> str:
    """Identify a mechanism independently of candidate name and cost tuning."""
    if not isinstance(proposal, Mapping):
        raise ValidationError("mechanism proposal is incomplete")
    fields = _PROPOSAL_FIELDS if "mechanism_id" in proposal else _LEGACY_PROPOSAL_FIELDS
    if not fields.issubset(proposal):
        raise ValidationError("mechanism proposal is incomplete")
    candidate = validate_candidate_proposal(
        {field: proposal[field] for field in fields}, allow_legacy=True
    )
    mechanism_id = candidate.get("mechanism_id")
    legacy_capabilities = sorted(candidate["capability_ids"])
    if mechanism_id is None and len(legacy_capabilities) == 1:
        mechanism_id = legacy_capabilities[0]
    if mechanism_id is not None:
        basis = {"mechanism_id": mechanism_id}
    else:
        basis = {
            "capability_ids": legacy_capabilities,
            "paths": sorted(candidate["paths"]),
            "expected_metric": candidate["expected_metric"],
        }
    encoded = json.dumps(
        basis,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _legacy_new_mechanism_overlap(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    """Match legacy capabilities to a new ID without conflating two legacies."""
    left_id = left.get("mechanism_id")
    right_id = right.get("mechanism_id")
    if bool(left_id) == bool(right_id):
        return False
    modern = left if left_id else right
    legacy = right if left_id else left
    modern_id = modern.get("mechanism_id")
    capabilities = legacy.get("capability_ids")
    return (
        isinstance(modern_id, str)
        and isinstance(capabilities, list)
        and modern_id in capabilities
    )


def _validate_contract_subset(
    contract: Mapping[str, Any]
) -> tuple[str, dict, dict, dict, list[str], str]:
    if not isinstance(contract, Mapping):
        raise ValidationError("contract must be an object")
    if contract.get("schema_version") != "cuda-optimizer/workload-contract-v1":
        raise ValidationError("contract must be a frozen workload contract")
    contract_sha = _sha(contract.get("contract_sha256"), "contract_sha256")
    budget = contract.get("budget")
    evidence = contract.get("evidence")
    objective = contract.get("objective")
    mutation = contract.get("mutation")
    if not all(
        isinstance(item, Mapping)
        for item in (budget, evidence, objective, mutation)
    ):
        raise ValidationError(
            "contract budget, evidence, objective, and mutation are required"
        )
    max_seconds = _finite(budget.get("max_seconds"), "budget.max_seconds", minimum=1.0)
    max_candidates = _positive_int(budget.get("max_candidates"), "budget.max_candidates")
    max_age = _finite(evidence.get("max_age_seconds"), "evidence.max_age_seconds", minimum=1.0)
    metric = _identifier(objective.get("metric"), "objective.metric")
    direction = objective.get("direction")
    if direction not in {"lower", "higher"}:
        raise ValidationError("objective.direction must be lower or higher")
    mutation_paths = _paths(mutation.get("project_paths"))
    project_root = contract.get("project_root")
    if type(project_root) is not str or not Path(project_root).is_absolute():
        raise ValidationError("contract project_root must be an absolute path")
    return (
        contract_sha,
        {"max_seconds": max_seconds, "max_candidates": max_candidates},
        {"max_age_seconds": max_age},
        {"metric": metric, "direction": direction},
        mutation_paths,
        project_root,
    )


def initialize_state(contract: Mapping[str, Any], *, now: float) -> dict:
    (
        contract_sha,
        budget,
        evidence,
        objective,
        mutation_paths,
        project_root,
    ) = _validate_contract_subset(contract)
    timestamp = _finite(now, "now")
    return {
        "schema_version": STATE_SCHEMA,
        "phase": "INIT",
        "contract_sha256": contract_sha,
        "started_at": timestamp,
        "updated_at": timestamp,
        "max_seconds": budget["max_seconds"],
        "max_candidates": budget["max_candidates"],
        "max_evidence_age_seconds": evidence["max_age_seconds"],
        "objective_metric": objective["metric"],
        "objective_direction": objective["direction"],
        "mutation_paths": mutation_paths,
        "project_root": project_root,
        "candidates_started": 0,
        "active_candidate": None,
        "candidate_history": [],
        "champion_candidate_id": None,
        "environment_state": None,
        "measurable": None,
        "stop_reason": None,
        "drift_reason": None,
        "audit_reason": None,
    }


def _validate_state(value: Mapping[str, Any]) -> dict:
    if type(value) is not dict:
        raise ValidationError("state must be an object")
    fields = {
        "schema_version",
        "phase",
        "contract_sha256",
        "started_at",
        "updated_at",
        "max_seconds",
        "max_candidates",
        "max_evidence_age_seconds",
        "objective_metric",
        "objective_direction",
        "mutation_paths",
        "project_root",
        "candidates_started",
        "active_candidate",
        "candidate_history",
        "champion_candidate_id",
        "environment_state",
        "measurable",
        "stop_reason",
        "drift_reason",
        "audit_reason",
    }
    _closed(value, fields, "state")
    if value["schema_version"] != STATE_SCHEMA:
        raise ValidationError(f"state schema_version must be {STATE_SCHEMA}")
    if value["phase"] not in _PHASES:
        raise ValidationError("state phase is unsupported")
    _sha(value["contract_sha256"], "contract_sha256")
    started_at = _finite(value["started_at"], "started_at")
    updated_at = _finite(value["updated_at"], "updated_at")
    if updated_at < started_at:
        raise ValidationError("updated_at must not precede started_at")
    _finite(value["max_seconds"], "max_seconds", minimum=1.0)
    _positive_int(value["max_candidates"], "max_candidates")
    _finite(value["max_evidence_age_seconds"], "max_evidence_age_seconds", minimum=1.0)
    _identifier(value["objective_metric"], "objective_metric")
    if value["objective_direction"] not in {"lower", "higher"}:
        raise ValidationError("objective_direction must be lower or higher")
    _paths(value["mutation_paths"])
    if type(value["project_root"]) is not str or not Path(value["project_root"]).is_absolute():
        raise ValidationError("project_root must be an absolute path")
    if isinstance(value["candidates_started"], bool) or not isinstance(value["candidates_started"], int) or value["candidates_started"] < 0:
        raise ValidationError("candidates_started must be a nonnegative integer")
    if type(value["candidate_history"]) is not list:
        raise ValidationError("candidate_history must be an array")
    active = value["active_candidate"]
    if active is not None:
        _validate_stored_candidate(active, history=False, field="active_candidate")
    history_ids = []
    pass_ids = []
    for index, item in enumerate(value["candidate_history"]):
        candidate = _validate_stored_candidate(
            item, history=True, field=f"candidate_history[{index}]"
        )
        history_ids.append(candidate["candidate_id"])
        if candidate["outcome"] == "PASS":
            pass_ids.append(candidate["candidate_id"])
    active_id = active["candidate_id"] if active is not None else None
    all_ids = history_ids + ([active_id] if active_id is not None else [])
    if len(all_ids) != len(set(all_ids)):
        raise ValidationError("candidate ids must be unique across active_candidate and history")
    expected_started = len(value["candidate_history"]) + (1 if active is not None else 0)
    if value["candidates_started"] != expected_started:
        raise ValidationError("candidates_started must equal candidate history plus the active candidate")
    champion = value["champion_candidate_id"]
    if champion is not None:
        _identifier(champion, "champion_candidate_id")
    expected_champion = pass_ids[-1] if pass_ids else None
    if champion != expected_champion:
        raise ValidationError("champion_candidate_id must name the latest PASS candidate")
    if value["environment_state"] not in {None, "green", "yellow", "red"}:
        raise ValidationError("environment_state must be green, yellow, red, or null")
    if value["measurable"] is not None and type(value["measurable"]) is not bool:
        raise ValidationError("measurable must be a boolean or null")
    for field in ("stop_reason", "drift_reason", "audit_reason"):
        if value[field] is not None:
            _identifier(value[field], field)
    if value["phase"] == "EXPLORING" and (
        value["environment_state"] != "green" or value["measurable"] is not True
    ):
        raise ValidationError("EXPLORING requires a green, measurable environment")
    if value["phase"] == "STOPPED" and value["stop_reason"] is None:
        raise ValidationError("STOPPED requires stop_reason")
    if value["phase"] == "DRIFTED" and value["drift_reason"] is None:
        raise ValidationError("DRIFTED requires drift_reason")
    return _json_copy(value, "state")


def _validate_stored_candidate(
    value: Any, *, history: bool, field: str
) -> dict:
    if type(value) is not dict:
        raise ValidationError(f"{field} must be an object")
    proposal_fields = (
        _PROPOSAL_FIELDS
        if "mechanism_id" in value
        else _LEGACY_PROPOSAL_FIELDS
    )
    fields = set(proposal_fields) | {"registered_at"}
    if history:
        fields |= {
            "outcome",
            "resolved_at",
            "correctness_ok",
            "performance_gate_passed",
        }
    _closed(value, fields, field)
    proposal = {key: value[key] for key in proposal_fields}
    validate_candidate_proposal(proposal, allow_legacy=True)
    registered_at = _finite(value["registered_at"], f"{field}.registered_at")
    if history:
        if value["outcome"] not in _OUTCOMES:
            raise ValidationError(f"{field}.outcome is unsupported")
        resolved_at = _finite(value["resolved_at"], f"{field}.resolved_at")
        if resolved_at < registered_at:
            raise ValidationError(f"{field}.resolved_at precedes registration")
        for gate in ("correctness_ok", "performance_gate_passed"):
            if value[gate] is not None and type(value[gate]) is not bool:
                raise ValidationError(f"{field}.{gate} must be a boolean or null")
        if value["outcome"] == "PASS" and (
            value["correctness_ok"] is not True
            or value["performance_gate_passed"] is not True
        ):
            raise ValidationError(f"{field} PASS bypasses a promotion gate")
    return _json_copy(value, field)


def _stop_for_budget(current: dict, *, timestamp: float, reason: str) -> dict:
    if current["active_candidate"] is not None:
        deferred = current["active_candidate"] | {
            "outcome": "DEFERRED",
            "resolved_at": timestamp,
            "correctness_ok": None,
            "performance_gate_passed": None,
        }
        current["candidate_history"].append(deferred)
        current["active_candidate"] = None
    current["phase"] = "STOPPED"
    current["stop_reason"] = reason
    current["updated_at"] = timestamp
    return current


def advance(
    state: Mapping[str, Any],
    action: str,
    *,
    now: float,
    environment_state: str | None = None,
    measurable: bool | None = None,
    reason: str | None = None,
    new_contract_sha256: str | None = None,
) -> dict:
    current = _validate_state(state)
    timestamp = _finite(now, "now")
    if timestamp < current["updated_at"]:
        raise ValidationError("now must not move backwards")
    key = (current["phase"], action)
    if key not in _TRANSITIONS:
        raise ValidationError(f"illegal state transition: {current['phase']} + {action}")
    if current["active_candidate"] is not None and action not in {"drift", "environment_red", "stop"}:
        raise ValidationError("active candidate must be resolved before state transition")
    if (
        timestamp - current["started_at"] >= current["max_seconds"]
        and action not in {"stop", "drift", "environment_red"}
    ):
        return _stop_for_budget(
            current, timestamp=timestamp, reason="time_budget_exhausted"
        )

    target = _TRANSITIONS[key]
    if action == "start_exploration":
        if environment_state != "green":
            raise ValidationError("start_exploration requires a green environment")
        if measurable is not True:
            raise ValidationError("start_exploration requires a measurable effect")
        current["environment_state"] = "green"
        current["measurable"] = True
    elif action == "audit_pass":
        current["environment_state"] = "green"
        current["audit_reason"] = None
    elif action == "environment_yellow":
        current["environment_state"] = "yellow"
        current["audit_reason"] = _identifier(reason, "reason")
    elif action == "environment_red":
        current["environment_state"] = "red"
        current["stop_reason"] = _identifier(reason, "reason")
    elif action == "drift":
        current["drift_reason"] = _identifier(reason, "reason")
    elif action == "audit":
        current["audit_reason"] = _identifier(reason or "scheduled_replay", "reason")
    elif action == "stop":
        current["stop_reason"] = _identifier(reason, "reason")

    if current["active_candidate"] is not None and action in {"drift", "environment_red", "stop"}:
        deferred = current["active_candidate"] | {
            "outcome": "DEFERRED",
            "resolved_at": timestamp,
            "correctness_ok": None,
            "performance_gate_passed": None,
        }
        current["candidate_history"].append(deferred)
        current["active_candidate"] = None
    current["phase"] = target
    current["updated_at"] = timestamp
    return current


def _register_candidate_state(
    state: Mapping[str, Any],
    proposal: Mapping[str, Any],
    *,
    contract_sha256: str,
    evidence_age_seconds: float,
    now: float,
    allow_legacy: bool = False,
) -> dict:
    current = _validate_state(state)
    if current["phase"] != "EXPLORING":
        raise ValidationError("candidate registration requires EXPLORING phase")
    if current["environment_state"] != "green":
        raise ValidationError("candidate registration requires a green environment")
    if current["active_candidate"] is not None:
        raise ValidationError("one active candidate is already registered")
    if _sha(contract_sha256, "contract_sha256") != current["contract_sha256"]:
        raise ValidationError("candidate contract identity does not match the run")
    age = _finite(evidence_age_seconds, "evidence_age_seconds")
    if age > current["max_evidence_age_seconds"]:
        raise ValidationError("candidate evidence is stale")
    timestamp = _finite(now, "now")
    if timestamp < current["updated_at"]:
        raise ValidationError("now must not move backwards")
    candidate = validate_candidate_proposal(
        proposal, allow_legacy=allow_legacy
    )
    if candidate["expected_metric"] != {
        "name": current["objective_metric"],
        "direction": current["objective_direction"],
    }:
        raise ValidationError("candidate metric does not match the frozen objective")
    for path in candidate["paths"]:
        if not any(
            path == root or path.startswith(root.rstrip("/") + "/")
            for root in current["mutation_paths"]
        ):
            raise ValidationError(
                f"candidate paths are outside the allowed mutation roots: {path}"
            )
        _candidate_path_has_no_symlink(current["project_root"], path)
    elapsed = timestamp - current["started_at"]
    if current["candidates_started"] >= current["max_candidates"]:
        return _stop_for_budget(
            current, timestamp=timestamp, reason="candidate_budget_exhausted"
        )
    if elapsed >= current["max_seconds"] or elapsed + candidate["estimated_cost_seconds"] > current["max_seconds"]:
        return _stop_for_budget(
            current, timestamp=timestamp, reason="time_budget_exhausted"
        )
    existing_ids = {item["candidate_id"] for item in current["candidate_history"]}
    if candidate["candidate_id"] in existing_ids:
        raise ValidationError("candidate_id was already used")
    fingerprint = mechanism_fingerprint(candidate)
    if any(
        mechanism_fingerprint(item) == fingerprint
        or _legacy_new_mechanism_overlap(candidate, item)
        for item in current["candidate_history"]
    ):
        raise ValidationError("candidate mechanism was already attempted")
    current["active_candidate"] = candidate | {"registered_at": timestamp}
    current["candidates_started"] += 1
    current["updated_at"] = timestamp
    return current


def register_candidate(
    state: Mapping[str, Any],
    proposal: Mapping[str, Any],
    *,
    admission: Mapping[str, Any],
    controller_seal_key: bytes,
    now: float,
    _allow_legacy: bool = False,
) -> dict:
    """Register only a candidate carrying a current Controller admission."""
    clean_state = _validate_state(state)
    clean_proposal = validate_candidate_proposal(
        proposal, allow_legacy=_allow_legacy
    )
    clean_admission = _sibling("planner_admission").validate_admission(
        admission,
        controller_seal_key=controller_seal_key,
        proposal=clean_proposal,
        expected_contract_sha256=clean_state["contract_sha256"],
        expected_admitted_at=now,
    )
    return _register_candidate_state(
        clean_state,
        clean_proposal,
        contract_sha256=clean_admission["contract_sha256"],
        evidence_age_seconds=clean_admission["evidence_age_seconds"],
        now=now,
        allow_legacy=_allow_legacy,
    )


def resolve_candidate(
    state: Mapping[str, Any],
    *,
    candidate_id: str,
    outcome: str,
    correctness_ok: bool | None,
    performance_gate_passed: bool | None,
    now: float,
) -> dict:
    current = _validate_state(state)
    active = current["active_candidate"]
    if active is None:
        raise ValidationError("no active candidate to resolve")
    candidate = _identifier(candidate_id, "candidate_id")
    if candidate != active["candidate_id"]:
        raise ValidationError("candidate_id does not match the active candidate")
    if outcome not in _OUTCOMES:
        raise ValidationError("outcome must be PASS, KILL, INCONCLUSIVE, or DEFERRED")
    if correctness_ok is not None and type(correctness_ok) is not bool:
        raise ValidationError("correctness_ok must be a boolean or null")
    if performance_gate_passed is not None and type(performance_gate_passed) is not bool:
        raise ValidationError("performance_gate_passed must be a boolean or null")
    if outcome == "PASS":
        raise ValidationError(
            "PASS requires verified evidence artifacts; the promotion adapter is not connected"
        )
    timestamp = _finite(now, "now")
    if timestamp < current["updated_at"]:
        raise ValidationError("now must not move backwards")
    closed = active | {
        "outcome": outcome,
        "resolved_at": timestamp,
        "correctness_ok": correctness_ok,
        "performance_gate_passed": performance_gate_passed,
    }
    current["candidate_history"].append(closed)
    current["active_candidate"] = None
    if timestamp - current["started_at"] >= current["max_seconds"]:
        current["phase"] = "STOPPED"
        current["stop_reason"] = "time_budget_exhausted"
    current["updated_at"] = timestamp
    return current


_RUNTIME_MODULES = {}


def _sibling(name: str):
    if name not in _RUNTIME_MODULES:
        path = Path(__file__).with_name(f"{name}.py")
        spec = importlib.util.spec_from_file_location(f"cuda_run_control_{name}", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        _RUNTIME_MODULES[name] = module
    return _RUNTIME_MODULES[name]


def _ledger_path(run_dir: str | os.PathLike) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(run_dir)))) / "ledger"


def _payload_fields(payload: Any, fields: set[str], event_type: str) -> dict:
    if type(payload) is not dict:
        raise ValidationError(f"{event_type} payload must be an object")
    _closed(payload, fields, f"{event_type} payload")
    return payload


def _replay_records(
    contract: Mapping[str, Any],
    records: list[Mapping[str, Any]],
    *,
    contract_path: str | os.PathLike | None = None,
    controller_seal_key: bytes | None = None,
) -> dict:
    if not records:
        raise ValidationError("run ledger is empty")
    state = None
    latest_calibration_sha256 = None
    candidates_since_stability = 0
    audit_cadence = contract["stability"]["audit_every_candidates"]
    for index, record in enumerate(records):
        event_type = record["event_type"]
        payload = record["payload"]
        if index == 0:
            if event_type != "run_initialized":
                raise ValidationError("run replay must begin with run_initialized")
            payload = _payload_fields(payload, {"state"}, event_type)
            recorded_state = _validate_state(payload["state"])
            state = initialize_state(contract, now=recorded_state["started_at"])
        elif event_type == "state_transition":
            payload = _payload_fields(
                payload, {"action", "now", "arguments", "state"}, event_type
            )
            if type(payload["arguments"]) is not dict:
                raise ValidationError("state_transition arguments must be an object")
            allowed_arguments = {
                "environment_state",
                "measurable",
                "reason",
                "new_contract_sha256",
            }
            unknown_arguments = sorted(set(payload["arguments"]) - allowed_arguments)
            if unknown_arguments:
                raise ValidationError("state_transition arguments contain unknown fields")
            if payload["action"] in {
                "start_exploration",
                "audit_pass",
                "environment_yellow",
                "environment_red",
            }:
                raise ValidationError(
                    "stability-derived state transition bypasses its evidence artifact"
                )
            state = advance(
                state,
                _identifier(payload["action"], "action"),
                now=payload["now"],
                **payload["arguments"],
            )
            recorded_state = _validate_state(payload["state"])
        elif event_type == "candidate_registered":
            payload = _payload_fields(
                payload,
                {"proposal", "admission", "now", "state"},
                event_type,
            )
            if controller_seal_key is None:
                raise ValidationError(
                    "candidate replay requires the Controller seal key"
                )
            if latest_calibration_sha256 is None:
                raise ValidationError(
                    "candidate replay requires stability calibration"
                )
            if candidates_since_stability >= audit_cadence:
                raise ValidationError(
                    "candidate replay exceeds the stability audit cadence"
                )
            state = register_candidate(
                state,
                payload["proposal"],
                admission=payload["admission"],
                controller_seal_key=controller_seal_key,
                now=payload["now"],
                _allow_legacy=True,
            )
            candidates_since_stability += 1
            recorded_state = _validate_state(payload["state"])
        elif event_type == "stability_calibrated":
            payload = _payload_fields(
                payload, {"calibration", "now", "state"}, event_type
            )
            if controller_seal_key is None or contract_path is None:
                raise ValidationError(
                    "stability calibration replay requires the Controller seal key and contract"
                )
            calibration = _sibling("stability_calibration").validate_calibration(
                payload["calibration"],
                contract_path=contract_path,
                controller_seal_key=controller_seal_key,
            )
            state = _apply_calibration_state(state, calibration, now=payload["now"])
            latest_calibration_sha256 = calibration["calibration_sha256"]
            candidates_since_stability = 0
            recorded_state = _validate_state(payload["state"])
        elif event_type == "stability_audited":
            payload = _payload_fields(
                payload, {"audit", "now", "state"}, event_type
            )
            if controller_seal_key is None:
                raise ValidationError(
                    "stability audit replay requires the Controller seal key"
                )
            audit = _sibling("stability_calibration").validate_audit(
                payload["audit"], controller_seal_key=controller_seal_key
            )
            if latest_calibration_sha256 is None or (
                audit["anchor_calibration_sha256"] != latest_calibration_sha256
            ):
                raise ValidationError("stability audit anchor does not match this run")
            state = _apply_stability_audit_state(state, audit, now=payload["now"])
            candidates_since_stability = 0
            recorded_state = _validate_state(payload["state"])
        elif event_type == "candidate_resolved":
            payload = _payload_fields(
                payload,
                {
                    "candidate_id",
                    "outcome",
                    "correctness_ok",
                    "performance_gate_passed",
                    "now",
                    "state",
                },
                event_type,
            )
            state = resolve_candidate(
                state,
                candidate_id=payload["candidate_id"],
                outcome=payload["outcome"],
                correctness_ok=payload["correctness_ok"],
                performance_gate_passed=payload["performance_gate_passed"],
                now=payload["now"],
            )
            recorded_state = _validate_state(payload["state"])
        else:
            raise ValidationError(f"run replay does not recognize event_type {event_type}")
        if state != recorded_state:
            raise ValidationError(f"run replay state mismatch at sequence {index + 1}")
    return state


def load_run(
    contract_path: str | os.PathLike,
    run_dir: str | os.PathLike,
    *,
    controller_seal_key: bytes | None = None,
) -> dict:
    """Verify the frozen contract and reconstruct state from the complete ledger."""
    contract = _sibling("workload_contract").verify_frozen_contract(contract_path)
    records = _sibling("evidence_ledger").verify_ledger(
        _ledger_path(run_dir), expected_contract_sha256=contract["contract_sha256"]
    )
    state = _replay_records(
        contract,
        records,
        contract_path=contract_path,
        controller_seal_key=controller_seal_key,
    )
    return {
        "state": state,
        "tail_sha256": records[-1]["record_sha256"],
        "event_count": len(records),
    }


def initialize_run(
    contract_path: str | os.PathLike,
    run_dir: str | os.PathLike,
    *,
    now: float,
) -> dict:
    """Create the first control event; an existing run cannot be overwritten."""
    contract = _sibling("workload_contract").verify_frozen_contract(contract_path)
    state = initialize_state(contract, now=now)
    record = _sibling("evidence_ledger").append_event(
        _ledger_path(run_dir),
        event_type="run_initialized",
        contract_sha256=contract["contract_sha256"],
        expected_previous_sha256="0" * 64,
        payload={"state": state},
    )
    return {"state": state, "tail_sha256": record["record_sha256"], "event_count": 1}


def _expected_tail(loaded: Mapping[str, Any], expected: str | None) -> str:
    tail = loaded["tail_sha256"]
    if expected is not None:
        _sha(expected, "expected_tail_sha256")
        if expected != tail:
            raise ValidationError("stale run tail: a newer control event already exists")
    return tail


def transition_run(
    contract_path: str | os.PathLike,
    run_dir: str | os.PathLike,
    action: str,
    *,
    now: float,
    environment_state: str | None = None,
    measurable: bool | None = None,
    reason: str | None = None,
    expected_tail_sha256: str | None = None,
    controller_seal_key: bytes | None = None,
) -> dict:
    if action == "refreeze":
        raise ValidationError("refreeze requires a new contract and a new run ledger")
    if action in {
        "start_exploration",
        "audit_pass",
        "environment_yellow",
        "environment_red",
    }:
        raise ValidationError(
            "stability-derived transition requires a Controller calibration or audit artifact"
        )
    loaded = load_run(
        contract_path, run_dir, controller_seal_key=controller_seal_key
    )
    tail = _expected_tail(loaded, expected_tail_sha256)
    arguments = {
        key: value
        for key, value in {
            "environment_state": environment_state,
            "measurable": measurable,
            "reason": reason,
        }.items()
        if value is not None
    }
    state = advance(loaded["state"], action, now=now, **arguments)
    record = _sibling("evidence_ledger").append_event(
        _ledger_path(run_dir),
        event_type="state_transition",
        contract_sha256=state["contract_sha256"],
        expected_previous_sha256=tail,
        payload={"action": action, "now": now, "arguments": arguments, "state": state},
    )
    return {
        "state": state,
        "tail_sha256": record["record_sha256"],
        "event_count": loaded["event_count"] + 1,
    }


def _artifact_reason(artifact: Mapping[str, Any], fallback: str) -> str:
    reasons = artifact.get("reasons")
    if type(reasons) is list and reasons:
        return _identifier(reasons[0], "stability reason")
    return fallback


def _apply_calibration_state(
    state: Mapping[str, Any], calibration: Mapping[str, Any], *, now: float
) -> dict:
    timestamp = _finite(now, "now")
    if calibration["recorded_at"] != timestamp:
        raise ValidationError("calibration recorded_at must equal transition time")
    if calibration["contract_sha256"] != state["contract_sha256"]:
        raise ValidationError("calibration contract identity does not match the run")
    environment_state = calibration["environment_state"]
    if environment_state == "green":
        return advance(
            state,
            "start_exploration",
            now=timestamp,
            environment_state="green",
            measurable=calibration["measurable"],
        )
    if environment_state == "yellow":
        return advance(
            state,
            "environment_yellow",
            now=timestamp,
            reason=_artifact_reason(calibration, "stability_inconclusive"),
        )
    return advance(
        state,
        "environment_red",
        now=timestamp,
        reason=_artifact_reason(calibration, "stability_guardrail_failed"),
    )


def apply_run_calibration(
    contract_path: str | os.PathLike,
    run_dir: str | os.PathLike,
    calibration: Mapping[str, Any],
    *,
    now: float,
    controller_seal_key: bytes,
    expected_tail_sha256: str | None = None,
) -> dict:
    """Advance CALIBRATING only from a verified Controller artifact."""
    loaded = load_run(
        contract_path, run_dir, controller_seal_key=controller_seal_key
    )
    tail = _expected_tail(loaded, expected_tail_sha256)
    clean = _sibling("stability_calibration").validate_calibration(
        calibration,
        contract_path=contract_path,
        controller_seal_key=controller_seal_key,
    )
    state = _apply_calibration_state(loaded["state"], clean, now=now)
    record = _sibling("evidence_ledger").append_event(
        _ledger_path(run_dir),
        event_type="stability_calibrated",
        contract_sha256=state["contract_sha256"],
        expected_previous_sha256=tail,
        payload={"calibration": clean, "now": now, "state": state},
    )
    return {
        "state": state,
        "tail_sha256": record["record_sha256"],
        "event_count": loaded["event_count"] + 1,
    }


def _apply_stability_audit_state(
    state: Mapping[str, Any], audit: Mapping[str, Any], *, now: float
) -> dict:
    timestamp = _finite(now, "now")
    if audit["recorded_at"] != timestamp:
        raise ValidationError("stability audit recorded_at must equal transition time")
    if audit["contract_sha256"] != state["contract_sha256"]:
        raise ValidationError("stability audit contract identity does not match the run")
    environment_state = audit["environment_state"]
    if environment_state == "green":
        return advance(state, "audit_pass", now=timestamp)
    if environment_state == "yellow":
        return advance(
            state,
            "environment_yellow",
            now=timestamp,
            reason=_artifact_reason(audit, "stability_audit_inconclusive"),
        )
    return advance(
        state,
        "environment_red",
        now=timestamp,
        reason=_artifact_reason(audit, "stability_audit_guardrail_failed"),
    )


def apply_run_stability_audit(
    contract_path: str | os.PathLike,
    run_dir: str | os.PathLike,
    audit: Mapping[str, Any],
    *,
    now: float,
    controller_seal_key: bytes,
    expected_tail_sha256: str | None = None,
) -> dict:
    """Resume or stop AUDITING only from a Controller-attested replay."""
    contract = _sibling("workload_contract").verify_frozen_contract(contract_path)
    loaded = load_run(
        contract_path, run_dir, controller_seal_key=controller_seal_key
    )
    tail = _expected_tail(loaded, expected_tail_sha256)
    clean = _sibling("stability_calibration").validate_audit(
        audit, controller_seal_key=controller_seal_key
    )
    if clean["contract_sha256"] != contract["contract_sha256"]:
        raise ValidationError("stability audit contract identity does not match the run")
    records = _sibling("evidence_ledger").verify_ledger(
        _ledger_path(run_dir),
        expected_contract_sha256=contract["contract_sha256"],
    )
    anchors = [
        record["payload"]["calibration"]["calibration_sha256"]
        for record in records
        if record["event_type"] == "stability_calibrated"
    ]
    if not anchors or clean["anchor_calibration_sha256"] != anchors[-1]:
        raise ValidationError("stability audit anchor does not match this run")
    state = _apply_stability_audit_state(loaded["state"], clean, now=now)
    record = _sibling("evidence_ledger").append_event(
        _ledger_path(run_dir),
        event_type="stability_audited",
        contract_sha256=state["contract_sha256"],
        expected_previous_sha256=tail,
        payload={"audit": clean, "now": now, "state": state},
    )
    return {
        "state": state,
        "tail_sha256": record["record_sha256"],
        "event_count": loaded["event_count"] + 1,
    }


def _enforce_audit_cadence(
    contract: Mapping[str, Any], records: list[Mapping[str, Any]]
) -> None:
    cadence = contract["stability"]["audit_every_candidates"]
    last_stability_index = -1
    for index, record in enumerate(records):
        if record["event_type"] in {"stability_calibrated", "stability_audited"}:
            last_stability_index = index
    if last_stability_index < 0:
        raise ValidationError("candidate registration requires stability calibration")
    candidates_since = sum(
        record["event_type"] == "candidate_registered"
        for record in records[last_stability_index + 1 :]
    )
    if candidates_since >= cadence:
        raise ValidationError("stability audit cadence reached before candidate registration")


def register_run_candidate(
    contract_path: str | os.PathLike,
    run_dir: str | os.PathLike,
    proposal: Mapping[str, Any],
    *,
    admission: Mapping[str, Any],
    controller_seal_key: bytes,
    now: float,
    expected_tail_sha256: str | None = None,
) -> dict:
    contract = _sibling("workload_contract").verify_frozen_contract(contract_path)
    loaded = load_run(
        contract_path, run_dir, controller_seal_key=controller_seal_key
    )
    records = _sibling("evidence_ledger").verify_ledger(
        _ledger_path(run_dir),
        expected_contract_sha256=contract["contract_sha256"],
    )
    _enforce_audit_cadence(contract, records)
    tail = _expected_tail(loaded, expected_tail_sha256)
    state = register_candidate(
        loaded["state"],
        proposal,
        admission=admission,
        controller_seal_key=controller_seal_key,
        now=now,
    )
    clean_proposal = validate_candidate_proposal(proposal)
    record = _sibling("evidence_ledger").append_event(
        _ledger_path(run_dir),
        event_type="candidate_registered",
        contract_sha256=contract["contract_sha256"],
        expected_previous_sha256=tail,
        payload={
            "proposal": clean_proposal,
            "admission": dict(admission),
            "now": now,
            "state": state,
        },
    )
    return {
        "state": state,
        "tail_sha256": record["record_sha256"],
        "event_count": loaded["event_count"] + 1,
    }


def resolve_run_candidate(
    contract_path: str | os.PathLike,
    run_dir: str | os.PathLike,
    *,
    candidate_id: str,
    outcome: str,
    correctness_ok: bool | None,
    performance_gate_passed: bool | None,
    now: float,
    expected_tail_sha256: str | None = None,
    controller_seal_key: bytes | None = None,
) -> dict:
    loaded = load_run(
        contract_path, run_dir, controller_seal_key=controller_seal_key
    )
    tail = _expected_tail(loaded, expected_tail_sha256)
    state = resolve_candidate(
        loaded["state"],
        candidate_id=candidate_id,
        outcome=outcome,
        correctness_ok=correctness_ok,
        performance_gate_passed=performance_gate_passed,
        now=now,
    )
    record = _sibling("evidence_ledger").append_event(
        _ledger_path(run_dir),
        event_type="candidate_resolved",
        contract_sha256=state["contract_sha256"],
        expected_previous_sha256=tail,
        payload={
            "candidate_id": candidate_id,
            "outcome": outcome,
            "correctness_ok": correctness_ok,
            "performance_gate_passed": performance_gate_passed,
            "now": now,
            "state": state,
        },
    )
    return {
        "state": state,
        "tail_sha256": record["record_sha256"],
        "event_count": loaded["event_count"] + 1,
    }
