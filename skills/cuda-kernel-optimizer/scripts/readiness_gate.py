#!/usr/bin/env python3
"""Aggregate capability evidence and gate workload diagnosis."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any, Callable, Mapping


REPORT_SCHEMA = "cuda-workload-optimizer/readiness-report-v1"
MARKER_SCHEMA = "cuda-workload-optimizer/readiness-report-completion-v1"
STATE_SCHEMA = "cuda-workload-optimizer/readiness-gate-state-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTITY_FIELDS = {
    "toolchain_digest",
    "uid",
    "container_identity",
    "gpu_identity",
    "visible_devices",
    "permission_state",
}


def _load_sibling(name: str, module_name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_STORE = _load_sibling("artifact_store.py", "cuda_readiness_gate_store")
_CONTRACT = _load_sibling("readiness_contract.py", "cuda_readiness_gate_contract")
_PROBE = _load_sibling("readiness_probe.py", "cuda_readiness_gate_probe")
_INSTALL = _load_sibling("readiness_install.py", "cuda_readiness_gate_install")


class ValidationError(ValueError):
    """Raised when gate control or resumable evidence is unsafe."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValidationError(f"value must be strict JSON: {error}") from error


def _validate_environment_identity(value: Any) -> dict:
    if type(value) is not dict or set(value) != _IDENTITY_FIELDS:
        raise ValidationError(
            "environment_identity must contain exactly: "
            + ", ".join(sorted(_IDENTITY_FIELDS))
        )
    if (
        type(value["toolchain_digest"]) is not str
        or _SHA256.fullmatch(value["toolchain_digest"]) is None
    ):
        raise ValidationError("environment_identity.toolchain_digest must be SHA-256")
    uid = value["uid"]
    if uid is not None and (isinstance(uid, bool) or not isinstance(uid, int) or uid < 0):
        raise ValidationError("environment_identity.uid must be a non-negative integer or null")
    for field in (
        "container_identity",
        "gpu_identity",
        "permission_state",
    ):
        item = value[field]
        if item is not None and (type(item) is not str or len(item) > 4096):
            raise ValidationError(f"environment_identity.{field} must be a bounded string or null")
    visible = value["visible_devices"]
    if type(visible) is not dict or set(visible) != {"cuda", "nvidia"}:
        raise ValidationError(
            "environment_identity.visible_devices must contain cuda and nvidia"
        )
    for field, item in visible.items():
        if item is not None and (type(item) is not str or len(item) > 4096):
            raise ValidationError(
                f"environment_identity.visible_devices.{field} must be a bounded string or null"
            )
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def environment_identity_digest(value: Mapping[str, Any]) -> str:
    """Return the digest that invalidates all v1 readiness evidence on drift."""
    return hashlib.sha256(_canonical_bytes(_validate_environment_identity(value))).hexdigest()


def _validate_control(value: Any) -> tuple[Path, Path, dict, str]:
    if type(value) is not dict or set(value) != {
        "project_root",
        "environment_root",
        "environment_identity",
    }:
        raise ValidationError(
            "control must contain project_root, environment_root, and environment_identity"
        )
    project = _CONTRACT._safe_root(value["project_root"], "control.project_root")
    environment = _CONTRACT._safe_root(
        value["environment_root"], "control.environment_root"
    )
    identity = _validate_environment_identity(value["environment_identity"])
    return project, environment, identity, environment_identity_digest(identity)


def evaluate_result(
    requirement: Mapping[str, Any],
    probe: Mapping[str, Any],
    repairs_left: int,
    *,
    identity_digest: str,
    observed_at: float,
    evidence_path: str | None = None,
) -> dict:
    """Apply the fixed admission mapping to one validated probe result."""
    probe = _PROBE.validate_probe(probe, requirement["id"])
    status = probe["status"]
    remediation = requirement["remediation"]
    necessity = requirement["necessity"]
    if status == "ready":
        admission = "ready"
    elif remediation["mode"] == "isolated_pip" and repairs_left > 0:
        admission = "auto_fixable"
    elif remediation["mode"] == "user_action":
        admission = "user_action_required"
    elif necessity in {"diagnostic", "optional"}:
        admission = "degraded"
    else:
        admission = "blocked"
    return {
        "requirement_id": requirement["id"],
        "necessity": necessity,
        "phase": requirement["phase"],
        "kind": requirement["kind"],
        "probe_status": status,
        "admission_status": admission,
        "valid_until": observed_at + requirement["max_age_seconds"],
        "identity_digest": identity_digest,
        "unsupported_capabilities": (
            [requirement["kind"]]
            if admission in {"degraded", "user_action_required", "blocked"}
            else []
        ),
        "evidence_path": evidence_path,
    }


def _strict_json(raw: bytes, field: str) -> dict:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_CONTRACT._pairs_without_duplicates,
            parse_constant=_CONTRACT._invalid_number,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValidationError(f"invalid {field}: {error}") from error
    if type(value) is not dict:
        raise ValidationError(f"{field} must be an object")
    return value


def _load_prior_report(readiness_dir: Path) -> dict | None:
    marker_path = readiness_dir / "report.complete.json"
    try:
        marker_raw = _STORE.read_regular_bytes(marker_path)
    except ValueError as error:
        directory_fd = None
        try:
            directory_fd, leaf, _target = _STORE._open_parent_directory(
                marker_path, create=False
            )
            os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        finally:
            if directory_fd is not None:
                os.close(directory_fd)
        raise ValidationError(f"unsafe readiness report marker: {error}") from error
    marker = _strict_json(marker_raw, "readiness report marker")
    if set(marker) != {"schema_version", "report_sha256", "published_at"}:
        raise ValidationError("readiness report marker fields are invalid")
    if marker["schema_version"] != MARKER_SCHEMA:
        raise ValidationError("readiness report marker schema is invalid")
    digest = marker["report_sha256"]
    if type(digest) is not str or _SHA256.fullmatch(digest) is None:
        raise ValidationError("readiness report marker digest is invalid")
    try:
        report_raw = _STORE.read_regular_bytes(readiness_dir / "report.json")
    except ValueError as error:
        raise ValidationError(f"readiness report is missing or unsafe: {error}") from error
    if hashlib.sha256(report_raw).hexdigest() != digest:
        raise ValidationError("readiness report digest does not match completion marker")
    report = _strict_json(report_raw, "readiness report")
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValidationError("readiness report schema is invalid")
    return report


def _publish_report(readiness_dir: Path, report: dict, published_at: float) -> None:
    marker_path = readiness_dir / "report.complete.json"
    _STORE.remove_regular_file(marker_path, missing_ok=True)
    report_bytes = (
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    _STORE.atomic_write_bytes(readiness_dir / "report.json", report_bytes)
    marker = {
        "schema_version": MARKER_SCHEMA,
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "published_at": published_at,
    }
    _STORE.atomic_write_json(marker_path, marker)


def _load_gate_state(readiness_dir: Path) -> dict | None:
    path = readiness_dir / "gate_state.json"
    try:
        raw = _STORE.read_regular_bytes(path)
    except ValueError as error:
        if "does not exist" in str(error):
            return None
        raise ValidationError(f"unsafe readiness gate state: {error}") from error
    envelope = _strict_json(raw, "readiness gate state")
    if set(envelope) != {"schema_version", "state", "state_sha256"}:
        raise ValidationError("readiness gate state fields are invalid")
    if envelope["schema_version"] != STATE_SCHEMA:
        raise ValidationError("readiness gate state schema is invalid")
    state = envelope["state"]
    if type(state) is not dict or set(state) != {
        "contract_digest",
        "started_at",
        "repairs_used",
    }:
        raise ValidationError("readiness gate state payload is invalid")
    if hashlib.sha256(_canonical_bytes(state)).hexdigest() != envelope["state_sha256"]:
        raise ValidationError("readiness gate state digest is invalid")
    if (
        type(state["contract_digest"]) is not str
        or _SHA256.fullmatch(state["contract_digest"]) is None
        or isinstance(state["started_at"], bool)
        or not isinstance(state["started_at"], (int, float))
        or not math.isfinite(float(state["started_at"]))
        or isinstance(state["repairs_used"], bool)
        or not isinstance(state["repairs_used"], int)
        or state["repairs_used"] < 0
    ):
        raise ValidationError("readiness gate state values are invalid")
    return state


def _write_gate_state(
    readiness_dir: Path, *, contract_digest: str, started_at: float, repairs_used: int
) -> None:
    state = {
        "contract_digest": contract_digest,
        "started_at": started_at,
        "repairs_used": repairs_used,
    }
    envelope = {
        "schema_version": STATE_SCHEMA,
        "state": state,
        "state_sha256": hashlib.sha256(_canonical_bytes(state)).hexdigest(),
    }
    _STORE.atomic_write_json(readiness_dir / "gate_state.json", envelope)


def _synthetic_probe(requirement_id: str, status: str, reason: str) -> dict:
    return {
        "schema_version": _PROBE.PROBE_SCHEMA,
        "requirement_id": requirement_id,
        "status": status,
        "observations": {"reason": reason},
        "artifacts": [],
    }


def _final_status(results: list[dict]) -> str:
    admissions = {item["admission_status"] for item in results}
    if "blocked" in admissions:
        return "blocked"
    if "user_action_required" in admissions:
        return "user_action_required"
    if "degraded" in admissions:
        return "degraded"
    return "ready"


def _next_actions(results: list[dict], requirements: dict[str, dict]) -> list[str]:
    actions = []
    for result in results:
        admission = result["admission_status"]
        if admission == "ready":
            continue
        requirement = requirements[result["requirement_id"]]
        remediation = requirement["remediation"]
        if admission == "user_action_required":
            action = remediation["message"]
        elif admission == "degraded":
            action = (
                f"Capability unavailable: {requirement['id']} ({requirement['kind']})."
            )
        else:
            action = f"Resolve required readiness requirement: {requirement['id']}."
        if action not in actions:
            actions.append(action)
    return actions


def run_gate(
    *,
    contract: Mapping[str, Any],
    control: Mapping[str, Any],
    run_dir: Path,
    probe_runner: Callable = _PROBE.run_requirement,
    installer: Callable = _INSTALL.install_isolated_pip,
    identity_provider: Callable[[], Mapping[str, Any]] | None = None,
    now: Callable[[], float] = time.time,
) -> dict:
    """Run ordered probes, bounded remediation, and marker-last report publish."""
    project, environment, _identity, identity_digest = _validate_control(control)
    validated_contract = _CONTRACT.validate_contract(
        contract, project_root=project, environment_root=environment
    )
    contract_digest = _CONTRACT.contract_digest(validated_contract)
    readiness_dir = Path(run_dir) / "readiness"
    prior = _load_prior_report(readiness_dir)
    gate_state = _load_gate_state(readiness_dir)
    current_time = float(now())
    if not math.isfinite(current_time):
        raise ValidationError("now() must return a finite epoch")
    same_contract = prior is not None and prior.get("contract_digest") == contract_digest
    same_identity = (
        same_contract
        and prior.get("environment_identity_digest") == identity_digest
    )
    if gate_state is not None and gate_state["contract_digest"] != contract_digest:
        raise ValidationError(
            "readiness contract changed inside an existing run; create a child run"
        )
    started_at = (
        float(gate_state["started_at"])
        if gate_state is not None
        else (
            float(prior["started_at"])
            if same_contract and isinstance(prior.get("started_at"), (int, float))
            else current_time
        )
    )
    budget = validated_contract["budget"]
    deadline_epoch = started_at + float(budget["max_seconds"])
    repairs_used = max(
        int(gate_state["repairs_used"]) if gate_state is not None else 0,
        (
            int(prior.get("budget", {}).get("repairs_used", 0))
            if same_contract
            else 0
        ),
    )
    _write_gate_state(
        readiness_dir,
        contract_digest=contract_digest,
        started_at=started_at,
        repairs_used=repairs_used,
    )
    prior_elapsed = (
        float(prior.get("budget", {}).get("elapsed_seconds", 0.0))
        if same_contract
        else 0.0
    )
    prior_results = {
        item.get("requirement_id"): item
        for item in (prior.get("results", []) if same_identity else [])
        if type(item) is dict
    }
    requirements_by_id = {
        item["id"]: item for item in validated_contract["requirements"]
    }
    results = []

    ordered = sorted(
        enumerate(validated_contract["requirements"]),
        key=lambda pair: (0 if pair[1]["phase"] == "foundation" else 1, pair[0]),
    )
    position = 0
    while position < len(ordered):
        _index, requirement = ordered[position]
        requirement_id = requirement["id"]
        cached = prior_results.get(requirement_id)
        reusable = (
            cached is not None
            and cached.get("identity_digest") == identity_digest
            and isinstance(cached.get("valid_until"), (int, float))
            and not isinstance(cached.get("valid_until"), bool)
            and math.isfinite(float(cached["valid_until"]))
            and float(cached["valid_until"]) > current_time
        )
        if reusable:
            result = dict(cached)
        else:
            attempt_root = (
                readiness_dir
                / "attempts"
                / f"{requirement_id}-{secrets.token_hex(8)}"
            )
            evidence_path = str(
                (
                    attempt_root
                    / "readiness"
                    / "probes"
                    / f"{requirement_id}.json"
                ).relative_to(Path(run_dir))
            )
            if current_time >= deadline_epoch:
                probe = _synthetic_probe(
                    requirement_id,
                    "unavailable",
                    "readiness_deadline_exhausted",
                )
            else:
                try:
                    probe = probe_runner(
                        requirement,
                        run_dir=attempt_root,
                        project_root=project,
                        environment_identity_digest=identity_digest,
                        deadline_epoch=deadline_epoch,
                    )
                except (OSError, ValueError) as error:
                    probe = _synthetic_probe(
                        requirement_id,
                        "failed",
                        f"probe_runner_failed: {error}",
                    )
            observed_at = float(now())
            current_time = max(current_time, observed_at)
            repairs_left = max(0, int(budget["max_repairs"]) - repairs_used)
            result = evaluate_result(
                requirement,
                probe,
                repairs_left,
                identity_digest=identity_digest,
                observed_at=observed_at,
                evidence_path=evidence_path,
            )
            if result["admission_status"] == "auto_fixable":
                repairs_used += 1
                _write_gate_state(
                    readiness_dir,
                    contract_digest=contract_digest,
                    started_at=started_at,
                    repairs_used=repairs_used,
                )
                install_result = installer(
                    requirement["remediation"],
                    project_root=project,
                    environment_root=environment,
                    run_dir=attempt_root,
                    deadline_epoch=deadline_epoch,
                )
                current_time = max(current_time, float(now()))
                if install_result.get("status") != "succeeded":
                    result["admission_status"] = "blocked"
                    result["unsupported_capabilities"] = [requirement["kind"]]
                elif current_time >= deadline_epoch:
                    result["admission_status"] = "blocked"
                    result["unsupported_capabilities"] = [requirement["kind"]]
                else:
                    try:
                        refreshed_identity = _validate_environment_identity(
                            (
                                identity_provider()
                                if identity_provider is not None
                                else _identity
                            )
                        )
                        refreshed_digest = environment_identity_digest(
                            refreshed_identity
                        )
                    except (OSError, ValueError):
                        refreshed_digest = identity_digest
                    if refreshed_digest == identity_digest:
                        result["admission_status"] = "blocked"
                        result["unsupported_capabilities"] = [requirement["kind"]]
                    else:
                        _identity = refreshed_identity
                        identity_digest = refreshed_digest
                        prior_results = {}
                        results = []
                        position = 0
                        continue
        results.append(result)
        if (
            requirement["phase"] == "foundation"
            and requirement["necessity"] == "required"
            and result["admission_status"] != "ready"
        ):
            break
        position += 1

    finished_at = max(current_time, float(now()))
    elapsed = max(prior_elapsed, max(0.0, finished_at - started_at))
    status = _final_status(results)
    all_required_ready = all(
        result["admission_status"] == "ready"
        for result in results
        if result["necessity"] == "required"
    ) and all(
        requirement["id"] in {item["requirement_id"] for item in results}
        for requirement in validated_contract["requirements"]
        if requirement["necessity"] == "required"
    )
    status_counts = {
        name: sum(item["admission_status"] == name for item in results)
        for name in ("ready", "degraded", "user_action_required", "blocked")
    }
    requested_claim = validated_contract["requested_claim"]
    report = {
        "schema_version": REPORT_SCHEMA,
        "requested_claim": requested_claim,
        "status": status,
        "can_start_diagnosis": all_required_ready,
        "claim_ceiling": requested_claim if all_required_ready else "static",
        "contract_digest": contract_digest,
        "environment_identity_digest": identity_digest,
        "started_at": started_at,
        "finished_at": finished_at,
        "budget": {
            "max_seconds": budget["max_seconds"],
            "elapsed_seconds": elapsed,
            "max_repairs": budget["max_repairs"],
            "repairs_used": repairs_used,
        },
        "results": results,
        "status_counts": status_counts,
        "next_actions": _next_actions(results, requirements_by_id),
    }
    _publish_report(readiness_dir, report, finished_at)
    return report
