#!/usr/bin/env python3
"""Build a bounded observation summary from verified ledger artifacts."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.util
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SUMMARY_SCHEMA = "cuda-optimizer/observation-summary-v1"
GATE_SCHEMA = "cuda-optimizer/gate-resolution-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_GATE_KINDS = {
    "correctness_reference",
    "dispatch_identity",
    "target_compile_probe",
    "candidate_correctness",
    "paired_measurement",
    "workload_replay",
}
_DIAGNOSTIC_KINDS = {"nsys_timeline", "pytorch_profile"}
_KINDS = _GATE_KINDS | _DIAGNOSTIC_KINDS
_GATES = {
    "pre_execution": {
        "correctness_reference",
        "dispatch_identity",
        "target_compile_probe",
    },
    "promotion": {
        "candidate_correctness",
        "paired_measurement",
        "workload_replay",
    },
}
_OBSERVATION_FIELDS = {
    "run_id",
    "ledger_id",
    "observation_id",
    "artifact",
    "adapter_implementation_sha256",
    "adapter_request_sha256",
    "controller_attestation",
}
_ARTIFACT_FIELDS = {
    "path",
    "sha256",
    "size_bytes",
}


class ValidationError(ValueError):
    pass


def _sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"cuda_summary_{name}", path)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise ValidationError(f"cannot load {name}.py")
    spec.loader.exec_module(module)
    return module


_LEDGER = _sibling("evidence_ledger")
_ARTIFACT_STORE = _sibling("artifact_store")
_GATE_EVIDENCE = _sibling("gate_evidence")
_DIAGNOSTIC_EVIDENCE = _sibling("diagnostic_evidence")


def _closed(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    missing = fields - set(value)
    extra = set(value) - fields
    if missing or extra:
        raise ValidationError(
            f"{label} must be closed; missing={sorted(missing)} extra={sorted(extra)}"
        )


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be lowercase SHA-256")
    return value


def _identifier(value: Any, label: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a safe identifier")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValidationError(f"{label} must be finite")
    number = float(value)
    if positive and number <= 0:
        raise ValidationError(f"{label} must be positive")
    return number


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValidationError(f"{label} must be a positive integer")
    return value


def _canonical_digest(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _seal_key(value: Any) -> bytes:
    if not isinstance(value, bytes) or len(value) < 32:
        raise ValidationError("controller_seal_key must contain at least 32 bytes")
    return value


def _controller_attestation(
    key: bytes,
    *,
    contract_sha256: str,
    environment_sha256: str,
    run_id: str,
    ledger_id: str,
    observation_id: str,
    artifact: Mapping[str, Any],
    adapter_implementation_sha256: str,
    adapter_request_sha256: str,
) -> str:
    material = {
        "event_type": "observation_sealed",
        "contract_sha256": contract_sha256,
        "environment_sha256": environment_sha256,
        "run_id": run_id,
        "ledger_id": ledger_id,
        "observation_id": observation_id,
        "artifact": dict(artifact),
        "adapter_implementation_sha256": adapter_implementation_sha256,
        "adapter_request_sha256": adapter_request_sha256,
    }
    return hmac.new(key, _canonical_bytes(material), hashlib.sha256).hexdigest()


def _relative_path(value: Any) -> str:
    if type(value) is not str or not value:
        raise ValidationError("artifact.path must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value in {".", ".."}:
        raise ValidationError("artifact.path must remain under artifact_root")
    normalized = os.path.normpath(value)
    if normalized in {"", ".", ".."}:
        raise ValidationError("artifact.path must name a file")
    return normalized


def _strings(value: Any, label: str) -> list[str]:
    if type(value) is not list:
        raise ValidationError(f"{label} must be an array")
    result = []
    for index, item in enumerate(value):
        result.append(_identifier(item, f"{label}[{index}]") )
    if len(result) != len(set(result)):
        raise ValidationError(f"{label} must not contain duplicates")
    return result


def _validate_payload(
    payload: Any,
    *,
    artifact_root: Path,
    contract_sha256: str,
    environment_sha256: str,
    run_id: str,
    ledger_id: str,
    controller_seal_key: bytes,
    as_of: float,
    max_age_seconds: float,
) -> dict:
    if type(payload) is not dict:
        raise ValidationError("observation payload must be an object")
    _closed(payload, _OBSERVATION_FIELDS, "observation payload")
    payload_run_id = _identifier(payload["run_id"], "run_id")
    payload_ledger_id = _identifier(payload["ledger_id"], "ledger_id")
    if payload_run_id != run_id or payload_ledger_id != ledger_id:
        raise ValidationError("observation run or ledger identity mismatch")
    observation_id = _identifier(payload["observation_id"], "observation_id")
    adapter_implementation = _sha(
        payload["adapter_implementation_sha256"],
        "adapter_implementation_sha256",
    )
    adapter_request = _sha(
        payload["adapter_request_sha256"], "adapter_request_sha256"
    )
    artifact = payload["artifact"]
    if type(artifact) is not dict:
        raise ValidationError("artifact must be an object")
    _closed(artifact, _ARTIFACT_FIELDS, "artifact")
    relative = _relative_path(artifact["path"])
    expected_sha = _sha(artifact["sha256"], "artifact.sha256")
    if type(artifact["size_bytes"]) is not int or artifact["size_bytes"] < 1:
        raise ValidationError("artifact.size_bytes must be a positive integer")
    attestation = _sha(
        payload["controller_attestation"], "controller_attestation"
    )
    expected_attestation = _controller_attestation(
        _seal_key(controller_seal_key),
        contract_sha256=contract_sha256,
        environment_sha256=environment_sha256,
        run_id=run_id,
        ledger_id=ledger_id,
        observation_id=observation_id,
        artifact=artifact,
        adapter_implementation_sha256=adapter_implementation,
        adapter_request_sha256=adapter_request,
    )
    if not hmac.compare_digest(attestation, expected_attestation):
        raise ValidationError("controller attestation does not match this observation")
    try:
        raw = _ARTIFACT_STORE.read_regular_bytes(artifact_root / relative)
    except ValueError as exc:
        raise ValidationError(f"observation artifact is missing, a symlink, or unsafe: {relative}") from exc
    if len(raw) != artifact["size_bytes"]:
        raise ValidationError(f"observation artifact size identity changed: {relative}")
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValidationError(f"observation artifact hash identity changed: {relative}")
    try:
        artifact_value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid evidence artifact {relative}: strict JSON required") from exc
    schema_version = artifact_value.get("schema_version") if type(artifact_value) is dict else None
    if schema_version == _GATE_EVIDENCE.SCHEMA:
        validator = _GATE_EVIDENCE.validate_gate_evidence
        label = "gate"
    elif schema_version == _DIAGNOSTIC_EVIDENCE.EVIDENCE_SCHEMA:
        validator = _DIAGNOSTIC_EVIDENCE.validate_diagnostic_evidence
        label = "diagnostic"
    else:
        raise ValidationError(f"unsupported evidence schema in artifact {relative}")
    try:
        derived = validator(
            raw,
            expected_contract_sha256=contract_sha256,
            expected_environment_sha256=environment_sha256,
        )
    except ValueError as exc:
        raise ValidationError(f"invalid {label} evidence artifact {relative}: {exc}") from exc
    if derived["producer"]["implementation_sha256"] != adapter_implementation:
        raise ValidationError("adapter implementation identity mismatch")
    if derived["adapter_request_sha256"] != adapter_request:
        raise ValidationError("adapter request identity mismatch")

    recorded_at = derived["recorded_at"]
    if recorded_at > as_of:
        freshness = "future"
        age_seconds = 0.0
    else:
        age_seconds = as_of - recorded_at
        freshness = "stale" if age_seconds > max_age_seconds else "current"
    return {
        "observation_id": observation_id,
        "adapter_request_sha256": adapter_request,
        "kind": derived["kind"],
        "layer": derived["layer"],
        "summary": derived["summary"],
        "signals": derived["signals"],
        "producer": derived["producer"],
        "subject": derived["subject"],
        "result": derived["result"],
        "artifact": {
            **artifact,
            "recorded_at": recorded_at,
            "environment_sha256": environment_sha256,
        },
        "age_seconds": age_seconds,
        "freshness": freshness,
    }


def build_summary(
    ledger_path: str | os.PathLike,
    *,
    artifact_root: str | os.PathLike,
    contract_sha256: str,
    environment_sha256: str,
    run_id: str,
    ledger_id: str,
    as_of: float,
    max_age_seconds: float,
    max_observations: int,
    context_budget_bytes: int,
    controller_seal_key: bytes,
) -> dict:
    """Verify a ledger and its artifacts, then return bounded observation metadata."""
    contract = _sha(contract_sha256, "contract_sha256")
    environment = _sha(environment_sha256, "environment_sha256")
    clean_run_id = _identifier(run_id, "run_id")
    clean_ledger_id = _identifier(ledger_id, "ledger_id")
    timestamp = _finite(as_of, "as_of")
    if timestamp < 0:
        raise ValidationError("as_of must be non-negative")
    max_age = _finite(max_age_seconds, "max_age_seconds", positive=True)
    observation_limit = _positive_int(max_observations, "max_observations")
    context_budget = _positive_int(context_budget_bytes, "context_budget_bytes")
    root = Path(os.path.abspath(os.path.expanduser(os.fspath(artifact_root))))
    if root.is_symlink() or not root.is_dir():
        raise ValidationError("artifact_root is missing, a symlink, or unsafe")
    records = _LEDGER.verify_ledger(
        ledger_path, expected_contract_sha256=contract
    )
    observations = []
    seen = set()
    for record in records:
        if record["event_type"] != "observation_sealed":
            continue
        if len(observations) >= observation_limit:
            raise ValidationError("sealed observation count exceeds max_observations limit")
        observation = _validate_payload(
            record["payload"],
            artifact_root=root,
            contract_sha256=contract,
            environment_sha256=environment,
            run_id=clean_run_id,
            ledger_id=clean_ledger_id,
            controller_seal_key=_seal_key(controller_seal_key),
            as_of=timestamp,
            max_age_seconds=max_age,
        )
        if observation["observation_id"] in seen:
            raise ValidationError(
                f"duplicate observation_id: {observation['observation_id']}"
            )
        seen.add(observation["observation_id"])
        observations.append(observation)
        selected_context_bytes = len(_canonical_bytes(observations))
        if selected_context_bytes > context_budget:
            raise ValidationError("observation context exceeds context_budget_bytes")
    if not observations:
        raise ValidationError("ledger contains no sealed observations")
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "run_id": clean_run_id,
        "ledger_id": clean_ledger_id,
        "contract_sha256": contract,
        "environment_sha256": environment,
        "as_of": timestamp,
        "max_age_seconds": max_age,
        "max_observations": observation_limit,
        "context_budget_bytes": context_budget,
        "selected_context_bytes": len(_canonical_bytes(observations)),
        "ledger_tail_sha256": records[-1]["record_sha256"],
        "observations": observations,
    }
    summary["summary_sha256"] = _canonical_digest(summary)
    return summary


def _append_controller_gate_observation(
    ledger_path: str | os.PathLike,
    *,
    artifact_root: str | os.PathLike,
    contract_sha256: str,
    environment_sha256: str,
    run_id: str,
    ledger_id: str,
    observation_id: str,
    artifact: Mapping[str, Any],
    adapter_implementation_sha256: str,
    adapter_request_sha256: str,
    as_of: float,
    max_age_seconds: float,
    controller_seal_key: bytes,
    expected_previous_sha256: str | None = None,
) -> dict:
    """Validate Controller-derived evidence before appending the reserved event.

    The historical name is retained as an internal compatibility surface; the
    validator accepts both gate and diagnostic evidence schemas.
    """
    contract = _sha(contract_sha256, "contract_sha256")
    environment = _sha(environment_sha256, "environment_sha256")
    clean_run_id = _identifier(run_id, "run_id")
    clean_ledger_id = _identifier(ledger_id, "ledger_id")
    implementation = _sha(
        adapter_implementation_sha256, "adapter_implementation_sha256"
    )
    request_sha = _sha(adapter_request_sha256, "adapter_request_sha256")
    key = _seal_key(controller_seal_key)
    clean_artifact = dict(artifact)
    payload = {
        "run_id": clean_run_id,
        "ledger_id": clean_ledger_id,
        "observation_id": observation_id,
        "artifact": clean_artifact,
        "adapter_implementation_sha256": implementation,
        "adapter_request_sha256": request_sha,
        "controller_attestation": _controller_attestation(
            key,
            contract_sha256=contract,
            environment_sha256=environment,
            run_id=clean_run_id,
            ledger_id=clean_ledger_id,
            observation_id=observation_id,
            artifact=clean_artifact,
            adapter_implementation_sha256=implementation,
            adapter_request_sha256=request_sha,
        ),
    }
    _validate_payload(
        payload,
        artifact_root=Path(os.path.abspath(os.fspath(artifact_root))),
        contract_sha256=contract,
        environment_sha256=environment,
        run_id=clean_run_id,
        ledger_id=clean_ledger_id,
        controller_seal_key=key,
        as_of=_finite(as_of, "as_of"),
        max_age_seconds=_finite(
            max_age_seconds, "max_age_seconds", positive=True
        ),
    )
    return _LEDGER._append_reserved_event(
        ledger_path,
        event_type="observation_sealed",
        contract_sha256=contract_sha256,
        payload=payload,
        expected_previous_sha256=expected_previous_sha256,
    )


def _validate_summary(value: Any) -> dict:
    if type(value) is not dict:
        raise ValidationError("summary must be an object")
    fields = {
        "schema_version",
        "run_id",
        "ledger_id",
        "contract_sha256",
        "environment_sha256",
        "as_of",
        "max_age_seconds",
        "max_observations",
        "context_budget_bytes",
        "selected_context_bytes",
        "ledger_tail_sha256",
        "observations",
        "summary_sha256",
    }
    _closed(value, fields, "summary")
    if value["schema_version"] != SUMMARY_SCHEMA:
        raise ValidationError("unsupported summary schema")
    _identifier(value["run_id"], "run_id")
    _identifier(value["ledger_id"], "ledger_id")
    for field in ("contract_sha256", "environment_sha256", "ledger_tail_sha256"):
        _sha(value[field], field)
    _finite(value["as_of"], "as_of")
    _finite(value["max_age_seconds"], "max_age_seconds", positive=True)
    _positive_int(value["max_observations"], "max_observations")
    _positive_int(value["context_budget_bytes"], "context_budget_bytes")
    if type(value["selected_context_bytes"]) is not int or value["selected_context_bytes"] < 1:
        raise ValidationError("selected_context_bytes must be a positive integer")
    if type(value["observations"]) is not list or not value["observations"]:
        raise ValidationError("summary observations must be non-empty")
    if len(value["observations"]) > value["max_observations"]:
        raise ValidationError("summary exceeds max_observations")
    actual_context_bytes = len(_canonical_bytes(value["observations"]))
    if actual_context_bytes != value["selected_context_bytes"]:
        raise ValidationError("summary selected_context_bytes changed")
    if actual_context_bytes > value["context_budget_bytes"]:
        raise ValidationError("summary exceeds context_budget_bytes")
    expected = _sha(value["summary_sha256"], "summary_sha256")
    unsigned = dict(value)
    unsigned.pop("summary_sha256")
    if _canonical_digest(unsigned) != expected:
        raise ValidationError("summary digest changed")
    return value


def verify_summary(
    value: Mapping[str, Any],
    *,
    ledger_path: str | os.PathLike,
    artifact_root: str | os.PathLike,
    controller_seal_key: bytes,
) -> dict:
    """Rebuild a persisted summary from its ledger and artifacts."""
    clean = _validate_summary(value)
    rebuilt = build_summary(
        ledger_path,
        artifact_root=artifact_root,
        contract_sha256=clean["contract_sha256"],
        environment_sha256=clean["environment_sha256"],
        run_id=clean["run_id"],
        ledger_id=clean["ledger_id"],
        as_of=clean["as_of"],
        max_age_seconds=clean["max_age_seconds"],
        max_observations=clean["max_observations"],
        context_budget_bytes=clean["context_budget_bytes"],
        controller_seal_key=_seal_key(controller_seal_key),
    )
    if rebuilt != clean:
        raise ValidationError("summary does not match its verified ledger snapshot")
    return rebuilt


def resolve_gate_requirements(
    summary: Mapping[str, Any],
    gate_requirements: Mapping[str, Any],
    *,
    ledger_path: str | os.PathLike | None = None,
    artifact_root: str | os.PathLike | None = None,
    expected_run_id: str | None = None,
    expected_ledger_id: str | None = None,
    expected_contract_sha256: str | None = None,
    expected_environment_sha256: str | None = None,
    current_as_of: float | None = None,
    max_age_seconds: float | None = None,
    expected_ledger_tail_sha256: str | None = None,
    expected_reference_sha256: str | None = None,
    expected_target_sha256: str | None = None,
    expected_workload_sha256: str | None = None,
    expected_candidate_id: str | None = None,
    expected_candidate_sha256: str | None = None,
    expected_arch: str | None = None,
    controller_seal_key: bytes | None = None,
) -> dict:
    """Resolve phase gates only against current, hash-bound observations."""
    if (
        ledger_path is None
        or artifact_root is None
        or expected_run_id is None
        or expected_ledger_id is None
        or expected_contract_sha256 is None
        or expected_environment_sha256 is None
        or current_as_of is None
        or max_age_seconds is None
        or expected_ledger_tail_sha256 is None
        or expected_reference_sha256 is None
        or expected_target_sha256 is None
        or expected_workload_sha256 is None
        or expected_arch is None
        or controller_seal_key is None
    ):
        raise ValidationError(
            "gate resolution requires controller-owned identities, time, ledger, and artifacts"
        )
    original = verify_summary(
        summary,
        ledger_path=ledger_path,
        artifact_root=artifact_root,
        controller_seal_key=_seal_key(controller_seal_key),
    )
    contract = _sha(expected_contract_sha256, "expected_contract_sha256")
    run_id = _identifier(expected_run_id, "expected_run_id")
    ledger_id = _identifier(expected_ledger_id, "expected_ledger_id")
    environment = _sha(
        expected_environment_sha256, "expected_environment_sha256"
    )
    tail = _sha(expected_ledger_tail_sha256, "expected_ledger_tail_sha256")
    reference_sha = _sha(expected_reference_sha256, "expected_reference_sha256")
    target_sha = _sha(expected_target_sha256, "expected_target_sha256")
    workload_sha = _sha(expected_workload_sha256, "expected_workload_sha256")
    candidate_id = (
        None
        if expected_candidate_id is None
        else _identifier(expected_candidate_id, "expected_candidate_id")
    )
    if type(expected_arch) is not str or re.fullmatch(r"sm_[0-9]+", expected_arch) is None:
        raise ValidationError("expected_arch must be an exact SM architecture")
    candidate_sha = (
        None
        if expected_candidate_sha256 is None
        else _sha(expected_candidate_sha256, "expected_candidate_sha256")
    )
    if candidate_id is not None and candidate_sha is None:
        raise ValidationError(
            "expected_candidate_sha256 is required with expected_candidate_id"
        )
    if original["contract_sha256"] != contract:
        raise ValidationError("summary contract identity does not match the controller")
    if original["run_id"] != run_id or original["ledger_id"] != ledger_id:
        raise ValidationError("summary run or ledger identity does not match the controller")
    if original["environment_sha256"] != environment:
        raise ValidationError("summary environment identity does not match the controller")
    if original["ledger_tail_sha256"] != tail:
        raise ValidationError("summary ledger identity does not match the controller")
    clean = build_summary(
        ledger_path,
        artifact_root=artifact_root,
        contract_sha256=contract,
        environment_sha256=environment,
        run_id=run_id,
        ledger_id=ledger_id,
        as_of=current_as_of,
        max_age_seconds=max_age_seconds,
        max_observations=original["max_observations"],
        context_budget_bytes=original["context_budget_bytes"],
        controller_seal_key=_seal_key(controller_seal_key),
    )
    if clean["ledger_tail_sha256"] != tail:
        raise ValidationError("ledger advanced after the controller snapshot")
    if type(gate_requirements) is not dict or set(gate_requirements) != set(_GATES):
        raise ValidationError("gate_requirements must contain both phases")
    result = {
        "schema_version": GATE_SCHEMA,
        "run_id": clean["run_id"],
        "ledger_id": clean["ledger_id"],
        "contract_sha256": clean["contract_sha256"],
        "environment_sha256": clean["environment_sha256"],
        "observation_summary_sha256": clean["summary_sha256"],
    }
    for phase, expected in _GATES.items():
        requested = _strings(gate_requirements[phase], f"gate_requirements.{phase}")
        if set(requested) != expected:
            raise ValidationError(f"invalid gate set for {phase}")
        gates = []
        missing = []
        for gate in requested:
            def identity_matches(observation: Mapping[str, Any]) -> bool:
                subject = observation.get("subject", {})
                result = observation.get("result", {})
                if gate == "correctness_reference":
                    return subject.get("reference_sha256") == reference_sha
                if gate in {"dispatch_identity", "target_compile_probe"}:
                    if subject.get("target_sha256") != target_sha:
                        return False
                    return gate != "target_compile_probe" or result.get("arch") == expected_arch
                if (
                    candidate_id is None
                    or subject.get("candidate_id") != candidate_id
                    or subject.get("candidate_sha256") != candidate_sha
                ):
                    return False
                if gate == "candidate_correctness":
                    return result.get("reference_sha256") == reference_sha
                if gate == "workload_replay":
                    return result.get("workload_sha256") == workload_sha
                return gate == "paired_measurement"

            matches = [
                observation
                for observation in clean["observations"]
                if observation.get("kind") == gate
                and observation.get("freshness") == "current"
                and identity_matches(observation)
            ]
            satisfied = bool(matches)
            if not satisfied:
                missing.append(gate)
            gates.append(
                {
                    "gate": gate,
                    "satisfied": satisfied,
                    "observation_ids": [item["observation_id"] for item in matches],
                    "artifact_sha256s": [item["artifact"]["sha256"] for item in matches],
                }
            )
        result[phase] = {
            "satisfied": not missing,
            "missing_gates": missing,
            "gates": gates,
        }
    unsigned = dict(result)
    result["resolution_sha256"] = _canonical_digest(unsigned)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--environment-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ledger-id", required=True)
    parser.add_argument("--as-of", required=True, type=float)
    parser.add_argument("--max-age-seconds", required=True, type=float)
    parser.add_argument("--max-observations", required=True, type=int)
    parser.add_argument("--context-budget-bytes", required=True, type=int)
    parser.add_argument("--controller-seal-key-file", required=True)
    args = parser.parse_args(argv)
    seal_key = _ARTIFACT_STORE.read_regular_bytes(args.controller_seal_key_file)
    result = build_summary(
        args.ledger,
        artifact_root=args.artifact_root,
        contract_sha256=args.contract_sha256,
        environment_sha256=args.environment_sha256,
        run_id=args.run_id,
        ledger_id=args.ledger_id,
        as_of=args.as_of,
        max_age_seconds=args.max_age_seconds,
        max_observations=args.max_observations,
        context_budget_bytes=args.context_budget_bytes,
        controller_seal_key=seal_key,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
