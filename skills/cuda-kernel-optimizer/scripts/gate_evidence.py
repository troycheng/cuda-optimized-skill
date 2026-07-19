#!/usr/bin/env python3
"""Validate closed, adapter-specific artifacts used by V3 controller gates."""

from __future__ import annotations

import json
import math
import re
from typing import Any


SCHEMA = "cuda-optimizer/gate-evidence-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PRODUCERS = {
    "correctness_reference": "correctness-reference-adapter",
    "dispatch_identity": "dispatch-identity-adapter",
    "target_compile_probe": "compiler-evidence-adapter",
    "candidate_correctness": "candidate-correctness-adapter",
    "paired_measurement": "paired-measurement-adapter",
    "workload_replay": "workload-replay-adapter",
}
_SUBJECT_FIELDS = {
    "correctness_reference": {"reference_sha256"},
    "dispatch_identity": {"target_sha256"},
    "target_compile_probe": {"target_sha256"},
    "candidate_correctness": {"candidate_id", "candidate_sha256"},
    "paired_measurement": {"candidate_id", "candidate_sha256"},
    "workload_replay": {"candidate_id", "candidate_sha256"},
}
_RESULT_FIELDS = {
    "correctness_reference": {"oracle_sha256", "cases_total"},
    "dispatch_identity": {"dispatch_sha256", "cases_total"},
    "target_compile_probe": {"arch", "binary_sha256", "compiler_sha256"},
    "candidate_correctness": {"reference_sha256", "cases_total", "cases_passed"},
    "paired_measurement": {"samples_sha256", "pairs_total", "decision"},
    "workload_replay": {
        "workload_sha256",
        "constraints_passed",
        "objective_gate_passed",
    },
}


class ValidationError(ValueError):
    pass


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate gate evidence key: {key}")
        result[key] = value
    return result


def _invalid_number(token: str):
    raise ValidationError(f"gate evidence number must be finite: {token}")


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


def _identifier(value: Any, label: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a safe identifier")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValidationError(f"{label} must be a positive integer")
    return value


def _validate_subject(kind: str, value: Any) -> dict:
    subject = _closed(value, _SUBJECT_FIELDS[kind], "gate evidence subject")
    for field, item in subject.items():
        if field.endswith("_sha256"):
            _sha(item, f"subject.{field}")
        else:
            _identifier(item, f"subject.{field}")
    return subject


def _validate_result(kind: str, value: Any) -> dict:
    result = _closed(value, _RESULT_FIELDS[kind], "gate evidence result")
    for field, item in result.items():
        if field.endswith("_sha256"):
            _sha(item, f"result.{field}")
    if kind in {"correctness_reference", "dispatch_identity"}:
        _positive_int(result["cases_total"], "result.cases_total")
    elif kind == "target_compile_probe":
        if type(result["arch"]) is not str or re.fullmatch(r"sm_[0-9]+", result["arch"]) is None:
            raise ValidationError("result.arch must be an exact SM architecture")
    elif kind == "candidate_correctness":
        total = _positive_int(result["cases_total"], "result.cases_total")
        passed = _positive_int(result["cases_passed"], "result.cases_passed")
        if passed != total:
            raise ValidationError("candidate correctness cases do not support PASS")
    elif kind == "paired_measurement":
        _positive_int(result["pairs_total"], "result.pairs_total")
        if result["decision"] != "PASS":
            raise ValidationError("paired measurement decision does not support PASS")
    elif kind == "workload_replay":
        if result["constraints_passed"] is not True:
            raise ValidationError("workload replay constraints do not support PASS")
        if result["objective_gate_passed"] is not True:
            raise ValidationError("workload replay objective does not support PASS")
    return result


def validate_gate_evidence(
    raw: bytes,
    *,
    expected_contract_sha256: str,
    expected_environment_sha256: str,
) -> dict:
    """Parse one gate artifact and derive its trusted routing metadata."""
    if not isinstance(raw, bytes):
        raise ValidationError("gate evidence must be bytes")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_invalid_number,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError("gate evidence artifact must be strict JSON") from exc
    fields = {
        "schema_version",
        "kind",
        "producer",
        "contract_sha256",
        "environment_sha256",
        "recorded_at",
        "status",
        "subject",
        "result",
    }
    evidence = _closed(value, fields, "gate evidence")
    if evidence["schema_version"] != SCHEMA:
        raise ValidationError("unsupported gate evidence schema")
    kind = evidence["kind"]
    if kind not in _PRODUCERS:
        raise ValidationError(f"unsupported gate evidence kind: {kind}")
    producer = _closed(evidence["producer"], {"id", "version"}, "gate evidence producer")
    if producer != {"id": _PRODUCERS[kind], "version": "1.0.0"}:
        raise ValidationError(f"untrusted producer for {kind}")
    if _sha(evidence["contract_sha256"], "contract_sha256") != _sha(
        expected_contract_sha256, "expected_contract_sha256"
    ):
        raise ValidationError("gate evidence contract identity mismatch")
    if _sha(evidence["environment_sha256"], "environment_sha256") != _sha(
        expected_environment_sha256, "expected_environment_sha256"
    ):
        raise ValidationError("gate evidence environment identity mismatch")
    recorded_at = evidence["recorded_at"]
    if type(recorded_at) not in {int, float} or not math.isfinite(recorded_at) or recorded_at < 0:
        raise ValidationError("gate evidence recorded_at must be non-negative and finite")
    if evidence["status"] != "PASS":
        raise ValidationError("gate evidence status does not support PASS")
    subject = _validate_subject(kind, evidence["subject"])
    result = _validate_result(kind, evidence["result"])
    return {
        "kind": kind,
        "layer": "workload" if kind == "workload_replay" else "kernel",
        "summary": f"Validated {kind} evidence from {_PRODUCERS[kind]}.",
        "signals": [],
        "producer": dict(producer),
        "recorded_at": float(recorded_at),
        "subject": dict(subject),
        "result": dict(result),
    }
