#!/usr/bin/env python3
"""Validate Controller-derived PyTorch and Nsys diagnostic observations."""

from __future__ import annotations

import json
import math
import re
from typing import Any


EVIDENCE_SCHEMA = "cuda-optimizer/diagnostic-evidence-v1"
MEASUREMENT_SCHEMA = "cuda-optimizer/diagnostic-measurement-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PRODUCERS = {
    "nsys_timeline": "nsys-timeline-adapter",
    "pytorch_profile": "pytorch-profile-adapter",
}
_SIGNALS = {
    "nsys_timeline": {
        "launch_gap_short_context",
        "gpu_idle_gap",
        "cpu_launch_overhead",
    },
    "pytorch_profile": {
        "gqa_head_ratio",
        "shape_fragmentation",
        "framework_dispatch_overhead",
    },
}


class ValidationError(ValueError):
    pass


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValidationError(f"duplicate diagnostic key: {key}")
        value[key] = item
    return value


def _invalid_number(token: str):
    raise ValidationError(f"diagnostic number must be finite: {token}")


def _strict_json(raw: bytes, label: str) -> dict:
    if not isinstance(raw, bytes):
        raise ValidationError(f"{label} must be bytes")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_invalid_number,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{label} must be strict JSON") from exc
    if type(value) is not dict:
        raise ValidationError(f"{label} must be an object")
    return value


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


def _signals(kind: str, value: Any) -> list[str]:
    if type(value) is not list:
        raise ValidationError("diagnostic signals must be an array")
    result = [_identifier(item, "diagnostic signal") for item in value]
    if len(result) != len(set(result)):
        raise ValidationError("diagnostic signals must not contain duplicates")
    unknown = set(result) - _SIGNALS[kind]
    if unknown:
        raise ValidationError(f"unsupported signals for {kind}: {sorted(unknown)}")
    return sorted(result)


def _subject(value: Any) -> dict:
    subject = _closed(value, {"target_sha256"}, "diagnostic subject")
    _sha(subject["target_sha256"], "subject.target_sha256")
    return dict(subject)


def _report(value: Any) -> dict:
    report = _closed(
        value, {"artifact_sha256", "events_total"}, "diagnostic report"
    )
    _sha(report["artifact_sha256"], "report.artifact_sha256")
    if type(report["events_total"]) is not int or report["events_total"] < 1:
        raise ValidationError("report.events_total must be a positive integer")
    return dict(report)


def _checks(value: Any) -> None:
    if type(value) is not list or not value:
        raise ValidationError("diagnostic checks must be a non-empty array")
    names = set()
    for index, raw in enumerate(value):
        check = _closed(raw, {"name", "passed"}, f"diagnostic check {index}")
        name = _identifier(check["name"], f"diagnostic check {index}.name")
        if name in names:
            raise ValidationError("diagnostic check names must be unique")
        names.add(name)
        if check["passed"] is not True:
            raise ValidationError("diagnostic checks do not support a usable observation")


def derive_diagnostic_evidence(
    raw_measurement: bytes,
    *,
    kind: str,
    producer_id: str,
    producer_version: str,
    implementation_sha256: str,
    adapter_request_sha256: str,
    contract_sha256: str,
    environment_sha256: str,
    recorded_at: float,
) -> bytes:
    if kind not in _PRODUCERS:
        raise ValidationError(f"unsupported diagnostic kind: {kind}")
    if producer_id != _PRODUCERS[kind] or producer_version != "1.0.0":
        raise ValidationError(f"untrusted producer for {kind}")
    measurement = _strict_json(raw_measurement, "diagnostic measurement")
    _closed(
        measurement,
        {"schema_version", "subject", "report", "signals", "checks"},
        "diagnostic measurement",
    )
    if measurement["schema_version"] != MEASUREMENT_SCHEMA:
        raise ValidationError("unsupported diagnostic measurement schema")
    _checks(measurement["checks"])
    if type(recorded_at) not in {int, float} or not math.isfinite(recorded_at) or recorded_at < 0:
        raise ValidationError("controller recorded_at must be non-negative and finite")
    evidence = {
        "schema_version": EVIDENCE_SCHEMA,
        "kind": kind,
        "producer": {
            "id": producer_id,
            "version": producer_version,
            "implementation_sha256": _sha(
                implementation_sha256, "producer.implementation_sha256"
            ),
        },
        "adapter_request_sha256": _sha(
            adapter_request_sha256, "adapter_request_sha256"
        ),
        "contract_sha256": _sha(contract_sha256, "contract_sha256"),
        "environment_sha256": _sha(
            environment_sha256, "environment_sha256"
        ),
        "recorded_at": float(recorded_at),
        "subject": _subject(measurement["subject"]),
        "report": _report(measurement["report"]),
        "signals": _signals(kind, measurement["signals"]),
    }
    return (
        json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def validate_diagnostic_evidence(
    raw: bytes,
    *,
    expected_contract_sha256: str,
    expected_environment_sha256: str,
) -> dict:
    evidence = _strict_json(raw, "diagnostic evidence artifact")
    _closed(
        evidence,
        {
            "schema_version",
            "kind",
            "producer",
            "adapter_request_sha256",
            "contract_sha256",
            "environment_sha256",
            "recorded_at",
            "subject",
            "report",
            "signals",
        },
        "diagnostic evidence",
    )
    if evidence["schema_version"] != EVIDENCE_SCHEMA:
        raise ValidationError("unsupported diagnostic evidence schema")
    kind = evidence["kind"]
    if kind not in _PRODUCERS:
        raise ValidationError(f"unsupported diagnostic kind: {kind}")
    producer = _closed(
        evidence["producer"],
        {"id", "version", "implementation_sha256"},
        "diagnostic producer",
    )
    if producer["id"] != _PRODUCERS[kind] or producer["version"] != "1.0.0":
        raise ValidationError(f"untrusted producer for {kind}")
    _sha(producer["implementation_sha256"], "producer.implementation_sha256")
    request_sha = _sha(evidence["adapter_request_sha256"], "adapter_request_sha256")
    contract = _sha(evidence["contract_sha256"], "contract_sha256")
    environment = _sha(evidence["environment_sha256"], "environment_sha256")
    if contract != _sha(expected_contract_sha256, "expected_contract_sha256"):
        raise ValidationError("diagnostic contract identity mismatch")
    if environment != _sha(
        expected_environment_sha256, "expected_environment_sha256"
    ):
        raise ValidationError("diagnostic environment identity mismatch")
    recorded_at = evidence["recorded_at"]
    if type(recorded_at) not in {int, float} or not math.isfinite(recorded_at) or recorded_at < 0:
        raise ValidationError("diagnostic recorded_at must be non-negative and finite")
    return {
        "kind": kind,
        "layer": "workload",
        "summary": f"Validated {kind} observation from {_PRODUCERS[kind]}.",
        "signals": _signals(kind, evidence["signals"]),
        "producer": dict(producer),
        "adapter_request_sha256": request_sha,
        "recorded_at": float(recorded_at),
        "subject": _subject(evidence["subject"]),
        "result": _report(evidence["report"]),
    }
