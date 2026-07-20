#!/usr/bin/env python3
"""Seal and verify Controller-owned Planner admission records."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections.abc import Mapping
from typing import Any


SCHEMA = "cuda-optimizer/planner-admission-v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = {
    "schema_version",
    "status",
    "run_id",
    "ledger_id",
    "contract_sha256",
    "environment_sha256",
    "candidate_id",
    "mechanism_id",
    "observation_id",
    "observation_summary_sha256",
    "capability_query_sha256",
    "capability_ids",
    "admitted_at",
    "evidence_age_seconds",
    "pre_execution",
    "admission_sha256",
    "controller_attestation",
}
_LEGACY_FIELDS = _FIELDS - {"mechanism_id"}


class ValidationError(ValueError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("admission must contain finite JSON values") from exc


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be lowercase SHA-256")
    return value


def _identifier(value: Any, label: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a safe identifier")
    return value


def _key(value: Any) -> bytes:
    if not isinstance(value, bytes) or len(value) < 32:
        raise ValidationError("controller_seal_key must contain at least 32 bytes")
    return value


def _time(value: Any, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        raise ValidationError(f"{label} must be non-negative and finite")
    return float(value)


def _validate_pre_execution(value: Any) -> dict:
    if type(value) is not dict or set(value) != {"satisfied", "missing_gates", "gates"}:
        raise ValidationError("pre_execution must be a closed gate resolution")
    if value["satisfied"] is not True or value["missing_gates"] != []:
        raise ValidationError("pre_execution gates must all be satisfied")
    gates = value["gates"]
    if type(gates) is not list or len(gates) != 3:
        raise ValidationError("pre_execution must contain exactly three gates")
    expected = {"correctness_reference", "dispatch_identity", "target_compile_probe"}
    actual = set()
    for gate in gates:
        if type(gate) is not dict or set(gate) != {
            "gate", "satisfied", "observation_ids", "artifact_sha256s"
        }:
            raise ValidationError("pre_execution gate entry must be closed")
        actual.add(gate["gate"])
        if gate["satisfied"] is not True:
            raise ValidationError("pre_execution gate must be satisfied")
        observations = gate["observation_ids"]
        artifacts = gate["artifact_sha256s"]
        if (
            type(observations) is not list
            or not observations
            or len(observations) != len(set(observations))
            or type(artifacts) is not list
            or not artifacts
            or len(artifacts) != len(set(artifacts))
        ):
            raise ValidationError("pre_execution gate evidence must be non-empty and unique")
        for item in observations:
            _identifier(item, "pre_execution.observation_id")
        for item in artifacts:
            _sha(item, "pre_execution.artifact_sha256")
    if actual != expected:
        raise ValidationError("pre_execution gate set is incomplete")
    return json.loads(json.dumps(value))


def seal_admission(material: Mapping[str, Any], *, controller_seal_key: bytes) -> dict:
    if type(material) is not dict:
        raise ValidationError("admission material must be an object")
    expected = _FIELDS - {"admission_sha256", "controller_attestation"}
    if set(material) != expected:
        raise ValidationError("admission material fields are incomplete or unknown")
    unsigned = json.loads(_canonical_bytes(material))
    unsigned["admission_sha256"] = hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    unsigned["controller_attestation"] = hmac.new(
        _key(controller_seal_key), _canonical_bytes(unsigned), hashlib.sha256
    ).hexdigest()
    return validate_admission(unsigned, controller_seal_key=controller_seal_key)


def validate_admission(
    value: Mapping[str, Any],
    *,
    controller_seal_key: bytes,
    proposal: Mapping[str, Any] | None = None,
    expected_contract_sha256: str | None = None,
    expected_admitted_at: float | None = None,
) -> dict:
    fields = set(value) if type(value) is dict else set()
    if type(value) is not dict or (
        fields != _FIELDS and fields != _LEGACY_FIELDS
    ):
        raise ValidationError("planner admission must be a closed object")
    clean = json.loads(_canonical_bytes(value))
    if clean["schema_version"] != SCHEMA or clean["status"] != "ADMITTED":
        raise ValidationError("unsupported planner admission")
    for field in ("run_id", "ledger_id", "candidate_id", "observation_id"):
        _identifier(clean[field], field)
    if "mechanism_id" in clean:
        _identifier(clean["mechanism_id"], "mechanism_id")
    for field in (
        "contract_sha256", "environment_sha256", "observation_summary_sha256",
        "capability_query_sha256", "admission_sha256", "controller_attestation"
    ):
        _sha(clean[field], field)
    if type(clean["capability_ids"]) is not list or not clean["capability_ids"]:
        raise ValidationError("admission capability_ids must be non-empty")
    if len(clean["capability_ids"]) != len(set(clean["capability_ids"])):
        raise ValidationError("admission capability_ids must be unique")
    for item in clean["capability_ids"]:
        _identifier(item, "capability_id")
    admitted_at = _time(clean["admitted_at"], "admitted_at")
    _time(clean["evidence_age_seconds"], "evidence_age_seconds")
    _validate_pre_execution(clean["pre_execution"])
    attestation = clean.pop("controller_attestation")
    expected_attestation = hmac.new(
        _key(controller_seal_key), _canonical_bytes(clean), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(attestation, expected_attestation):
        raise ValidationError("controller admission attestation changed")
    digest = clean.pop("admission_sha256")
    if hashlib.sha256(_canonical_bytes(clean)).hexdigest() != digest:
        raise ValidationError("planner admission digest changed")
    clean["admission_sha256"] = digest
    clean["controller_attestation"] = attestation
    if expected_contract_sha256 is not None and clean["contract_sha256"] != _sha(
        expected_contract_sha256, "expected_contract_sha256"
    ):
        raise ValidationError("planner admission contract does not match")
    if expected_admitted_at is not None and admitted_at != _time(
        expected_admitted_at, "expected_admitted_at"
    ):
        raise ValidationError("planner admission time does not match registration time")
    if proposal is not None:
        bindings = {
            "candidate_id": "candidate_id",
            "observation_id": "observation_id",
            "observation_summary_sha256": "observation_summary_sha256",
            "capability_query_sha256": "capability_query_sha256",
            "capability_ids": "capability_ids",
        }
        if "mechanism_id" in clean:
            bindings["mechanism_id"] = "mechanism_id"
        for admission_field, proposal_field in bindings.items():
            if clean[admission_field] != proposal.get(proposal_field):
                raise ValidationError(
                    f"planner admission {admission_field} does not match proposal"
                )
    return clean
