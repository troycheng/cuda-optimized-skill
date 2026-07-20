#!/usr/bin/env python3
"""Validate Controller-owned V3.1 analysis epochs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any


EPOCH_SCHEMA = "cuda-optimizer/analysis-epoch-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_TRIGGERS = {
    "initial",
    "mechanism_change",
    "workload_regime_change",
    "bottleneck_migration",
    "identity_change",
    "conservative_boundary",
}
_PROFILERS = {"nsys", "pytorch", "perfetto", "custom"}
_IDENTITY_FIELDS = {
    "workload_contract_sha256",
    "environment_sha256",
    "source_sha256",
    "analysis_policy_sha256",
}


class ValidationError(ValueError):
    """Raised when an epoch is not closed, current, and reproducible."""


def _closed(value: Any, fields: set[str], label: str) -> dict:
    if type(value) is not dict:
        raise ValidationError(f"{label} must be an object")
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing:
        raise ValidationError(f"{label} is missing fields: {sorted(missing)}")
    if unknown:
        raise ValidationError(f"{label} contains unknown fields: {sorted(unknown)}")
    return value


def _text(value: Any, label: str, *, maximum: int = 256) -> str:
    if type(value) is not str or not value.strip():
        raise ValidationError(f"{label} must be a non-empty string")
    if len(value) > maximum:
        raise ValidationError(f"{label} exceeds {maximum} characters")
    return value


def _identifier(value: Any, label: str) -> str:
    text = _text(value, label, maximum=128)
    if _IDENTIFIER.fullmatch(text) is None:
        raise ValidationError(f"{label} must be a safe identifier")
    return text


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be lowercase SHA-256")
    return value


def _identities(value: Any, label: str = "epoch.identities") -> dict:
    identities = _closed(value, _IDENTITY_FIELDS, label)
    return {field: _sha(identities[field], f"{label}.{field}") for field in sorted(_IDENTITY_FIELDS)}


def validate_epoch(
    value: Mapping[str, Any], *, expected_identities: Mapping[str, Any] | None = None
) -> dict:
    """Return a detached epoch after enforcing Controller lineage and identities."""
    epoch = _closed(
        value,
        {
            "schema_version",
            "epoch_id",
            "sequence",
            "trigger",
            "parent_epoch_id",
            "started_at",
            "identities",
            "source",
            "regime",
            "boundary_ambiguous",
        },
        "epoch",
    )
    if epoch["schema_version"] != EPOCH_SCHEMA:
        raise ValidationError(f"epoch.schema_version must be {EPOCH_SCHEMA}")
    _identifier(epoch["epoch_id"], "epoch.epoch_id")
    sequence = epoch["sequence"]
    if type(sequence) is not int or sequence < 1:
        raise ValidationError("epoch.sequence must be a positive integer")
    trigger = epoch["trigger"]
    if trigger not in _TRIGGERS:
        raise ValidationError("epoch.trigger is unsupported")
    parent = epoch["parent_epoch_id"]
    if sequence == 1 or trigger == "initial":
        if sequence != 1 or trigger != "initial" or parent is not None:
            raise ValidationError("initial epoch must be sequence 1 without a parent")
    else:
        _identifier(parent, "epoch.parent_epoch_id")
        if parent == epoch["epoch_id"]:
            raise ValidationError("epoch parent must differ from epoch_id")
    started_at = epoch["started_at"]
    if type(started_at) not in {int, float} or not math.isfinite(float(started_at)):
        raise ValidationError("epoch.started_at must be finite")
    if started_at < 0:
        raise ValidationError("epoch.started_at must be non-negative")

    identities = _identities(epoch["identities"])
    if expected_identities is not None:
        expected = _identities(dict(expected_identities), "expected_identities")
        for field in sorted(_IDENTITY_FIELDS):
            if identities[field] != expected[field]:
                label = field.removesuffix("_sha256")
                raise ValidationError(f"epoch {label} identity does not match Controller")

    source = _closed(
        epoch["source"],
        {
            "profiler",
            "profiler_version",
            "export_schema",
            "adapter_id",
            "adapter_version",
            "adapter_sha256",
        },
        "epoch.source",
    )
    if source["profiler"] not in _PROFILERS:
        raise ValidationError("epoch.source.profiler is unsupported")
    for field in ("profiler_version", "export_schema", "adapter_version"):
        _text(source[field], f"epoch.source.{field}")
    _identifier(source["adapter_id"], "epoch.source.adapter_id")
    _sha(source["adapter_sha256"], "epoch.source.adapter_sha256")

    regime = _closed(
        epoch["regime"],
        {
            "shape_distribution_sha256",
            "dynamic_branch_sha256",
            "execution_regime_sha256",
        },
        "epoch.regime",
    )
    for field in regime:
        _sha(regime[field], f"epoch.regime.{field}")
    if type(epoch["boundary_ambiguous"]) is not bool:
        raise ValidationError("epoch.boundary_ambiguous must be a boolean")
    if trigger == "conservative_boundary" and not epoch["boundary_ambiguous"]:
        raise ValidationError("conservative boundary must remain ambiguous")
    return copy.deepcopy(dict(epoch))


def epoch_digest(value: Mapping[str, Any]) -> str:
    """Return the canonical digest of a validated epoch."""
    normalized = validate_epoch(value)
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
