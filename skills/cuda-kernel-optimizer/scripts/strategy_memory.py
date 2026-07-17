#!/usr/bin/env python3
"""Workload-scoped, advisory strategy memory primitives."""

from __future__ import annotations

import copy
import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable


_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import artifact_store  # noqa: E402
import decision  # noqa: E402
import orchestrate  # noqa: E402
import state  # noqa: E402
import workload_adapter  # noqa: E402


MEMORY_SCHEMA = "cuda-kernel-optimizer/strategy-memory-v1"
RECORD_SCHEMA = "cuda-kernel-optimizer/strategy-record-v1"
SUGGESTION_SCHEMA = "cuda-kernel-optimizer/strategy-suggestion-v1"
ADVISORY_GUARDRAIL = (
    "Advisory only: cannot delete or prune branches; cannot override profiler "
    "evidence or budget policy; cannot bypass correctness, sanitizer, paired "
    "benchmark, workload, or decision gates; and cannot authorize promotion."
)
MAX_SCOPES = 256
MAX_RUNS_PER_SCOPE = 128
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SCOPE_FIELDS = {
    "manifest_schema_version",
    "input_hash",
    "backend",
    "primary_sm_arch",
    "dims",
    "ptr_size",
    "baseline_sha256",
    "ref_sha256",
    "workload",
}
_RECORD_FIELDS = {
    "input_hash",
    "candidate_sha256",
    "decision_sha256",
    "checkpoint_identity",
}
_VERIFIED_RECORD_FIELDS = {
    "schema_version",
    "input_hash",
    "candidate_sha256",
    "decision_sha256",
    "checkpoint_identity",
    "run_root",
    "scope",
    "completed_at",
    "terminal",
    "bundle",
    "methods",
    "evidence",
}
_TERMINAL_FIELDS = {
    "status", "mode", "reason", "statistics", "workload_status",
    "workload_statistics", "workload_failure", "constraints", "pareto",
}
_EVIDENCE_FIELDS = {"role", "path", "sha256"}


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
        raise ValueError("value must be finite strict JSON") from error


def _strict_json_bytes(payload: bytes, field: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{field} contains duplicate key: {key}")
            result[key] = value
        return result

    def nonfinite(token):
        raise ValueError(f"{field} contains non-finite JSON constant: {token}")

    try:
        text = payload.decode("utf-8")
        return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)
    except UnicodeDecodeError as error:
        raise ValueError(f"{field} is not UTF-8 JSON") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} is malformed JSON: {error}") from error


def _strict_copy(value: Any, field: str) -> Any:
    return _strict_json_bytes(_canonical_bytes(value), field)


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be 64 lowercase hexadecimal characters")
    return value


def _exact_fields(value: Any, fields: set[str], field: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing:
        raise ValueError(f"{field} is missing field: {missing[0]}")
    if unknown:
        raise ValueError(f"{field} contains unknown field: {unknown[0]}")
    return value


def _validate_scope_document(value: Any) -> dict:
    scope = _strict_copy(value, "scope")
    _exact_fields(scope, _SCOPE_FIELDS, "scope")
    if type(scope["manifest_schema_version"]) is not int or scope[
        "manifest_schema_version"
    ] != artifact_store.CURRENT_SCHEMA_VERSION:
        raise ValueError("scope.manifest_schema_version is invalid")
    _sha(scope["input_hash"], "scope.input_hash")
    for field in ("backend", "primary_sm_arch"):
        if not isinstance(scope[field], str) or not scope[field].strip():
            raise ValueError(f"scope.{field} must be a non-empty string")
    dims = scope["dims"]
    if not isinstance(dims, dict):
        raise ValueError("scope.dims must be a JSON object")
    for name, size in dims.items():
        if not isinstance(name, str) or not name:
            raise ValueError("scope.dims keys must be non-empty strings")
        if type(size) is not int or size <= 0:
            raise ValueError(f"scope.dims.{name} must be a positive integer")
    if type(scope["ptr_size"]) is not int or scope["ptr_size"] <= 0:
        raise ValueError("scope.ptr_size must be a positive integer")
    _sha(scope["baseline_sha256"], "scope.baseline_sha256")
    _sha(scope["ref_sha256"], "scope.ref_sha256")
    workload = scope["workload"]
    if not isinstance(workload, dict):
        raise ValueError("scope.workload must be a JSON object")
    mode = workload.get("mode")
    if mode == "kernel-only":
        _exact_fields(workload, {"mode"}, "scope.workload")
    elif mode == "full":
        _exact_fields(
            workload,
            {"mode", "source", "source_hash", "objective", "cases", "kind"},
            "scope.workload",
        )
        _sha(workload["source_hash"], "scope.workload.source_hash")
        if workload["kind"] == "python":
            if not isinstance(workload["source"], str) or not os.path.isabs(
                workload["source"]
            ):
                raise ValueError("scope.workload.source must be an absolute path")
        elif workload["kind"] == "command":
            source = workload["source"]
            if (
                not isinstance(source, list)
                or not source
                or any(not isinstance(item, str) or not item for item in source)
            ):
                raise ValueError("scope command workload.source must be an argv list")
        else:
            raise ValueError("scope.workload.kind must be python or command")
        if not isinstance(workload["objective"], Mapping):
            raise ValueError("scope.workload.objective must be a JSON object")
        normalized_objective = workload_adapter.validate_objective(
            workload["objective"]
        )
        if normalized_objective != workload["objective"]:
            raise ValueError("scope.workload.objective is not normalized")
        if not isinstance(workload["cases"], list) or any(
            not isinstance(case, dict) for case in workload["cases"]
        ):
            raise ValueError("scope.workload.cases must be an array of objects")
        if not isinstance(workload["kind"], str) or not workload["kind"].strip():
            raise ValueError("scope.workload.kind must be a non-empty string")
    else:
        raise ValueError("scope.workload.mode must be full or kernel-only")
    return scope


def _scope_key_from_document(scope: Mapping) -> str:
    clean = _validate_scope_document(scope)
    return hashlib.sha256(_canonical_bytes(clean)).hexdigest()


def _input_identity(
    manifest: Mapping, name: str, *, verify_content: bool = True
) -> tuple[str, int]:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {"baseline", "ref"}:
        raise ValueError("manifest.inputs must contain exactly baseline and ref")
    record = _exact_fields(
        inputs[name], {"path", "sha256", "size_bytes"}, f"manifest.inputs.{name}"
    )
    path = record["path"]
    if not isinstance(path, str) or not path or not os.path.isabs(path):
        raise ValueError(f"manifest.inputs.{name}.path must be absolute")
    expected = _sha(record["sha256"], f"manifest.inputs.{name}.sha256")
    if type(record["size_bytes"]) is not int or record["size_bytes"] < 0:
        raise ValueError(f"manifest.inputs.{name}.size_bytes is invalid")
    if not verify_content:
        return expected, record["size_bytes"]
    current = artifact_store.read_regular_bytes(path)
    actual = hashlib.sha256(current).hexdigest()
    if actual != expected:
        raise ValueError(f"manifest.inputs.{name}.sha256 does not match current content")
    if len(current) != record["size_bytes"]:
        raise ValueError(f"manifest.inputs.{name}.size_bytes does not match current content")
    return actual, len(current)


def _validated_manifest_scope(
    manifest_path: str | os.PathLike, *, verify_files: bool
) -> tuple[dict, dict]:
    """Load a frozen manifest and derive its complete strategy scope."""
    raw = artifact_store.read_regular_bytes(manifest_path)
    manifest = _strict_json_bytes(raw, "manifest")
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    required = {
        "schema_version",
        "input_hash",
        "inputs",
        "environment",
        "backend",
        "dims",
        "ptr_size",
        "mode",
        "workload",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"manifest is missing field: {missing[0]}")
    if type(manifest["schema_version"]) is not int or manifest[
        "schema_version"
    ] != artifact_store.CURRENT_SCHEMA_VERSION:
        raise ValueError("manifest.schema_version is invalid")
    input_hash = _sha(manifest["input_hash"], "manifest.input_hash")
    environment = manifest["environment"]
    if not isinstance(environment, Mapping):
        raise ValueError("manifest.environment must be a JSON object")
    arch = environment.get("primary_sm_arch")
    if not isinstance(arch, str) or not arch.strip():
        raise ValueError("manifest.environment.primary_sm_arch is required")
    backend = manifest["backend"]
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("manifest.backend must be a non-empty string")
    dims = manifest["dims"]
    ptr_size = manifest["ptr_size"]
    baseline_sha, _ = _input_identity(
        manifest, "baseline", verify_content=verify_files
    )
    ref_sha, _ = _input_identity(manifest, "ref", verify_content=verify_files)
    mode = manifest["mode"]
    raw_workload = manifest["workload"]
    if mode == "kernel-only":
        if raw_workload is not None:
            raise ValueError("kernel-only manifest.workload must be null")
        workload = {"mode": "kernel-only"}
    elif mode == "full":
        workload_fields = {"source", "source_hash", "objective", "cases", "kind"}
        raw_workload = _exact_fields(raw_workload, workload_fields, "manifest.workload")
        source_hash = _sha(
            raw_workload["source_hash"], "manifest.workload.source_hash"
        )
        objective = workload_adapter.validate_objective(raw_workload["objective"])
        if objective != raw_workload["objective"]:
            raise ValueError("manifest.workload.objective is not normalized")
        cases = raw_workload["cases"]
        if not isinstance(cases, list) or any(
            not isinstance(case, dict) for case in cases
        ):
            raise ValueError("manifest.workload.cases must be an array of objects")
        source = _strict_copy(raw_workload["source"], "manifest.workload.source")
        spec = workload_adapter.WorkloadSpec(
            kind=raw_workload["kind"],
            source=source,
            objective=objective,
            cases=tuple(_strict_copy(cases, "manifest.workload.cases")),
            source_hash=source_hash,
        )
        if verify_files:
            if spec.kind == "python" and isinstance(spec.source, str):
                artifact_store.read_regular_bytes(spec.source)
            elif spec.kind == "command" and isinstance(spec.source, list) and spec.source:
                artifact_store.read_regular_bytes(spec.source[0])
            workload_adapter.verify_frozen_spec(spec)
        workload = {"mode": "full", **dict(raw_workload)}
    else:
        raise ValueError("manifest.mode must be full or kernel-only")
    scope = {
        "manifest_schema_version": manifest["schema_version"],
        "input_hash": input_hash,
        "backend": backend,
        "primary_sm_arch": arch,
        "dims": dims,
        "ptr_size": ptr_size,
        "baseline_sha256": baseline_sha,
        "ref_sha256": ref_sha,
        "workload": workload,
    }
    return manifest, _validate_scope_document(scope)


def _scope_document_from_manifest(
    manifest_path: str | os.PathLike, *, verify_files: bool
) -> dict:
    return _validated_manifest_scope(
        manifest_path, verify_files=verify_files
    )[1]


def scope_document(manifest_path: str | os.PathLike) -> dict:
    """Load a frozen manifest and verify the files bound into its scope."""
    return _scope_document_from_manifest(manifest_path, verify_files=True)


def scope_key(manifest_or_scope: str | os.PathLike | Mapping) -> str:
    """Return the canonical SHA-256 key for a manifest path or scope document."""
    if isinstance(manifest_or_scope, Mapping):
        scope = manifest_or_scope
    else:
        scope = scope_document(manifest_or_scope)
    return _scope_key_from_document(scope)


def _validate_record(value: Any) -> dict:
    record = _strict_copy(value, "strategy run")
    _exact_fields(record, _VERIFIED_RECORD_FIELDS, "strategy run")
    if record["schema_version"] != RECORD_SCHEMA:
        raise ValueError(f"strategy run.schema_version must be {RECORD_SCHEMA}")
    for field in sorted(_RECORD_FIELDS):
        _sha(record[field], f"strategy run.{field}")
    clean_scope = _validate_scope_document(record["scope"])
    if clean_scope["input_hash"] != record["input_hash"]:
        raise ValueError("strategy run.scope does not match input_hash")
    if isinstance(record["completed_at"], bool) or not isinstance(
        record["completed_at"], (int, float)
    ) or not math.isfinite(float(record["completed_at"])):
        raise ValueError("strategy run.completed_at must be finite")
    run_root = _exact_fields(
        record["run_root"], {"path", "device", "inode"}, "strategy run.run_root"
    )
    if not isinstance(run_root["path"], str) or not os.path.isabs(run_root["path"]):
        raise ValueError("strategy run.run_root.path must be absolute")
    if run_root["path"] != os.path.abspath(run_root["path"]):
        raise ValueError("strategy run.run_root.path must be normalized")
    for field in ("device", "inode"):
        if type(run_root[field]) is not int or run_root[field] < 0:
            raise ValueError(f"strategy run.run_root.{field} must be non-negative")

    def evidence_identity(value: Any, field: str) -> dict:
        clean = _exact_fields(value, _EVIDENCE_FIELDS, field)
        if not isinstance(clean["role"], str) or not clean["role"]:
            raise ValueError(f"{field}.role must be non-empty")
        if not isinstance(clean["path"], str) or not os.path.isabs(clean["path"]):
            raise ValueError(f"{field}.path must be absolute")
        if clean["path"] != os.path.abspath(clean["path"]):
            raise ValueError(f"{field}.path must be normalized")
        try:
            Path(clean["path"]).absolute().relative_to(Path(run_root["path"]))
        except ValueError as error:
            raise ValueError(f"{field}.path escapes run_root") from error
        _sha(clean["sha256"], f"{field}.sha256")
        return dict(clean)

    raw_evidence = record["evidence"]
    if not isinstance(raw_evidence, list):
        raise ValueError("strategy run.evidence must be a JSON array")
    evidence = [
        evidence_identity(item, f"strategy run.evidence[{index}]")
        for index, item in enumerate(raw_evidence)
    ]
    evidence_keys = {
        _canonical_bytes(item) for item in evidence
    }
    if len(evidence_keys) != len(evidence):
        raise ValueError("strategy run.evidence contains duplicates")

    terminal = _exact_fields(record["terminal"], _TERMINAL_FIELDS, "strategy run.terminal")
    if terminal["status"] not in decision.TERMINAL_STATUSES:
        raise ValueError("strategy run.terminal.status is invalid")
    if terminal["mode"] not in {"kernel-only", "full"}:
        raise ValueError("strategy run.terminal.mode is invalid")
    if not isinstance(terminal["reason"], str) or not terminal["reason"]:
        raise ValueError("strategy run.terminal.reason must be non-empty")
    for field in ("statistics", "workload_statistics", "workload_failure", "pareto"):
        if terminal[field] is not None and not isinstance(terminal[field], dict):
            raise ValueError(f"strategy run.terminal.{field} must be an object or null")
    if terminal["workload_status"] not in {None, "evaluated", "workload_failed"}:
        raise ValueError("strategy run.terminal.workload_status is invalid")
    if not isinstance(terminal["constraints"], list) or any(
        not isinstance(item, dict) for item in terminal["constraints"]
    ):
        raise ValueError("strategy run.terminal.constraints must be an array of objects")
    normalized_constraints = decision._validate_constraints(terminal["constraints"])
    if normalized_constraints != terminal["constraints"]:
        raise ValueError("strategy run terminal constraints are not normalized")
    normalized_pareto = decision._validate_pareto(terminal["pareto"])
    if normalized_pareto != terminal["pareto"]:
        raise ValueError("strategy run terminal pareto is not normalized")
    for field in ("statistics", "workload_statistics"):
        statistics = terminal[field]
        if statistics is not None:
            normalized = decision.validate_paired_statistics(
                statistics, f"strategy run.terminal.{field}"
            )
            if normalized != statistics:
                raise ValueError(f"strategy run terminal {field} is not normalized")
    expected_statistic_status = {
        "kernel_only_win": "confirmed_win",
        "end_to_end_win": "confirmed_win",
        "confirmed_loss": "confirmed_loss",
        "inconclusive": "inconclusive",
    }.get(terminal["status"])
    if expected_statistic_status is not None:
        if not isinstance(terminal["statistics"], Mapping) or terminal["statistics"].get(
            "status"
        ) != expected_statistic_status:
            raise ValueError("strategy run terminal statistics contradict outcome")
    if terminal["workload_status"] == "evaluated":
        if not isinstance(terminal["workload_statistics"], Mapping):
            raise ValueError("evaluated strategy workload requires statistics")
        if terminal["workload_failure"] is not None:
            raise ValueError("evaluated strategy workload cannot contain failure")
    elif terminal["workload_status"] == "workload_failed":
        if terminal["workload_statistics"] is not None:
            raise ValueError("failed strategy workload cannot contain statistics")
        failure = decision._validate_workload(terminal["workload_failure"])
        if failure != terminal["workload_failure"]:
            raise ValueError("strategy workload failure is not normalized")
    elif terminal["workload_statistics"] is not None or terminal["workload_failure"] is not None:
        raise ValueError("strategy workload evidence requires workload_status")

    bundle = _exact_fields(
        record["bundle"],
        {"outcome", "method_ids", "promotion_authority", "decision_evidence"},
        "strategy run.bundle",
    )
    if bundle["outcome"] != terminal["status"]:
        raise ValueError("strategy run bundle outcome does not match terminal status")
    method_ids = bundle["method_ids"]
    if (
        not isinstance(method_ids, list)
        or method_ids != sorted(set(method_ids))
        or any(not isinstance(item, str) or not item for item in method_ids)
    ):
        raise ValueError("strategy run bundle method_ids must be sorted unique strings")
    if bundle["promotion_authority"] is not False:
        raise ValueError("strategy run bundle promotion_authority must be false")
    allowed_roles = {
        "state", "checkpoint", "decision", "methods", "candidate",
        "kernel_paired_samples", "workload_paired_samples", "sass_check",
        "attribution", "champion_bench",
        *{f"ablation_kernel:{method_id}" for method_id in method_ids},
        *{f"ablation_bench:{method_id}" for method_id in method_ids},
    }
    if any(item["role"] not in allowed_roles for item in evidence):
        raise ValueError("strategy run evidence contains an unknown role")
    decision_evidence = evidence_identity(
        bundle["decision_evidence"], "strategy run.bundle.decision_evidence"
    )
    if decision_evidence["role"] != "decision" or decision_evidence["sha256"] != record["decision_sha256"]:
        raise ValueError("strategy run bundle decision evidence is inconsistent")
    if _canonical_bytes(decision_evidence) not in evidence_keys:
        raise ValueError("strategy run bundle decision evidence is absent from evidence")

    methods = _exact_fields(
        record["methods"], {"performance", "implementation"}, "strategy run.methods"
    )
    for field in ("performance", "implementation"):
        if not isinstance(methods[field], dict):
            raise ValueError(f"strategy run methods.{field} must be an object")
        if not set(methods[field]).issubset(method_ids):
            raise ValueError(f"strategy run methods.{field} contains a non-bundle method")
    for method_id, item in methods["performance"].items():
        item = _exact_fields(
            item,
            {"outcome", "champion_ms", "ablated_ms", "attribution_ms", "attribution_pct", "evidence_quality", "promotion_authority", "evidence"},
            f"strategy run methods.performance.{method_id}",
        )
        if item["outcome"] not in {"positive", "negative"}:
            raise ValueError("strategy method performance outcome is invalid")
        numbers = []
        for field in ("champion_ms", "ablated_ms", "attribution_ms", "attribution_pct"):
            value = item[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"strategy method performance {field} must be finite")
            numbers.append(float(value))
        champion_ms, ablated_ms, attribution_ms, attribution_pct = numbers
        if champion_ms <= 0.0 or ablated_ms <= 0.0:
            raise ValueError("strategy method performance timings must be positive")
        if not math.isclose(ablated_ms - champion_ms, attribution_ms, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("strategy method performance attribution_ms is inconsistent")
        if not math.isclose(attribution_ms / champion_ms * 100.0, attribution_pct, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("strategy method performance attribution_pct is inconsistent")
        if attribution_ms == 0.0 or (attribution_ms > 0) != (item["outcome"] == "positive"):
            raise ValueError("strategy method performance outcome contradicts attribution")
        if attribution_pct == 0.0 or (attribution_pct > 0) != (item["outcome"] == "positive"):
            raise ValueError("strategy method performance percent contradicts outcome")
        if item["evidence_quality"] != "diagnostic_unpaired_ablation":
            raise ValueError("strategy method performance evidence_quality is invalid")
        if item["promotion_authority"] is not False:
            raise ValueError("strategy method performance cannot authorize promotion")
        if not isinstance(item["evidence"], list):
            raise ValueError("strategy method performance evidence must be an array")
        method_evidence = [
            evidence_identity(value, f"strategy method {method_id} evidence")
            for value in item["evidence"]
        ]
        required_roles = {
            "attribution", "champion_bench", f"ablation_kernel:{method_id}",
            f"ablation_bench:{method_id}",
        }
        if {value["role"] for value in method_evidence} != required_roles:
            raise ValueError("strategy method performance evidence roles are incomplete")
        if any(_canonical_bytes(value) not in evidence_keys for value in method_evidence):
            raise ValueError("strategy method performance evidence is absent from run evidence")
    for method_id, item in methods["implementation"].items():
        item = _exact_fields(
            item, {"status", "evidence"},
            f"strategy run methods.implementation.{method_id}",
        )
        if item["status"] not in {"passed", "failed", "unavailable", "not_applicable"}:
            raise ValueError("strategy method implementation status is invalid")
        identity = evidence_identity(item["evidence"], f"strategy implementation {method_id}")
        if identity["role"] != "sass_check" or _canonical_bytes(identity) not in evidence_keys:
            raise ValueError("strategy method implementation evidence is inconsistent")
    by_role: dict[str, list[dict]] = {}
    for item in evidence:
        by_role.setdefault(item["role"], []).append(item)
    for required in (
        "state", "checkpoint", "decision", "methods", "candidate",
        "kernel_paired_samples",
    ):
        if len(by_role.get(required, [])) != 1:
            raise ValueError(f"strategy run evidence role {required} must appear exactly once")
    workload_count = len(by_role.get("workload_paired_samples", []))
    workload_required = terminal["mode"] == "full" and terminal["workload_status"] in {
        "evaluated", "workload_failed"
    }
    if workload_required and workload_count != 1:
        raise ValueError("evaluated full workload evidence must appear exactly once")
    if not workload_required and workload_count != 0:
        raise ValueError("workload paired evidence is not valid for this terminal outcome")
    if terminal["workload_status"] in {"evaluated", "workload_failed"} and terminal["mode"] != "full":
        raise ValueError("workload evidence requires full mode")
    if by_role["candidate"][0]["sha256"] != record["candidate_sha256"]:
        raise ValueError("strategy run candidate evidence is inconsistent")
    if by_role["decision"][0]["sha256"] != record["decision_sha256"]:
        raise ValueError("strategy run decision evidence is inconsistent")
    checkpoint_items = by_role["checkpoint"]
    if checkpoint_items[0]["sha256"] != record["checkpoint_identity"]:
        raise ValueError("strategy run checkpoint evidence is inconsistent")
    return record


def _record_key(record: Mapping) -> str:
    identity = {field: record[field] for field in sorted(_RECORD_FIELDS)}
    return hashlib.sha256(_canonical_bytes(identity)).hexdigest()


def _derived_indices(records: list[dict]) -> tuple[dict, dict]:
    methods: dict[str, dict] = {}
    bundles: dict[str, dict] = {}
    for record in records:
        identity = _record_key(record)
        method_ids = record["bundle"]["method_ids"]
        for method_id in method_ids:
            performance = record["methods"]["performance"].get(method_id)
            implementation = record["methods"]["implementation"].get(method_id)
            if performance is None and implementation is None:
                continue
            methods.setdefault(method_id, {"records": []})["records"].append({
                "record_identity": identity,
                "performance": performance,
                "implementation": implementation,
                "completed_at": record["completed_at"],
            })
        bundle_key = hashlib.sha256(_canonical_bytes(method_ids)).hexdigest()
        bundles.setdefault(
            bundle_key, {"method_ids": method_ids, "records": []}
        )["records"].append({
            "record_identity": identity,
            "outcome": record["bundle"]["outcome"],
            "decision_evidence": record["bundle"]["decision_evidence"],
            "completed_at": record["completed_at"],
        })
    return methods, bundles


def _new_memory() -> dict:
    return {"schema_version": MEMORY_SCHEMA, "scopes": {}}


def _validate_memory(value: Any) -> dict:
    memory = _strict_copy(value, "strategy memory")
    _exact_fields(memory, {"schema_version", "scopes"}, "strategy memory")
    if memory["schema_version"] != MEMORY_SCHEMA:
        raise ValueError(f"strategy memory schema_version must be {MEMORY_SCHEMA}")
    scopes = memory["scopes"]
    if not isinstance(scopes, dict):
        raise ValueError("strategy memory.scopes must be a JSON object")
    if len(scopes) > MAX_SCOPES:
        raise ValueError("strategy memory exceeds scope capacity")
    for key, entry in scopes.items():
        _sha(key, "strategy memory scope key")
        _exact_fields(entry, {"scope", "runs", "methods", "bundles"}, "scope entry")
        clean_scope = _validate_scope_document(entry["scope"])
        if _scope_key_from_document(clean_scope) != key:
            raise ValueError("strategy memory scope key does not match scope document")
        if not isinstance(entry["runs"], list):
            raise ValueError("scope entry.runs must be a JSON array")
        if len(entry["runs"]) > MAX_RUNS_PER_SCOPE:
            raise ValueError("scope entry exceeds run capacity")
        identities = set()
        clean_records = []
        for record in entry["runs"]:
            clean_record = _validate_record(record)
            clean_records.append(clean_record)
            if clean_record["input_hash"] != clean_scope["input_hash"]:
                raise ValueError(
                    "strategy run.input_hash does not match its scope.input_hash"
                )
            identity = _record_key(clean_record)
            if identity in identities:
                raise ValueError("scope entry contains duplicate strategy runs")
            identities.add(identity)
        methods = entry["methods"]
        bundles = entry["bundles"]
        if not isinstance(methods, dict) or not isinstance(bundles, dict):
            raise ValueError("scope entry methods and bundles must be JSON objects")
        for method_id, method_entry in methods.items():
            if not isinstance(method_id, str) or not method_id:
                raise ValueError("scope method id must be non-empty")
            _exact_fields(method_entry, {"records"}, "scope method entry")
            if not isinstance(method_entry["records"], list):
                raise ValueError("scope method records must be an array")
            for item in method_entry["records"]:
                _exact_fields(
                    item,
                    {"record_identity", "performance", "implementation", "completed_at"},
                    "scope method record",
                )
                if _sha(item["record_identity"], "scope method record identity") not in identities:
                    raise ValueError("scope method record references an unknown run")
                if item["performance"] is None and item["implementation"] is None:
                    raise ValueError("scope method record contains no evidence")
                for field in ("performance", "implementation"):
                    if item[field] is not None and not isinstance(item[field], dict):
                        raise ValueError(f"scope method record {field} must be an object or null")
        for bundle_key, bundle_entry in bundles.items():
            _sha(bundle_key, "scope bundle key")
            _exact_fields(bundle_entry, {"method_ids", "records"}, "scope bundle entry")
            method_ids = bundle_entry["method_ids"]
            if (
                not isinstance(method_ids, list)
                or method_ids != sorted(set(method_ids))
                or any(not isinstance(item, str) or not item for item in method_ids)
            ):
                raise ValueError("scope bundle method_ids must be sorted unique strings")
            if hashlib.sha256(_canonical_bytes(method_ids)).hexdigest() != bundle_key:
                raise ValueError("scope bundle key does not match method_ids")
            if not isinstance(bundle_entry["records"], list):
                raise ValueError("scope bundle records must be an array")
            for item in bundle_entry["records"]:
                _exact_fields(
                    item,
                    {"record_identity", "outcome", "decision_evidence", "completed_at"},
                    "scope bundle record",
                )
                if _sha(item["record_identity"], "scope bundle record identity") not in identities:
                    raise ValueError("scope bundle record references an unknown run")
                if not isinstance(item["outcome"], str) or not item["outcome"]:
                    raise ValueError("scope bundle outcome must be non-empty")
                if not isinstance(item["decision_evidence"], dict):
                    raise ValueError("scope bundle decision evidence must be an object")
        expected_methods, expected_bundles = _derived_indices(clean_records)
        if methods != expected_methods:
            raise ValueError("scope method index does not match verified run records")
        if bundles != expected_bundles:
            raise ValueError("scope bundle index does not match verified run records")
    return memory


def load_memory(path: str | os.PathLike) -> dict:
    """Read and validate an existing strategy memory without following symlinks."""
    value = _strict_json_bytes(artifact_store.read_regular_bytes(path), "strategy memory")
    return _validate_memory(value)


def _leaf_identity(directory_fd: int, leaf: str) -> tuple[int, int] | None:
    try:
        metadata = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"strategy memory path is a symlink or unsafe: {leaf}")
    return metadata.st_dev, metadata.st_ino


def _path_identity(directory_fd: int, leaf: str) -> tuple[int, int] | None:
    """Return a no-follow identity without requiring a regular file."""
    try:
        metadata = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return metadata.st_dev, metadata.st_ino


def _read_memory_leaf(
    directory_fd: int, leaf: str
) -> tuple[bytes | None, tuple[int, int] | None]:
    fd = None
    try:
        try:
            fd = os.open(
                leaf,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            return None, None
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ValueError("strategy memory path is a symlink or unsafe") from error
            raise
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("strategy memory path is not a regular file")
        identity = (metadata.st_dev, metadata.st_ino)
        if _leaf_identity(directory_fd, leaf) != identity:
            raise ValueError("strategy memory path changed while opening")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        if _leaf_identity(directory_fd, leaf) != identity:
            raise ValueError("strategy memory path changed while reading")
        return b"".join(chunks), identity
    finally:
        if fd is not None:
            os.close(fd)


def _open_lock(directory_fd: int, leaf: str) -> tuple[int, tuple[int, int]]:
    base_flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    created = False
    try:
        try:
            fd = os.open(
                leaf,
                base_flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            created = True
        except FileExistsError:
            fd = os.open(leaf, base_flags, dir_fd=directory_fd)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError("strategy memory lock is a symlink or unsafe") from error
        raise
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(fd)
        raise ValueError("strategy memory lock is not a regular file")
    if created:
        os.fchmod(fd, 0o600)
        metadata = os.fstat(fd)
    identity = (metadata.st_dev, metadata.st_ino)
    return fd, identity


def _check_identity(
    directory_fd: int,
    leaf: str,
    expected: tuple[int, int] | None,
    field: str,
) -> None:
    if _leaf_identity(directory_fd, leaf) != expected:
        raise ValueError(f"{field} path was replaced or changed during update")


def _rename_with_flags(
    directory_fd: int, source_leaf: str, target_leaf: str, *, operation: str
) -> None:
    """Perform a dirfd-bound atomic rename operation supported by the kernel."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = getattr(libc, "renameatx_np", None)
        flag = 0x00000002 if operation == "exchange" else 0x00000004
    elif sys.platform.startswith("linux"):
        function = getattr(libc, "renameat2", None)
        flag = 0x00000002 if operation == "exchange" else 0x00000001
    else:
        function = None
        flag = 0
    if function is None:
        raise OSError(
            errno.ENOTSUP,
            "strategy memory requires renameatx_np or renameat2 for safe publication",
        )
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = function(
        directory_fd,
        os.fsencode(source_leaf),
        directory_fd,
        os.fsencode(target_leaf),
        flag,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            f"{source_leaf} -> {target_leaf}",
        )


def _atomic_exchange(directory_fd: int, source_leaf: str, target_leaf: str) -> None:
    _rename_with_flags(
        directory_fd, source_leaf, target_leaf, operation="exchange"
    )


def _atomic_install_noreplace(
    directory_fd: int, source_leaf: str, target_leaf: str
) -> None:
    _rename_with_flags(
        directory_fd, source_leaf, target_leaf, operation="noreplace"
    )


def _publish_compare_exchange(
    directory_fd: int,
    temporary_leaf: str,
    target_leaf: str,
    *,
    expected_identity: tuple[int, int] | None,
    new_identity: tuple[int, int],
) -> None:
    """Publish only if target still has the identity observed under the lock."""
    if expected_identity is None:
        try:
            _atomic_install_noreplace(directory_fd, temporary_leaf, target_leaf)
        except FileExistsError as error:
            raise ValueError(
                "strategy memory path was replaced or changed during compare-install"
            ) from error
        _check_identity(directory_fd, target_leaf, new_identity, "strategy memory")
        os.fsync(directory_fd)
        return

    _atomic_exchange(directory_fd, temporary_leaf, target_leaf)
    displaced_identity = _path_identity(directory_fd, temporary_leaf)
    if displaced_identity == expected_identity:
        _check_identity(directory_fd, target_leaf, new_identity, "strategy memory")
        os.fsync(directory_fd)
        os.unlink(temporary_leaf, dir_fd=directory_fd)
        os.fsync(directory_fd)
        return

    # The exchange preserved the unexpected replacement under temporary_leaf.
    # Swap it back before reporting the failed compare operation.
    _check_identity(directory_fd, target_leaf, new_identity, "strategy memory")
    _atomic_exchange(directory_fd, temporary_leaf, target_leaf)
    if _path_identity(directory_fd, target_leaf) != displaced_identity:
        raise ValueError(
            "strategy memory unexpected replacement changed during restore"
        )
    _check_identity(directory_fd, temporary_leaf, new_identity, "strategy memory temp")
    os.fsync(directory_fd)
    os.unlink(temporary_leaf, dir_fd=directory_fd)
    os.fsync(directory_fd)
    raise ValueError(
        "strategy memory path was replaced or changed during compare-exchange"
    )


def _locked_memory_update(
    path: str | os.PathLike, updater: Callable[[dict], dict]
) -> dict:
    """Update one store under an adjacent flock and atomic dirfd-bound replace."""
    directory_fd, leaf, target = artifact_store._open_parent_directory(path, create=True)
    lock_leaf = leaf + ".lock"
    lock_fd = None
    temporary_leaf = None
    temporary_identity = None
    try:
        lock_fd, lock_identity = _open_lock(directory_fd, lock_leaf)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        _check_identity(directory_fd, lock_leaf, lock_identity, "strategy memory lock")
        raw, memory_identity = _read_memory_leaf(directory_fd, leaf)
        if raw is None:
            current = _new_memory()
        else:
            current = _validate_memory(_strict_json_bytes(raw, "strategy memory"))
        updated = updater(copy.deepcopy(current))
        if not isinstance(updated, dict):
            raise ValueError("strategy memory updater must return a JSON object")
        clean = _validate_memory(updated)
        payload = _canonical_bytes(clean) + b"\n"
        if raw == payload:
            _check_identity(directory_fd, lock_leaf, lock_identity, "strategy memory lock")
            _check_identity(directory_fd, leaf, memory_identity, "strategy memory")
            return clean

        _check_identity(directory_fd, lock_leaf, lock_identity, "strategy memory lock")
        _check_identity(directory_fd, leaf, memory_identity, "strategy memory")
        temporary_leaf = f".{leaf}.{secrets.token_hex(12)}.tmp"
        temp_fd = os.open(
            temporary_leaf,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        try:
            os.fchmod(temp_fd, 0o600)
            temp_metadata = os.fstat(temp_fd)
            temporary_identity = (temp_metadata.st_dev, temp_metadata.st_ino)
            offset = 0
            while offset < len(payload):
                written = os.write(temp_fd, payload[offset:])
                if written <= 0:
                    raise OSError("strategy memory write made no progress")
                offset += written
            os.fsync(temp_fd)
        finally:
            os.close(temp_fd)
        _check_identity(directory_fd, lock_leaf, lock_identity, "strategy memory lock")
        _publish_compare_exchange(
            directory_fd,
            temporary_leaf,
            leaf,
            expected_identity=memory_identity,
            new_identity=temporary_identity,
        )
        temporary_leaf = None
        return clean
    finally:
        if temporary_leaf is not None:
            # Only delete the file we created.  After an exchange failure the
            # temporary name can hold an unexpected replacement that must be
            # preserved for fail-closed recovery.
            if _path_identity(directory_fd, temporary_leaf) == temporary_identity:
                try:
                    os.unlink(temporary_leaf, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        os.close(directory_fd)


def append_run(
    memory_path: str | os.PathLike, scope: Mapping, record: Mapping
) -> bool:
    """Append one unique run to a validated scope; return False for a duplicate."""
    clean_scope = _validate_scope_document(scope)
    key = _scope_key_from_document(clean_scope)
    clean_record = _validate_record(record)
    if clean_record["input_hash"] != clean_scope["input_hash"]:
        raise ValueError("strategy run.input_hash does not match scope.input_hash")
    identity = _record_key(clean_record)
    inserted = False

    def update(memory):
        nonlocal inserted
        scopes = memory["scopes"]
        entry = scopes.get(key)
        if entry is None:
            if len(scopes) >= MAX_SCOPES:
                raise ValueError("strategy memory scope capacity reached")
            entry = {
                "scope": clean_scope,
                "runs": [],
                "methods": {},
                "bundles": {},
            }
            scopes[key] = entry
        elif entry["scope"] != clean_scope:
            raise ValueError("strategy memory scope key collision")
        if any(_record_key(existing) == identity for existing in entry["runs"]):
            return memory
        if len(entry["runs"]) >= MAX_RUNS_PER_SCOPE:
            raise ValueError("strategy memory run capacity reached")
        entry["runs"].append(clean_record)
        entry["methods"], entry["bundles"] = _derived_indices(entry["runs"])
        inserted = True
        return memory

    _locked_memory_update(memory_path, update)
    return inserted


def _run_root(path: str | os.PathLike) -> tuple[Path, tuple[int, int]]:
    root = Path(path).expanduser().absolute()
    if root.is_symlink():
        raise ValueError("run directory must not be a symlink")
    try:
        info = root.lstat()
    except OSError as error:
        raise ValueError("run directory is missing or unsafe") from error
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("run directory must be a real directory")
    # The safe reader below validates every parent component; opening this
    # required leaf makes a parent-directory alias fail before any schema work.
    artifact_store.read_regular_bytes(root / "manifest.json")
    return root, (info.st_dev, info.st_ino)


def _artifact(path: str | os.PathLike, *, root: Path, role: str) -> tuple[dict, bytes]:
    candidate = Path(path).expanduser().absolute()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{role} escapes the run directory") from error
    payload = artifact_store.read_regular_bytes(candidate)
    return {
        "role": role,
        "path": str(candidate),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }, payload


def _strict_document(path: str | os.PathLike, *, root: Path, role: str) -> tuple[dict, dict]:
    evidence, payload = _artifact(path, root=root, role=role)
    value = _strict_json_bytes(payload, role)
    if not isinstance(value, dict):
        raise ValueError(f"{role} must contain a JSON object")
    return value, evidence


def _replay_decision(payload: Mapping) -> dict:
    evidence = payload.get("evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "kernel", "workload", "constraints", "pareto"
    }:
        raise ValueError("decision evidence is required for strict replay")
    replay = decision.decide(
        mode=payload.get("mode"),
        kernel=evidence["kernel"],
        workload=evidence["workload"],
        constraints=evidence["constraints"],
        pareto=evidence["pareto"],
    )
    for field, expected in replay.items():
        if payload.get(field) != expected:
            raise ValueError(f"decision replay mismatch: {field}")
    kernel = evidence["kernel"]
    if "statistics" not in replay and isinstance(kernel, Mapping) and isinstance(
        kernel.get("statistics"), Mapping
    ):
        if payload.get("statistics") != kernel["statistics"]:
            raise ValueError("decision replay mismatch: statistics")
        replay["statistics"] = _strict_copy(kernel["statistics"], "replayed statistics")
    return replay


def _finite_positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0.0 else None


def _inside_exact(path: Path, parent: Path) -> bool:
    return path.absolute().parent == parent.absolute()


def _method_evidence(
    root: Path,
    iteration: int,
    method_ids: list[str],
    champion_bench_path: Path,
    terminal_sass: Any,
) -> tuple[dict, list[dict]]:
    iter_dir = root / f"iterv{iteration}"
    performance: dict[str, dict] = {}
    implementation: dict[str, dict] = {}
    evidence: list[dict] = []

    if not isinstance(terminal_sass, Mapping):
        raise ValueError("terminal SASS snapshot must be a JSON object")
    sass_path = iter_dir / "sass_check.json"
    if terminal_sass:
        try:
            sass, sass_identity = _strict_document(
                sass_path, root=root, role="sass_check"
            )
            if sass != terminal_sass:
                raise ValueError("current SASS evidence differs from terminal snapshot")
            evidence.append(sass_identity)
            checks = sass.get("checks", [])
            if isinstance(checks, list):
                for check in checks:
                    if not isinstance(check, Mapping):
                        continue
                    method_id = check.get("method_id")
                    status_value = check.get("status")
                    if method_id in method_ids and isinstance(status_value, str):
                        implementation[method_id] = {
                            "status": status_value,
                            "evidence": sass_identity,
                        }
        except (OSError, ValueError) as error:
            raise ValueError(f"terminal SASS evidence is missing, unsafe, or drifted: {error}") from error

    attribution_path = iter_dir / "attribution.json"
    if not attribution_path.exists() and not attribution_path.is_symlink():
        return {"performance": performance, "implementation": implementation}, evidence
    attribution, attribution_identity = _strict_document(
        attribution_path, root=root, role="attribution"
    )
    evidence.append(attribution_identity)
    entries = attribution.get("attributions")
    if not isinstance(entries, list):
        return {"performance": performance, "implementation": implementation}, evidence
    try:
        champion_identity, champion_bytes = _artifact(
            champion_bench_path, root=root, role="champion_bench"
        )
        champion = _strict_json_bytes(champion_bytes, "champion_bench")
    except (OSError, ValueError):
        return {"performance": performance, "implementation": implementation}, evidence
    for item in entries:
        if not isinstance(item, Mapping):
            continue
        method_id = item.get("method_id")
        if method_id not in method_ids:
            continue
        method_dir = iter_dir / "ablations" / method_id.replace(".", "_")
        try:
            ablated_kernel = Path(item["ablated_kernel"]).absolute()
            ablated_bench_path = Path(item["ablated_bench"]).absolute()
            if not _inside_exact(ablated_kernel, method_dir) or not _inside_exact(
                ablated_bench_path, method_dir
            ):
                continue
            kernel_identity, _ = _artifact(
                ablated_kernel, root=root, role=f"ablation_kernel:{method_id}"
            )
            bench_identity, bench_bytes = _artifact(
                ablated_bench_path, root=root, role=f"ablation_bench:{method_id}"
            )
            if item.get("ablated_kernel_sha256") != kernel_identity["sha256"]:
                continue
            if item.get("ablated_bench_sha256") != bench_identity["sha256"]:
                continue
            if item.get("champion_bench") != champion_identity["path"]:
                continue
            if item.get("champion_bench_sha256") != champion_identity["sha256"]:
                continue
            ablated = _strict_json_bytes(bench_bytes, "ablated_bench")
            if not isinstance(champion, Mapping) or not isinstance(ablated, Mapping):
                continue
            champion_correctness = champion.get("correctness")
            if (
                not isinstance(champion_correctness, Mapping)
                or champion_correctness.get("passed") is not True
            ):
                continue
            correctness = ablated.get("correctness")
            if not isinstance(correctness, Mapping) or correctness.get("passed") is not True:
                continue
            champion_ms = _finite_positive((champion.get("kernel") or {}).get("average_ms"))
            ablated_ms = _finite_positive((ablated.get("kernel") or {}).get("average_ms"))
            if champion_ms is None or ablated_ms is None:
                continue
            delta = ablated_ms - champion_ms
            percentage = delta / champion_ms * 100.0
            if delta == 0.0 or percentage == 0.0:
                continue
            declared = (
                item.get("champion_ms"), item.get("ablated_ms"),
                item.get("attribution_ms"), item.get("attribution_pct"),
            )
            expected = (
                round(champion_ms, 4), round(ablated_ms, 4),
                round(delta, 4), round(percentage, 2),
            )
            if any(
                isinstance(actual, bool) or not isinstance(actual, (int, float))
                or float(actual) != wanted
                for actual, wanted in zip(declared, expected)
            ):
                continue
            performance[method_id] = {
                "outcome": "positive" if delta > 0 else "negative",
                "champion_ms": champion_ms,
                "ablated_ms": ablated_ms,
                "attribution_ms": delta,
                "attribution_pct": percentage,
                "evidence_quality": "diagnostic_unpaired_ablation",
                "promotion_authority": False,
                "evidence": [attribution_identity, champion_identity, kernel_identity, bench_identity],
            }
            evidence.extend([champion_identity, kernel_identity, bench_identity])
        except (KeyError, OSError, TypeError, ValueError):
            continue
    return {"performance": performance, "implementation": implementation}, evidence


def load_completed_run(run_dir: str | os.PathLike) -> dict:
    """Revalidate one completed v2.2 run and return detached strategy evidence."""
    root, root_identity = _run_root(run_dir)
    manifest_path = root / "manifest.json"
    state_path = root / "state.json"
    checkpoint_path = root / "checkpoint.json"
    initial: dict[str, str] = {}
    for role, path in (
        ("manifest", manifest_path), ("state", state_path),
        ("checkpoint", checkpoint_path),
    ):
        item, _ = _artifact(path, root=root, role=role)
        initial[item["path"]] = item["sha256"]

    manifest, input_hash = orchestrate._load_and_verify_manifest(root)
    state_payload, state_identity = _strict_document(
        state_path, root=root, role="state"
    )
    state.validate_state(state_payload)
    orchestrate._verify_state_candidates(state_payload)
    checkpoint, checkpoint_identity = _strict_document(
        checkpoint_path, root=root, role="checkpoint"
    )
    checkpoint = orchestrate._validate_checkpoint(checkpoint, input_hash=input_hash)
    if checkpoint["stage"] != "complete" or checkpoint["status"] != "complete":
        raise ValueError("checkpoint must be complete before strategy recording")
    terminal = state_payload.get("terminal_decision")
    if not isinstance(terminal, Mapping):
        raise ValueError("completed state is missing terminal_decision")
    if state_payload.get("run_dir") != str(root):
        raise ValueError("state run_dir does not match selected run")
    if not (
        state_payload.get("input_hash") == input_hash == terminal.get("input_hash")
        == checkpoint.get("input_hash")
    ):
        raise ValueError("run identity mismatch")
    if not (
        manifest.get("mode") == state_payload.get("mode") == terminal.get("mode")
    ):
        raise ValueError("manifest, state, and terminal mode mismatch")
    if checkpoint.get("run_dir") not in {None, str(root)}:
        raise ValueError("checkpoint run_dir does not match selected run")
    iteration = terminal.get("iteration")
    if type(iteration) is not int or checkpoint.get("iteration") != iteration:
        raise ValueError("terminal and checkpoint iteration mismatch")
    if checkpoint.get("candidate_id") != terminal.get("candidate_id") or checkpoint.get(
        "candidate_status"
    ) != terminal.get("status"):
        raise ValueError("terminal and checkpoint candidate identity mismatch")
    if checkpoint.get("candidate_file") != terminal.get("candidate_file") or checkpoint.get(
        "candidate_sha256"
    ) != terminal.get("candidate_sha256"):
        raise ValueError("checkpoint candidate file/hash does not match terminal candidate")
    resume = terminal.get("resume")
    if not isinstance(resume, Mapping) or resume.get("stage") != "complete" or resume.get(
        "status"
    ) != "complete":
        raise ValueError("terminal resume is not bound to complete checkpoint")

    decision_path = Path(terminal.get("decision_json", "")).absolute()
    if decision_path.parent != root / f"iterv{iteration}":
        raise ValueError("terminal decision escapes its iteration")
    decision_payload, decision_identity = _strict_document(
        decision_path, root=root, role="decision"
    )
    if decision_identity["sha256"] != terminal.get("decision_sha256"):
        raise ValueError("terminal decision sha256 drifted")
    replay = _replay_decision(decision_payload)
    # decision.json is compared field-for-field by _replay_decision.  State's
    # terminal snapshot predates a mandatory reason field, but if it records
    # one it must be the replayed reason rather than an independent claim.
    if "reason" in terminal and terminal.get("reason") != replay.get("reason"):
        raise ValueError("terminal decision replay mismatch: reason")
    for field in ("candidate_id", "candidate_file", "candidate_sha256"):
        if terminal.get(field) != decision_payload.get(field):
            raise ValueError(f"terminal decision candidate mismatch: {field}")
    for field in (
        "status", "mode", "statistics", "workload_status",
        "workload_statistics", "workload_failure", "constraints", "pareto",
    ):
        replay_value = replay.get(field)
        if field in {"constraints", "pareto"} and field not in replay:
            replay_value = replay["evidence"][field]
        if terminal.get(field) != replay_value:
            raise ValueError(f"terminal decision replay mismatch: {field}")
    if terminal.get("candidate_sha256") != decision_payload.get("candidate_sha256"):
        raise ValueError("terminal candidate sha256 mismatch")

    iter_dir = root / f"iterv{iteration}"
    methods_payload, methods_identity = _strict_document(
        iter_dir / "methods.json", root=root, role="methods"
    )
    methods = methods_payload.get("methods")
    if not isinstance(methods, list):
        raise ValueError("terminal methods.json must contain methods list")
    method_ids = []
    for item in methods:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str) or not item["id"]:
            raise ValueError("terminal method id is malformed")
        method_ids.append(item["id"])
    if len(set(method_ids)) != len(method_ids):
        raise ValueError("terminal method ids must be unique")
    method_evidence, optional_evidence = _method_evidence(
        root, iteration, method_ids, iter_dir / "bench.json", terminal.get("sass", {})
    )
    scope = scope_document(manifest_path)
    evidence = [state_identity, checkpoint_identity, decision_identity, methods_identity]
    evaluated_outcomes = {
        "kernel_only_win", "end_to_end_win", "confirmed_loss", "inconclusive"
    }
    if replay["status"] in evaluated_outcomes and not isinstance(
        terminal.get("kernel_paired_samples"), Mapping
    ):
        raise ValueError("evaluated terminal outcome requires kernel paired samples")
    if replay.get("workload_status") in {"evaluated", "workload_failed"} and not isinstance(
        terminal.get("workload_paired_samples"), Mapping
    ):
        raise ValueError("attempted workload requires workload paired samples")
    for field in ("kernel_paired_samples", "workload_paired_samples"):
        binding = terminal.get(field)
        if isinstance(binding, Mapping):
            item, _ = _artifact(binding["path"], root=root, role=field)
            if item["sha256"] != binding.get("sha256"):
                raise ValueError(f"{field} hash drifted")
            evidence.append(item)
    candidate_sha = terminal.get("candidate_sha256")
    if not isinstance(candidate_sha, str):
        raise ValueError("terminal candidate identity is required")
    candidate_item, _ = _artifact(
        terminal["candidate_file"], root=root, role="candidate"
    )
    if candidate_item["sha256"] != candidate_sha:
        raise ValueError("terminal candidate drifted")
    state_candidates = state_payload.get("candidates", {})
    if not any(
        isinstance(item, Mapping)
        and (item.get("candidate_file") or item.get("path")) == terminal["candidate_file"]
        and (item.get("candidate_sha256") or item.get("sha256")) == candidate_sha
        and item.get("status") == terminal.get("status")
        for item in state_candidates.values()
    ):
        raise ValueError("state candidate does not match terminal candidate")
    evidence.extend([candidate_item, *optional_evidence])

    # Re-open every critical artifact after all expensive validation and replay.
    if _run_root(root)[1] != root_identity:
        raise ValueError("run directory identity changed during validation")
    for path, digest in initial.items():
        current, _ = _artifact(path, root=root, role="critical_recheck")
        if current["sha256"] != digest:
            raise ValueError("critical run artifact changed during validation")
    for item in evidence:
        current, _ = _artifact(item["path"], root=root, role=item["role"])
        if current["sha256"] != item["sha256"]:
            raise ValueError("run evidence changed during validation")

    record = {
        "schema_version": RECORD_SCHEMA,
        "input_hash": input_hash,
        "candidate_sha256": candidate_sha,
        "decision_sha256": decision_identity["sha256"],
        "checkpoint_identity": checkpoint_identity["sha256"],
        "run_root": {
            "path": str(root), "device": root_identity[0], "inode": root_identity[1],
        },
        "scope": scope,
        "completed_at": checkpoint["updated_at"],
        "terminal": {
            **{
                key: replay.get(key)
                for key in (
                    "status", "mode", "reason", "statistics", "workload_status",
                    "workload_statistics", "workload_failure",
                )
            },
            "constraints": replay.get("constraints", replay["evidence"]["constraints"]),
            "pareto": replay.get("pareto", replay["evidence"]["pareto"]),
        },
        "bundle": {
            "outcome": replay["status"],
            "method_ids": sorted(method_ids),
            "promotion_authority": False,
            "decision_evidence": decision_identity,
        },
        "methods": method_evidence,
        "evidence": sorted(
            {item["path"]: item for item in evidence}.values(),
            key=lambda item: (item["role"], item["path"]),
        ),
    }
    return _validate_record(record)


def _atomic_write_output(path: str | os.PathLike, payload: Mapping) -> None:
    artifact_store.atomic_write_json(path, _strict_copy(payload, "strategy record output"))


def _paths_alias(first: str | os.PathLike, second: str | os.PathLike) -> bool:
    first_path = os.path.realpath(os.path.abspath(os.fspath(first)))
    second_path = os.path.realpath(os.path.abspath(os.fspath(second)))
    if first_path == second_path:
        return True
    try:
        return os.path.samefile(first_path, second_path)
    except OSError as error:
        if error.errno in {errno.ENOENT, errno.ENOTDIR}:
            return False
        raise ValueError("cannot safely compare strategy input and output paths") from error


def _memory_lock_path(memory_path: str | os.PathLike) -> str:
    return f"{os.fspath(memory_path)}.lock"


def _protected_frozen_inputs(
    manifest_path: str | os.PathLike,
) -> tuple[dict, list[str | os.PathLike]]:
    manifest, scope = _validated_manifest_scope(
        manifest_path, verify_files=False
    )
    protected: list[str | os.PathLike] = [
        manifest_path,
        manifest["inputs"]["baseline"]["path"],
        manifest["inputs"]["ref"]["path"],
    ]
    if manifest["mode"] == "full":
        raw = manifest["workload"]
        spec = workload_adapter.WorkloadSpec(
            kind=raw["kind"],
            source=_strict_copy(raw["source"], "manifest.workload.source"),
            objective=workload_adapter.validate_objective(raw["objective"]),
            cases=tuple(_strict_copy(raw["cases"], "manifest.workload.cases")),
            source_hash=raw["source_hash"],
        )
        if spec.kind == "python":
            bundle = workload_adapter._read_python_bundle(spec.source)
            snapshots = (bundle.source,) + tuple(
                snapshot for _, snapshot in bundle.dependencies
            )
        else:
            normalized, snapshots = workload_adapter._normalize_command_source(
                spec.source
            )
            if normalized != list(spec.source):
                raise ValueError(
                    "workload source_hash mismatch; command normalization changed"
                )
        workload_adapter._verify_source_hash(spec, snapshots)
        protected.extend(snapshot.path for snapshot in snapshots)
    return scope, protected


def _reject_output_aliases(
    output_path: str | os.PathLike,
    protected_paths: list[str | os.PathLike],
) -> None:
    for protected_path in protected_paths:
        if _paths_alias(output_path, protected_path):
            raise ValueError(
                f"strategy output aliases a protected input: {protected_path}"
            )


def record_run(
    memory_path: str | os.PathLike,
    run_dir: str | os.PathLike,
    output_path: str | os.PathLike,
) -> dict:
    """Validate fully, update memory under lock, then publish the record output."""
    _reject_output_aliases(
        output_path,
        [memory_path, _memory_lock_path(memory_path)],
    )
    _scope, frozen_inputs = _protected_frozen_inputs(
        Path(run_dir) / "manifest.json"
    )
    _reject_output_aliases(output_path, frozen_inputs)
    record = load_completed_run(run_dir)
    _reject_output_aliases(
        output_path,
        [
            *(item["path"] for item in record["evidence"]),
        ],
    )
    inserted = append_run(memory_path, record["scope"], record)
    _atomic_write_output(output_path, record)
    return {"inserted": inserted, "record": record}


def _optional_memory(path: str | os.PathLike) -> dict | None:
    """Read a memory snapshot without following path components."""
    try:
        directory_fd, leaf, target = artifact_store._open_parent_directory(
            path, create=False
        )
    except FileNotFoundError:
        return None
    try:
        raw = artifact_store._read_regular_leaf(
            directory_fd, leaf, target, missing_ok=True
        )
    finally:
        os.close(directory_fd)
    if raw is None:
        return None
    return _validate_memory(_strict_json_bytes(raw, "strategy memory"))


def _suggestion_base(scope: Mapping, key: str, *, status: str) -> dict:
    return {
        "schema_version": SUGGESTION_SCHEMA,
        "status": status,
        "advisory": True,
        "guardrail": ADVISORY_GUARDRAIL,
        "scope_key": key,
        "scope": _strict_copy(scope, "suggestion scope"),
        "preferred_method_ids": [],
        "caution_method_ids": [],
        "method_evidence": {},
        "prior_bundles": [],
    }


def _method_suggestion(method_id: str, records: list[Mapping]) -> dict:
    evidence_records = []
    positive_count = 0
    negative_count = 0
    for item in records:
        performance = item["performance"]
        if performance is None:
            continue
        outcome = performance["outcome"]
        positive_count += outcome == "positive"
        negative_count += outcome == "negative"
        run = item["run"]
        evidence_records.append(
            {
                "record_identity": item["record_identity"],
                "completed_at": item["completed_at"],
                "outcome": outcome,
                "decision_evidence": _strict_copy(
                    run["bundle"]["decision_evidence"],
                    "suggestion decision evidence",
                ),
                "ablation_evidence": sorted(
                    _strict_copy(
                        performance["evidence"], "suggestion ablation evidence"
                    ),
                    key=lambda value: (value["role"], value["path"], value["sha256"]),
                ),
                "count": 1,
            }
        )
    evidence_records.sort(
        key=lambda value: (value["completed_at"], value["record_identity"])
    )
    return {
        "method_id": method_id,
        "count": len(evidence_records),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "records": evidence_records,
    }


def _available_suggestion(scope: Mapping, key: str, entry: Mapping) -> dict:
    suggestion = _suggestion_base(scope, key, status="available")
    runs = {_record_key(record): record for record in entry["runs"]}
    selected = {}
    for method_id in sorted(entry["methods"]):
        indexed = []
        for item in entry["methods"][method_id]["records"]:
            indexed.append({**item, "run": runs[item["record_identity"]]})
        method = _method_suggestion(method_id, indexed)
        if method["positive_count"] >= 2 and method["negative_count"] == 0:
            suggestion["preferred_method_ids"].append(method_id)
            selected[method_id] = method
        elif method["negative_count"] > 0:
            suggestion["caution_method_ids"].append(method_id)
            selected[method_id] = method
    suggestion["method_evidence"] = {
        method_id: selected[method_id] for method_id in sorted(selected)
    }

    prior_bundles = []
    for bundle in entry["bundles"].values():
        for item in bundle["records"]:
            prior_bundles.append(
                {
                    "record_identity": item["record_identity"],
                    "completed_at": item["completed_at"],
                    "outcome": item["outcome"],
                    "method_ids": list(bundle["method_ids"]),
                    "decision_evidence": _strict_copy(
                        item["decision_evidence"], "suggestion bundle evidence"
                    ),
                    "count": 1,
                }
            )
    suggestion["prior_bundles"] = sorted(
        prior_bundles,
        key=lambda value: (value["completed_at"], value["record_identity"]),
    )
    return suggestion


def suggest_strategies(
    memory_path: str | os.PathLike,
    manifest_path: str | os.PathLike,
    output_path: str | os.PathLike,
) -> dict:
    """Publish exact-scope search hints without reading or changing run state."""
    _reject_output_aliases(
        output_path,
        [memory_path, _memory_lock_path(memory_path)],
    )
    scope, frozen_inputs = _protected_frozen_inputs(manifest_path)
    _reject_output_aliases(output_path, frozen_inputs)
    key = _scope_key_from_document(scope)
    memory = _optional_memory(memory_path)
    if memory is None:
        suggestion = _suggestion_base(scope, key, status="unavailable")
        suggestion["reason"] = "strategy memory does not exist"
    else:
        entry = memory["scopes"].get(key)
        if entry is None:
            suggestion = _suggestion_base(scope, key, status="unavailable")
            suggestion["reason"] = "strategy memory has no records for the exact scope"
        else:
            suggestion = _available_suggestion(scope, key, entry)
    _atomic_write_output(output_path, suggestion)
    return suggestion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage explicit CUDA strategy evidence")
    subparsers = parser.add_subparsers(dest="command", required=True)
    suggest_parser = subparsers.add_parser(
        "suggest", help="write exact-scope advisory search hints"
    )
    suggest_parser.add_argument("--memory", required=True)
    suggest_parser.add_argument("--manifest", required=True)
    suggest_parser.add_argument("--out", required=True)
    record_parser = subparsers.add_parser("record", help="record one completed v2.2 run")
    record_parser.add_argument("--memory", required=True)
    record_parser.add_argument("--run-dir", required=True)
    record_parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "suggest":
        result = suggest_strategies(args.memory, args.manifest, args.out)
        print(json.dumps({"status": result["status"], "out": args.out}, sort_keys=True))
        return 0 if result["status"] == "available" else 2
    if args.command == "record":
        result = record_run(args.memory, args.run_dir, args.out)
        print(json.dumps({"inserted": result["inserted"], "out": args.out}, sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
