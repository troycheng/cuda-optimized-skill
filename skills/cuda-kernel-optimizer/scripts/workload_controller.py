#!/usr/bin/env python3
"""Strict contracts and orchestration entry point for workload optimization."""

from __future__ import annotations

import argparse
import copy
import difflib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import stat
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


CONTROL_SCHEMA_V1 = "cuda-workload-optimizer/control-v1"
CONTROL_SCHEMA_V2 = "cuda-workload-optimizer/control-v2"
CONTROL_SCHEMA = CONTROL_SCHEMA_V1
CHANGE_SCHEMA = "cuda-workload-optimizer/change-v1"
_BUDGETS = {"fast", "quick", "balanced", "thorough"}
_PROBE_KINDS = {
    "environment",
    "timeline",
    "framework",
    "cpu_data",
    "transfer",
    "communication",
    "io",
    "custom",
}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SENSITIVE_KEY = re.compile(
    r"(^|[_-])(api[_-]?key|authorization|cookie|credential|password|secret|token)($|[_-])",
    re.IGNORECASE,
)
_LOG_SECRET = re.compile(
    r'''(?i)(["']?\b[A-Z0-9_]{0,128}(?:API[_-]?KEY|AUTH|COOKIE|CREDENTIAL|PASSWORD|SECRET|TOKEN)[A-Z0-9_]{0,128}\b["']?\s*[:=]\s*)(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\r\n,;}]+)'''
)
_DEFAULT_LOG_LIMIT = 64 * 1024
_OUTPUT_LIMIT = 1024 * 1024
_SAFE_ENV = {
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONPATH",
    "TMPDIR",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "NVIDIA_VISIBLE_DEVICES",
}
_DIAGNOSIS_MODULE = None
_REVIEWER_MODULE = None
_WORKLOAD_MODULE = None
_EVALUATE_MODULE = None
_BUDGET_MODULE = None
_READINESS_CONTRACT_MODULE = None
_READINESS_GATE_MODULE = None
_READINESS_IDENTITY_MODULE = None
_CHECK_ENV_MODULE = None
_ANALYSIS_EPOCH_MODULE = None
_EXECUTION_MAP_MODULE = None
_HYPOTHESIS_SPACE_MODULE = None
_EVIDENCE_SELECTOR_MODULE = None
_ACTIVE_DIAGNOSIS_CONTRACT_SCHEMA = "cuda-optimizer/active-diagnosis-contract-v1"
_GLOBAL_SCAN_DRAFT_SCHEMA = "cuda-optimizer/global-scan-draft-v1"
_BUDGET_RUNTIME = {
    "quick": {
        "soft_target_seconds": 900,
        "hard_ceiling_seconds": 2700,
        "blocks": 3,
        "retries": 0,
        "bootstrap": 200,
    },
    "balanced": {
        "soft_target_seconds": 3600,
        "hard_ceiling_seconds": 10800,
        "blocks": 5,
        "retries": 1,
        "bootstrap": 1000,
    },
    "thorough": {
        "soft_target_seconds": 14400,
        "hard_ceiling_seconds": 36000,
        "blocks": 9,
        "retries": 2,
        "bootstrap": 5000,
    },
}


class ValidationError(ValueError):
    """Raised when a workload-controller contract is not closed and safe."""


def _pairs_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_object(path: os.PathLike[str] | str) -> dict:
    """Load one JSON object while rejecting duplicate keys and non-finite numbers."""
    source = Path(path)
    try:
        payload = source.read_text(encoding="utf-8")
    except OSError as error:
        raise ValidationError(f"cannot read JSON file {source}: {error}") from error
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=lambda token: (_raise_invalid_number(token)),
        )
    except ValidationError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValidationError(f"invalid JSON in {source}: {error}") from error
    if type(value) is not dict:
        raise ValidationError(f"JSON root must be an object: {source}")
    return value


def _raise_invalid_number(token: str):
    raise ValidationError(f"JSON number must be finite: {token}")


def _object(value: Any, field: str) -> dict:
    if type(value) is not dict:
        raise ValidationError(f"{field} must be an object")
    return value


def _closed(value: dict, allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValidationError(f"{field} contains unknown fields: {', '.join(unknown)}")


def _required(value: dict, required: set[str], field: str) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ValidationError(f"{field} is missing required fields: {', '.join(missing)}")


def _string(value: Any, field: str, *, max_length: int = 4096) -> str:
    if type(value) is not str or not value.strip():
        raise ValidationError(f"{field} must be a non-empty string")
    if len(value) > max_length:
        raise ValidationError(f"{field} exceeds {max_length} characters")
    return value


def _identifier(value: Any, field: str) -> str:
    text = _string(value, field, max_length=128)
    if _IDENTIFIER.fullmatch(text) is None:
        raise ValidationError(f"{field} must be a safe identifier")
    return text


def _timeout(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or not 1 <= number <= 3600:
        raise ValidationError(f"{field} must be between 1 and 3600 seconds")
    return number


def _argv(value: Any, field: str) -> list[str]:
    if type(value) is not list or not value:
        raise ValidationError(f"{field} argv must be a non-empty array")
    result = []
    for index, item in enumerate(value):
        result.append(_string(item, f"{field} argv[{index}]"))
    return result


def _absolute(value: Any, field: str) -> Path:
    text = _string(value, field)
    expanded = Path(os.path.expanduser(text))
    if not expanded.is_absolute():
        raise ValidationError(f"{field} must be an absolute path")
    return expanded.resolve(strict=False)


def _relative(value: Any, field: str) -> Path:
    text = _string(value, field)
    path = Path(text)
    if path.is_absolute() or text in {".", ".."} or ".." in path.parts:
        raise ValidationError(f"{field} must be a contained relative path")
    normalized = Path(os.path.normpath(text))
    if str(normalized) in {"", ".", ".."} or ".." in normalized.parts:
        raise ValidationError(f"{field} must be a contained relative path")
    return normalized


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _json_copy(value: Any, field: str, *, reject_sensitive: bool = False) -> Any:
    if value is None or type(value) in {bool, str, int}:
        return copy.deepcopy(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValidationError(f"{field} numbers must be finite")
        return value
    if type(value) is list:
        return [
            _json_copy(item, f"{field}[{index}]", reject_sensitive=reject_sensitive)
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        result = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ValidationError(f"{field} keys must be non-empty strings")
            if reject_sensitive and _SENSITIVE_KEY.search(key):
                raise ValidationError(f"{field} must not contain credentials: {key}")
            result[key] = _json_copy(
                item, f"{field}.{key}", reject_sensitive=reject_sensitive
            )
        return result
    raise ValidationError(f"{field} must contain JSON-compatible values")


def _string_list(value: Any, field: str, *, identifiers: bool = False) -> list[str]:
    if type(value) is not list or not value:
        raise ValidationError(f"{field} must be a non-empty array")
    result = []
    for index, item in enumerate(value):
        if identifiers:
            result.append(_identifier(item, f"{field}[{index}]"))
        else:
            result.append(_string(item, f"{field}[{index}]"))
    if len(set(result)) != len(result):
        raise ValidationError(f"{field} must not contain duplicates")
    return result


def _sha256(value: Any, field: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValidationError(f"{field} must be lowercase SHA-256")
    return value


def _validate_active_diagnosis_contract(value: Mapping[str, Any]) -> dict:
    contract = _object(value, "analysis_contract")
    fields = {
        "schema_version",
        "global_scan_probe_id",
        "adapter_path",
        "analysis_policy_sha256",
        "source",
        "actions",
        "selection_policy",
    }
    _closed(contract, fields, "analysis_contract")
    _required(contract, fields, "analysis_contract")
    if contract["schema_version"] != _ACTIVE_DIAGNOSIS_CONTRACT_SCHEMA:
        raise ValidationError(
            f"analysis_contract.schema_version must be {_ACTIVE_DIAGNOSIS_CONTRACT_SCHEMA}"
        )
    _identifier(contract["global_scan_probe_id"], "analysis_contract.global_scan_probe_id")
    _absolute(contract["adapter_path"], "analysis_contract.adapter_path")
    _sha256(
        contract["analysis_policy_sha256"],
        "analysis_contract.analysis_policy_sha256",
    )
    source = _object(contract["source"], "analysis_contract.source")
    source_fields = {
        "profiler",
        "profiler_version",
        "export_schema",
        "adapter_id",
        "adapter_version",
        "adapter_sha256",
    }
    _closed(source, source_fields, "analysis_contract.source")
    _required(source, source_fields, "analysis_contract.source")
    if source["profiler"] not in {"nsys", "pytorch", "perfetto", "custom"}:
        raise ValidationError("analysis_contract.source.profiler is unsupported")
    for field in ("profiler_version", "export_schema", "adapter_version"):
        _string(source[field], f"analysis_contract.source.{field}", max_length=256)
    _identifier(source["adapter_id"], "analysis_contract.source.adapter_id")
    _sha256(source["adapter_sha256"], "analysis_contract.source.adapter_sha256")
    if type(contract["actions"]) is not list or not contract["actions"]:
        raise ValidationError("analysis_contract.actions must be a non-empty array")
    actions = []
    action_ids = set()
    for index, raw in enumerate(contract["actions"]):
        action = _object(raw, f"analysis_contract.actions[{index}]")
        action_fields = {
            "action_id",
            "adapter_path",
            "adapter_sha256",
            "argv",
            "timeout_seconds",
        }
        _closed(action, action_fields, f"analysis_contract.actions[{index}]")
        _required(action, action_fields, f"analysis_contract.actions[{index}]")
        action_id = _identifier(
            action["action_id"], f"analysis_contract.actions[{index}].action_id"
        )
        if action_id in action_ids:
            raise ValidationError("analysis_contract action ids must be unique")
        action_ids.add(action_id)
        actions.append(
            {
                "action_id": action_id,
                "adapter_path": str(
                    _absolute(
                        action["adapter_path"],
                        f"analysis_contract.actions[{index}].adapter_path",
                    )
                ),
                "adapter_sha256": _sha256(
                    action["adapter_sha256"],
                    f"analysis_contract.actions[{index}].adapter_sha256",
                ),
                "argv": _argv(
                    action["argv"], f"analysis_contract.actions[{index}]"
                ),
                "timeout_seconds": _timeout(
                    action["timeout_seconds"],
                    f"analysis_contract.actions[{index}].timeout_seconds",
                ),
            }
        )
    actions.sort(key=lambda item: item["action_id"])
    try:
        policy = _load_evidence_selector_module()._validate_policy(
            contract["selection_policy"]
        )
    except ValueError as error:
        raise ValidationError(f"invalid analysis selection policy: {error}") from error
    normalized = copy.deepcopy(dict(contract))
    normalized["actions"] = actions
    # Capability admission is Controller-owned and is rebuilt from the current
    # readiness report when the diagnosis context is created.
    policy["available_capability_ids"] = []
    normalized["selection_policy"] = policy
    return normalized


def validate_control_manifest(value: Mapping[str, Any], source_path=None) -> dict:
    """Validate and detach the closed v2.4 controller manifest."""
    control = _object(value, "control")
    allowed = {
        "schema_version",
        "project_root",
        "workload_manifest",
        "baseline_candidate",
        "budget",
        "evaluation_gate",
        "mutation",
        "probes",
        "reviewer",
        "readiness_contract",
        "analysis_contract",
    }
    schema_version = control.get("schema_version")
    if schema_version not in {CONTROL_SCHEMA_V1, CONTROL_SCHEMA_V2}:
        raise ValidationError(
            f"schema_version must be {CONTROL_SCHEMA_V1} or {CONTROL_SCHEMA_V2}"
        )
    required = allowed - {
        "reviewer",
        "evaluation_gate",
        "readiness_contract",
        "analysis_contract",
    }
    if schema_version == CONTROL_SCHEMA_V2:
        required.add("readiness_contract")
    _closed(control, allowed, "control")
    _required(control, required, "control")
    if schema_version == CONTROL_SCHEMA_V1 and "readiness_contract" in control:
        raise ValidationError("control-v1 must not contain readiness_contract")
    if schema_version == CONTROL_SCHEMA_V1 and "analysis_contract" in control:
        raise ValidationError("control-v1 must not contain analysis_contract")

    project_root = _absolute(control["project_root"], "project_root")
    workload_manifest = _absolute(
        control["workload_manifest"], "workload_manifest"
    )
    if not _is_within(workload_manifest, project_root):
        raise ValidationError("workload_manifest must be inside project_root")
    if schema_version == CONTROL_SCHEMA_V2:
        readiness_contract = _absolute(
            control["readiness_contract"], "readiness_contract"
        )
        if not _is_within(readiness_contract, project_root):
            raise ValidationError(
                "readiness_contract must be inside project_root"
            )
        if "analysis_contract" in control:
            analysis_contract = _absolute(
                control["analysis_contract"], "analysis_contract"
            )
            if not _is_within(analysis_contract, project_root):
                raise ValidationError(
                    "analysis_contract must be inside project_root"
                )
    baseline = _object(control["baseline_candidate"], "baseline_candidate")
    if not baseline:
        raise ValidationError("baseline_candidate must not be empty")
    _json_copy(baseline, "baseline_candidate", reject_sensitive=True)
    if control["budget"] not in _BUDGETS:
        raise ValidationError("budget must be quick, balanced, or thorough")
    if control.get("evaluation_gate", "promotion") not in {
        "promotion",
        "reject_only",
    }:
        raise ValidationError(
            "evaluation_gate must be promotion or reject_only"
        )

    mutation = _object(control["mutation"], "mutation")
    mutation_fields = {"project_paths", "environment_root", "host_policy"}
    _closed(mutation, mutation_fields, "mutation")
    _required(mutation, mutation_fields, "mutation")
    project_paths = _string_list(mutation["project_paths"], "project_paths")
    normalized_roots = []
    for index, item in enumerate(project_paths):
        relative = _relative(item, f"project_paths[{index}]")
        candidate = (project_root / relative).resolve(strict=False)
        if not _is_within(candidate, project_root):
            raise ValidationError(f"project_paths[{index}] escapes project_root")
        normalized_roots.append(relative)
    for index, root in enumerate(normalized_roots):
        for other in normalized_roots[index + 1 :]:
            if root == other or _is_within(root, other) or _is_within(other, root):
                raise ValidationError("project_paths must not overlap")
    environment_root = _absolute(mutation["environment_root"], "environment_root")
    protected_environment_roots = {
        Path(path).resolve(strict=False)
        for path in (
            "/System",
            "/Library",
            "/Applications",
            "/bin",
            "/sbin",
            "/usr",
            "/etc",
            "/private/etc",
        )
    }
    if (
        _is_within(environment_root, project_root)
        or _is_within(project_root, environment_root)
        or environment_root == Path("/")
        or any(_is_within(environment_root, root) for root in protected_environment_roots)
    ):
        raise ValidationError(
            "environment_root must be isolated from project_root and host system roots"
        )
    allowed_workspace_roots = [
        Path(tempfile.gettempdir()).resolve(strict=False),
        Path("/workspace").resolve(strict=False),
        Path("/data").resolve(strict=False),
    ]
    if os.geteuid() != 0:
        allowed_workspace_roots.append(Path.home().resolve(strict=False))
    if not any(
        environment_root != root and _is_within(environment_root, root)
        for root in allowed_workspace_roots
    ):
        raise ValidationError(
            "environment_root must be below a user workspace, data, or temporary root"
        )
    if mutation["host_policy"] != "recommend_only":
        raise ValidationError("host_policy must be recommend_only")

    probes = control["probes"]
    if type(probes) is not list or not probes:
        raise ValidationError("probes must be a non-empty array")
    probe_ids = set()
    for index, item in enumerate(probes):
        probe = _object(item, f"probes[{index}]")
        fields = {"id", "kind", "argv", "timeout_seconds"}
        _closed(probe, fields, f"probes[{index}]")
        _required(probe, fields, f"probes[{index}]")
        probe_id = _identifier(probe["id"], f"probes[{index}].id")
        if probe_id in probe_ids:
            raise ValidationError("probe ids must be unique")
        probe_ids.add(probe_id)
        if probe["kind"] not in _PROBE_KINDS:
            raise ValidationError(f"probes[{index}].kind is unsupported")
        _argv(probe["argv"], f"probes[{index}]")
        _timeout(probe["timeout_seconds"], f"probes[{index}].timeout_seconds")

    reviewer = control.get("reviewer")
    if reviewer is not None:
        reviewer = _object(reviewer, "reviewer")
        fields = {"argv", "timeout_seconds", "include_diff"}
        _closed(reviewer, fields, "reviewer")
        _required(reviewer, {"argv", "timeout_seconds"}, "reviewer")
        _argv(reviewer["argv"], "reviewer")
        _timeout(reviewer["timeout_seconds"], "reviewer.timeout_seconds")
        if "include_diff" in reviewer and type(reviewer["include_diff"]) is not bool:
            raise ValidationError("reviewer.include_diff must be a boolean")

    normalized = _json_copy(control, "control", reject_sensitive=True)
    if normalized["budget"] == "fast":
        normalized["budget"] = "quick"
    return normalized


def validate_change_set(value: Mapping[str, Any], control: Mapping[str, Any]) -> dict:
    """Validate a bounded project or isolated-environment ChangeSet."""
    change = _object(value, "change_set")
    fields = {
        "schema_version",
        "id",
        "hypothesis",
        "diagnosis_ids",
        "scope",
        "candidate",
        "paths",
        "commands",
        "rollback",
        "expected_metrics",
    }
    _closed(change, fields, "change_set")
    _required(change, fields, "change_set")
    if change["schema_version"] != CHANGE_SCHEMA:
        raise ValidationError(f"change_set.schema_version must be {CHANGE_SCHEMA}")
    _identifier(change["id"], "change_set.id")
    _string(change["hypothesis"], "change_set.hypothesis")
    _string_list(change["diagnosis_ids"], "change_set.diagnosis_ids")
    if change["scope"] not in {"project", "isolated_environment"}:
        raise ValidationError("change_set.scope must be project or isolated_environment")
    candidate = _object(change["candidate"], "change_set.candidate")
    if not candidate:
        raise ValidationError("change_set.candidate must not be empty")
    if "_cuda_optimizer_identity_digest" in candidate:
        raise ValidationError("change_set.candidate uses a reserved identity field")
    _json_copy(candidate, "change_set.candidate", reject_sensitive=True)
    runtime = _BUDGET_RUNTIME[control["budget"]]
    gate_contract = {
        "soft_target_seconds": runtime["soft_target_seconds"],
        "hard_ceiling_seconds": runtime["hard_ceiling_seconds"],
        "minimum_effect": {"mechanism_us": 1.0, "service_pct": 0.5},
    }
    try:
        _load_budget_module().validate_candidate_declaration(candidate, gate_contract)
    except ValueError as error:
        raise ValidationError(f"candidate declaration is invalid: {error}") from error

    paths = _string_list(change["paths"], "change_set.paths")
    relative_paths = [
        _relative(item, f"change_set.paths[{index}]")
        for index, item in enumerate(paths)
    ]
    if change["scope"] == "project":
        allowed_roots = [
            _relative(item, "control.mutation.project_paths")
            for item in control["mutation"]["project_paths"]
        ]
        for index, path in enumerate(relative_paths):
            if not any(path == root or _is_within(path, root) for root in allowed_roots):
                raise ValidationError(
                    f"change_set.paths[{index}] is outside declared project_paths"
                )

    commands = change["commands"]
    if type(commands) is not list:
        raise ValidationError("change_set.commands must be an array of argv arrays")
    if commands:
        raise ValidationError(
            "change_set.commands must be empty; correctness runs through the workload adapter"
        )
    for index, command in enumerate(commands):
        if type(command) is not list:
            raise ValidationError("change_set.commands must contain argv arrays")
        _argv(command, f"change_set.commands[{index}]")
    if change["rollback"] != "restore_frozen_snapshot":
        raise ValidationError("change_set.rollback must be restore_frozen_snapshot")
    _string_list(change["expected_metrics"], "change_set.expected_metrics")
    return _json_copy(change, "change_set", reject_sensitive=True)


def _load_diagnosis_module():
    global _DIAGNOSIS_MODULE
    if _DIAGNOSIS_MODULE is not None:
        return _DIAGNOSIS_MODULE
    path = Path(__file__).with_name("workload_diagnosis.py")
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_workload_diagnosis_runtime", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load workload diagnosis module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    _DIAGNOSIS_MODULE = module
    return module


def _load_reviewer_module():
    global _REVIEWER_MODULE
    if _REVIEWER_MODULE is not None:
        return _REVIEWER_MODULE
    path = Path(__file__).with_name("workload_reviewer.py")
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_workload_reviewer_runtime", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load workload reviewer module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    _REVIEWER_MODULE = module
    return module


def _load_sibling_module(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load controller dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_workload_module():
    global _WORKLOAD_MODULE
    if _WORKLOAD_MODULE is None:
        script_dir = str(Path(__file__).resolve().parent)
        inserted = script_dir not in sys.path
        if inserted:
            sys.path.insert(0, script_dir)
        try:
            _WORKLOAD_MODULE = _load_sibling_module(
                "workload_adapter.py", "workload_adapter"
            )
        finally:
            if inserted:
                sys.path.remove(script_dir)
    return _WORKLOAD_MODULE


def _load_evaluate_module():
    global _EVALUATE_MODULE
    if _EVALUATE_MODULE is None:
        # workload_evaluate imports workload_adapter and paired_stats by module
        # name, so expose this directory only while loading the trusted sibling.
        script_dir = str(Path(__file__).resolve().parent)
        inserted = script_dir not in sys.path
        if inserted:
            sys.path.insert(0, script_dir)
        try:
            _EVALUATE_MODULE = _load_sibling_module(
                "workload_evaluate.py", "cuda_optimizer_workload_evaluate_controller"
            )
        finally:
            if inserted:
                sys.path.remove(script_dir)
    return _EVALUATE_MODULE


def _load_budget_module():
    global _BUDGET_MODULE
    if _BUDGET_MODULE is None:
        _BUDGET_MODULE = _load_sibling_module(
            "budget.py", "cuda_optimizer_budget_controller"
        )
    return _BUDGET_MODULE


def _load_readiness_contract_module():
    global _READINESS_CONTRACT_MODULE
    if _READINESS_CONTRACT_MODULE is None:
        _READINESS_CONTRACT_MODULE = _load_sibling_module(
            "readiness_contract.py",
            "cuda_optimizer_readiness_contract_controller",
        )
    return _READINESS_CONTRACT_MODULE


def _load_readiness_gate_module():
    global _READINESS_GATE_MODULE
    if _READINESS_GATE_MODULE is None:
        _READINESS_GATE_MODULE = _load_sibling_module(
            "readiness_gate.py",
            "cuda_optimizer_readiness_gate_controller",
        )
    return _READINESS_GATE_MODULE


def _load_readiness_identity_module():
    global _READINESS_IDENTITY_MODULE
    if _READINESS_IDENTITY_MODULE is None:
        _READINESS_IDENTITY_MODULE = _load_sibling_module(
            "readiness_identity.py",
            "cuda_optimizer_readiness_identity_controller",
        )
    return _READINESS_IDENTITY_MODULE


def _load_check_env_module():
    global _CHECK_ENV_MODULE
    if _CHECK_ENV_MODULE is None:
        _CHECK_ENV_MODULE = _load_sibling_module(
            "check_env.py", "cuda_optimizer_check_env_controller"
        )
    return _CHECK_ENV_MODULE


def _load_analysis_epoch_module():
    global _ANALYSIS_EPOCH_MODULE
    if _ANALYSIS_EPOCH_MODULE is None:
        _ANALYSIS_EPOCH_MODULE = _load_sibling_module(
            "analysis_epoch.py", "cuda_optimizer_analysis_epoch_controller"
        )
    return _ANALYSIS_EPOCH_MODULE


def _load_execution_map_module():
    global _EXECUTION_MAP_MODULE
    if _EXECUTION_MAP_MODULE is None:
        _EXECUTION_MAP_MODULE = _load_sibling_module(
            "execution_map.py", "cuda_optimizer_execution_map_controller"
        )
    return _EXECUTION_MAP_MODULE


def _load_hypothesis_space_module():
    global _HYPOTHESIS_SPACE_MODULE
    if _HYPOTHESIS_SPACE_MODULE is None:
        _HYPOTHESIS_SPACE_MODULE = _load_sibling_module(
            "hypothesis_space.py", "cuda_optimizer_hypothesis_space_controller"
        )
    return _HYPOTHESIS_SPACE_MODULE


def _load_evidence_selector_module():
    global _EVIDENCE_SELECTOR_MODULE
    if _EVIDENCE_SELECTOR_MODULE is None:
        _EVIDENCE_SELECTOR_MODULE = _load_sibling_module(
            "evidence_selector.py", "cuda_optimizer_evidence_selector_controller"
        )
    return _EVIDENCE_SELECTOR_MODULE


def _load_diagnostic_knowledge_module():
    return _load_sibling_module(
        "diagnostic_knowledge.py", "cuda_optimizer_diagnostic_knowledge_controller"
    )


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _run_lock(run_root: Path):
    """Serialize initialization and active-diagnosis mutations for one run."""
    lock_path = run_root / ".workload-controller.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValidationError("active diagnosis lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


class _BoundedLog:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.value = bytearray()
        self.truncated = False

    def append(self, chunk: bytes) -> None:
        available = max(0, self.limit - len(self.value))
        self.value.extend(chunk[:available])
        if len(chunk) > available:
            self.truncated = True

    def text(self) -> str:
        decoded = bytes(self.value).decode("utf-8", errors="replace")
        return decoded + ("...[truncated]" if self.truncated else "")


def _drain(stream, capture: _BoundedLog) -> None:
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            capture.append(chunk)
    finally:
        stream.close()


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_group(process: Any) -> None:
    process_group = process.pid

    try:
        os.killpg(process_group, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    deadline = time.monotonic() + 0.25
    while _process_group_exists(process_group) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.01)
    if _process_group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass


def _is_secret_name(name: str) -> bool:
    return _SENSITIVE_KEY.search(name) is not None


def _probe_environment(overrides: Mapping[str, str]) -> tuple[dict, tuple[str, ...]]:
    inherited = dict(os.environ)
    explicit = {
        name.strip()
        for name in inherited.get("CUDA_OPTIMIZER_PASS_ENV", "").split(",")
        if name.strip()
    }
    allowed = _SAFE_ENV | explicit
    environment = {
        name: value
        for name, value in inherited.items()
        if name in allowed and not _is_secret_name(name)
    }
    environment.update(overrides)
    secrets = tuple(
        value
        for name, value in inherited.items()
        if _is_secret_name(name) and value
    )
    return environment, secrets


def _redact_log(value: str, secrets: Sequence[str]) -> str:
    result = _LOG_SECRET.sub(lambda match: f"{match.group(1)}[REDACTED]", value)
    for secret in sorted(set(secrets), key=len, reverse=True):
        if secret:
            result = result.replace(secret, "[REDACTED]")
    return result


def _failure_probe(probe: Mapping[str, Any], status: str, issue_id: str, message: str) -> dict:
    return {
        "schema_version": "cuda-workload-optimizer/probe-v1",
        "probe_id": probe["id"],
        "kind": probe["kind"],
        "status": status,
        "metrics": {},
        "issues": [
            {
                "id": issue_id,
                "category": "environment",
                "severity": "error",
                "message": message,
            }
        ],
        "artifacts": [],
    }


def _read_probe_output(path: Path) -> dict:
    try:
        info = path.stat()
    except FileNotFoundError as error:
        raise ValidationError("probe did not create CUDA_OPTIMIZER_OUTPUT") from error
    if not path.is_file() or info.st_size > _OUTPUT_LIMIT:
        raise ValidationError(f"probe output must be a regular file under {_OUTPUT_LIMIT} bytes")
    return load_json_object(path)


def run_probe(
    probe: Mapping[str, Any],
    control: Mapping[str, Any],
    run_dir: os.PathLike[str] | str,
    *,
    log_limit_bytes: int = _DEFAULT_LOG_LIMIT,
    deadline_epoch: float | None = None,
) -> dict:
    """Execute one argv-only probe and persist normalized evidence plus bounded logs."""
    if isinstance(log_limit_bytes, bool) or not isinstance(log_limit_bytes, int):
        raise ValidationError("log_limit_bytes must be a positive integer")
    if log_limit_bytes <= 0 or log_limit_bytes > _OUTPUT_LIMIT:
        raise ValidationError("log_limit_bytes must be between 1 and 1048576")
    normalized_control = validate_control_manifest(control)
    matching = [item for item in normalized_control["probes"] if item["id"] == probe.get("id")]
    if len(matching) != 1 or matching[0] != probe:
        raise ValidationError("probe must exactly match one validated control probe")
    selected = matching[0]
    actual_timeout = float(selected["timeout_seconds"])
    if deadline_epoch is not None:
        remaining = float(deadline_epoch) - time.time()
        if remaining <= 0:
            raise ValidationError("workload optimization budget deadline has expired")
        actual_timeout = min(actual_timeout, remaining)
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    probes_dir = run_root / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)
    output_path = probes_dir / f".{selected['id']}.output.json"
    active_output_path = None
    environment_overrides = {
        "CUDA_OPTIMIZER_OUTPUT": str(output_path),
        "CUDA_OPTIMIZER_RUN_DIR": str(run_root),
        "CUDA_OPTIMIZER_PROJECT_ROOT": normalized_control["project_root"],
    }
    if "analysis_contract" in normalized_control:
        frozen_analysis_contract = (
            run_root / "active_diagnosis" / "analysis_contract.json"
        )
        contract_path = (
            frozen_analysis_contract
            if frozen_analysis_contract.is_file()
            else Path(normalized_control["analysis_contract"])
        )
        active_contract = _validate_active_diagnosis_contract(
            load_json_object(contract_path)
        )
        if selected["id"] == active_contract["global_scan_probe_id"]:
            active_output_path = probes_dir / ".active-diagnosis.output.json"
            environment_overrides["CUDA_OPTIMIZER_ACTIVE_DIAGNOSIS_OUTPUT"] = str(
                active_output_path
            )
            if frozen_analysis_contract.is_file():
                state = read_run_state(run_root)
                bindings = _load_frozen_execution_bindings(run_root, state)
                _verify_adapter_execution_binding(
                    bindings["global_scan"],
                    Path(active_contract["adapter_path"]),
                    selected["argv"],
                    "analysis_contract global scan",
                )
    for transient in (output_path, active_output_path):
        if transient is None:
            continue
        try:
            transient.unlink()
        except FileNotFoundError:
            pass
    environment, secret_values = _probe_environment(environment_overrides)
    stdout = _BoundedLog(log_limit_bytes)
    stderr = _BoundedLog(log_limit_bytes)
    started = time.monotonic()
    timed_out = False
    exit_code = None
    process = None
    try:
        process = subprocess.Popen(
            selected["argv"],
            cwd=normalized_control["project_root"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        readers = [
            threading.Thread(target=_drain, args=(process.stdout, stdout), daemon=True),
            threading.Thread(target=_drain, args=(process.stderr, stderr), daemon=True),
        ]
        for reader in readers:
            reader.start()
        try:
            exit_code = process.wait(timeout=actual_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_group(process)
            exit_code = process.returncode
        else:
            if _process_group_exists(process.pid):
                _stop_group(process)
        for reader in readers:
            reader.join(timeout=1)
    except FileNotFoundError as error:
        result = _failure_probe(
            selected, "unavailable", "environment:probe-unavailable", str(error)
        )
    except OSError as error:
        result = _failure_probe(
            selected, "failed", "environment:probe-launch", str(error)
        )
    else:
        if timed_out:
            result = _failure_probe(
                selected,
                "unavailable",
                "environment:probe-timeout",
                f"probe exceeded {actual_timeout:.6g} seconds",
            )
        elif exit_code != 0:
            result = _failure_probe(
                selected,
                "failed",
                "environment:probe-exit",
                f"probe exited with status {exit_code}",
            )
        else:
            try:
                result = _load_diagnosis_module().validate_probe(
                    _read_probe_output(output_path)
                )
                if result["probe_id"] != selected["id"] or result["kind"] != selected["kind"]:
                    raise ValidationError("probe output identity does not match control")
            except (ValidationError, ValueError) as error:
                result = _failure_probe(
                    selected,
                    "failed",
                    "environment:probe-output",
                    f"invalid normalized probe output: {error}",
                )
    finally:
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass

    if active_output_path is not None and result.get("status") in {"ok", "degraded"}:
        try:
            active_draft = _read_probe_output(active_output_path)
            _atomic_json(
                run_root / "active_diagnosis" / "global_scan.json",
                active_draft,
            )
        except (ValidationError, ValueError) as error:
            result = _failure_probe(
                selected,
                "failed",
                "environment:active-diagnosis-output",
                f"invalid active diagnosis output: {error}",
            )
    if active_output_path is not None:
        try:
            active_output_path.unlink()
        except FileNotFoundError:
            pass

    result = _load_diagnosis_module().validate_probe(result)
    duration = time.monotonic() - started
    execution = {
        "schema_version": "cuda-workload-optimizer/probe-execution-v1",
        "probe_id": selected["id"],
        "argv_sha256": _canonical_digest(selected["argv"]),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_seconds": duration,
        "stdout": _redact_log(stdout.text(), secret_values),
        "stderr": _redact_log(stderr.text(), secret_values),
        "stdout_truncated": stdout.truncated,
        "stderr_truncated": stderr.truncated,
    }
    _atomic_json(probes_dir / f"{selected['id']}.json", result)
    _atomic_json(probes_dir / f"{selected['id']}.execution.json", execution)
    return result


def run_probes(
    control: Mapping[str, Any],
    run_dir: os.PathLike[str] | str,
    *,
    deadline_epoch: float | None = None,
) -> list[dict]:
    normalized = validate_control_manifest(control)
    return [
        run_probe(
            probe,
            normalized,
            run_dir,
            deadline_epoch=deadline_epoch,
        )
        for probe in normalized["probes"]
    ]


def diagnose_run(run_dir: os.PathLike[str] | str) -> dict:
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    probes_dir = run_root / "probes"
    values = []
    for path in sorted(probes_dir.glob("*.json")):
        if path.name.endswith(".execution.json"):
            continue
        values.append(load_json_object(path))
    policy_path = Path(__file__).resolve().parents[1] / "references" / "workload_diagnosis_policy.json"
    diagnosis_module = _load_diagnosis_module()
    result = diagnosis_module.diagnose(values, diagnosis_module.load_policy(policy_path))
    _atomic_json(run_root / "diagnosis.json", result)
    return result


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise ValidationError(f"cannot hash artifact {path}: {error}") from error
    return digest.hexdigest()


def review_change(
    control: Mapping[str, Any],
    run_dir: os.PathLike[str] | str,
    change_set: Mapping[str, Any],
    *,
    deadline_epoch: float | None = None,
) -> dict:
    normalized = validate_control_manifest(control)
    change = validate_change_set(change_set, normalized)
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    diagnosis_path = run_root / "diagnosis.json"
    diagnosis = load_json_object(diagnosis_path)
    diff_path = run_root / "candidate.diff"
    redacted_diff = "Diff content withheld; set reviewer.include_diff=true to opt in."
    if diff_path.exists():
        if diff_path.stat().st_size > 256 * 1024:
            raise ValidationError("candidate.diff exceeds reviewer request limit")
        if normalized.get("reviewer", {}).get("include_diff", False):
            redacted_diff = _redact_log(diff_path.read_text("utf-8"), ())
        else:
            redacted_diff += f" sha256={_sha256_path(diff_path)}"
    change_path = run_root / "change_set.json"
    _atomic_json(change_path, change)
    reviewer = _load_reviewer_module()
    blocks = {"quick": 3, "balanced": 5, "thorough": 9}[normalized["budget"]]
    request = reviewer.build_review_request(
        diagnosis=diagnosis,
        change_set=change,
        redacted_diff=redacted_diff,
        experiment={
            "blocks": blocks,
            "evaluation": "paired_ab_ba",
            "expected_metrics": change["expected_metrics"],
        },
        artifact_hashes={
            "diagnosis.json": _sha256_path(diagnosis_path),
            "change_set.json": _sha256_path(change_path),
        },
    )
    _atomic_json(run_root / "review_request.json", request)
    if "reviewer" not in normalized:
        return reviewer.write_skipped_review(request, run_root)
    reviewer_config = {
        "argv": normalized["reviewer"]["argv"],
        "timeout_seconds": normalized["reviewer"]["timeout_seconds"],
    }
    if deadline_epoch is not None:
        remaining = float(deadline_epoch) - time.time()
        if remaining < 1:
            return reviewer.write_skipped_review(request, run_root)
        reviewer_config["timeout_seconds"] = min(
            float(reviewer_config["timeout_seconds"]), remaining
        )
    return reviewer.run_reviewer(reviewer_config, request, run_root)


def _scope_layout(control: Mapping[str, Any], scope: str) -> tuple[Path, list[str], str]:
    if scope == "project":
        return (
            Path(control["project_root"]),
            list(control["mutation"]["project_paths"]),
            "project",
        )
    if scope == "isolated_environment":
        return Path(control["mutation"]["environment_root"]), ["."], "environment"
    raise ValidationError("unsupported ChangeSet scope")


def _identity(control: Mapping[str, Any], scope: str) -> dict:
    base, roots, _snapshot_name = _scope_layout(control, scope)
    files = {}
    missing_roots = []
    for relative_root in roots:
        root = base if relative_root == "." else base / relative_root
        if not root.exists():
            missing_roots.append(relative_root)
            continue
        if relative_root == "." and not root.is_dir():
            raise ValidationError("environment_root must be a directory")
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in candidates:
            if path.is_symlink():
                raise ValidationError(f"mutation root contains a symlink: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(base).as_posix()
            files[relative] = {
                "sha256": _sha256_path(path),
                "size_bytes": path.stat().st_size,
                "mode": path.stat().st_mode & 0o777,
            }
    return {
        "schema_version": "cuda-workload-optimizer/project-identity-v1",
        "scope": scope,
        "roots": roots,
        "missing_roots": sorted(missing_roots),
        "files": files,
        "digest": _canonical_digest(
            {"missing_roots": sorted(missing_roots), "files": files}
        ),
    }


def _project_surface_identity(project_root: Path) -> dict:
    root = project_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValidationError("project_root must be a non-symlink directory")
    entries = {}
    excluded_directories = {".git", ".worktrees", "__pycache__"}
    try:
        for current, raw_directories, raw_files in os.walk(root, followlinks=False):
            current_path = Path(current)
            directories = []
            for name in sorted(raw_directories):
                path = current_path / name
                relative = path.relative_to(root).as_posix()
                if name in excluded_directories:
                    continue
                if path.is_symlink():
                    metadata = path.lstat()
                    entries[relative] = {
                        "type": "symlink",
                        "target": os.readlink(path),
                        "mode": metadata.st_mode & 0o777,
                    }
                else:
                    directories.append(name)
            raw_directories[:] = directories
            for name in sorted(raw_files):
                if name.endswith(".pyc"):
                    continue
                path = current_path / name
                relative = path.relative_to(root).as_posix()
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    entries[relative] = {
                        "type": "symlink",
                        "target": os.readlink(path),
                        "mode": metadata.st_mode & 0o777,
                    }
                elif stat.S_ISREG(metadata.st_mode):
                    entries[relative] = {
                        "type": "file",
                        "size_bytes": metadata.st_size,
                        "mode": metadata.st_mode & 0o777,
                        "mtime_ns": metadata.st_mtime_ns,
                        "sha256": _sha256_path(path),
                    }
                else:
                    entries[relative] = {
                        "type": "other",
                        "mode": metadata.st_mode,
                    }
    except OSError as error:
        raise ValidationError(f"cannot identify the complete project surface: {error}") from error
    return {
        "schema_version": "cuda-workload-optimizer/project-surface-identity-v1",
        "entries": entries,
        "digest": _canonical_digest(entries),
    }


def _snapshot_scope(
    control: Mapping[str, Any], run_root: Path, scope: str
) -> dict:
    base, roots, snapshot_name = _scope_layout(control, scope)
    snapshot = run_root / "snapshot" / snapshot_name
    if snapshot.exists():
        raise ValidationError("frozen ChangeSet snapshot already exists")
    identity = _identity(control, scope)
    if scope == "isolated_environment":
        if identity["missing_roots"]:
            raise ValidationError("environment_root must exist before registration")
        if base.is_symlink() or base.stat().st_uid != os.getuid():
            raise ValidationError(
                "environment_root must be a user-owned non-symlink directory"
            )
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(base, snapshot, symlinks=False)
        _atomic_json(
            run_root / "rounds" / "round-1" / "before_identity.json", identity
        )
        return identity
    snapshot.mkdir(parents=True)
    for relative_root in roots:
        source = base / relative_root
        destination = snapshot / relative_root
        if not source.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination, symlinks=False)
        else:
            shutil.copy2(source, destination)
    _atomic_json(run_root / "rounds" / "round-1" / "before_identity.json", identity)
    return identity


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _restore_snapshot(
    control: Mapping[str, Any],
    run_root: Path,
    scope: str,
    expected_identity_digest: str,
) -> None:
    base, roots, snapshot_name = _scope_layout(control, scope)
    snapshot = run_root / "snapshot" / snapshot_name
    if snapshot.is_symlink():
        raise ValidationError("frozen snapshot must not be a symlink")
    snapshot_control = copy.deepcopy(control)
    if scope == "project":
        snapshot_control["project_root"] = str(snapshot)
    else:
        snapshot_control["mutation"]["environment_root"] = str(snapshot)
    snapshot_identity = _identity(snapshot_control, scope)
    if snapshot_identity["digest"] != expected_identity_digest:
        raise ValidationError("frozen snapshot identity does not match registration")
    if scope == "isolated_environment":
        if base.exists() or base.is_symlink():
            _remove_path(base)
        base.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(snapshot, base, symlinks=False)
        return
    for relative_root in roots:
        current = base / relative_root
        frozen = snapshot / relative_root
        if current.exists() or current.is_symlink():
            _remove_path(current)
        if not frozen.exists():
            continue
        current.parent.mkdir(parents=True, exist_ok=True)
        if frozen.is_dir():
            shutil.copytree(frozen, current, symlinks=False)
        else:
            shutil.copy2(frozen, current)


def _changed_paths(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    names = set(before["files"]) | set(after["files"])
    return sorted(
        name for name in names if before["files"].get(name) != after["files"].get(name)
    )


def _path_allowed(relative: str, allowed: Sequence[str]) -> bool:
    path = Path(relative)
    return any(
        path == Path(root) or _is_within(path, Path(root))
        for root in allowed
    )


def _candidate_diff(
    control: Mapping[str, Any],
    run_root: Path,
    changed: Sequence[str],
    scope: str,
) -> str:
    base, _roots, snapshot_name = _scope_layout(control, scope)
    snapshot = run_root / "snapshot" / snapshot_name
    chunks = []
    for relative in changed:
        before_path = snapshot / relative
        after_path = base / relative
        try:
            before = before_path.read_text("utf-8").splitlines(keepends=True) if before_path.exists() else []
            after = after_path.read_text("utf-8").splitlines(keepends=True) if after_path.exists() else []
        except (OSError, UnicodeError):
            before_hash = _sha256_path(before_path) if before_path.exists() else "missing"
            after_hash = _sha256_path(after_path) if after_path.exists() else "missing"
            chunks.append(f"binary {relative}: {before_hash} -> {after_hash}\n")
            continue
        chunks.extend(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    return _redact_log("".join(chunks), ())


def read_run_state(run_dir: os.PathLike[str] | str) -> dict:
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    commit = load_json_object(run_root / "state_commit.json")
    if set(commit) != {"schema_version", "state_digest"} or commit.get(
        "schema_version"
    ) != "cuda-workload-optimizer/state-commit-v1":
        raise ValidationError("state commit marker is invalid")
    digest = commit.get("state_digest")
    if type(digest) is not str or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
        raise ValidationError("state commit digest is invalid")
    state = load_json_object(run_root / "state_generations" / f"{digest}.json")
    if _canonical_digest(state) != digest:
        raise ValidationError("committed state generation digest is invalid")
    for name in ("state.json", "checkpoint.json"):
        path = run_root / name
        try:
            mirror = load_json_object(path)
        except (OSError, ValidationError):
            mirror = None
        if mirror != state:
            _atomic_json(path, state)
    return state


def _write_state(run_root: Path, state: Mapping[str, Any]) -> dict:
    detached = _json_copy(state, "state")
    digest = _canonical_digest(detached)
    _atomic_json(run_root / "state_generations" / f"{digest}.json", detached)
    _atomic_json(
        run_root / "state_commit.json",
        {
            "schema_version": "cuda-workload-optimizer/state-commit-v1",
            "state_digest": digest,
        },
    )
    _atomic_json(run_root / "state.json", detached)
    _atomic_json(run_root / "checkpoint.json", detached)
    return detached


def _advance(
    run_root: Path,
    state: Mapping[str, Any],
    completed: str,
    *,
    stage: str,
    next_action: str,
) -> dict:
    updated = copy.deepcopy(state)
    if completed not in updated["completed_stages"]:
        updated["completed_stages"].append(completed)
    updated["stage"] = stage
    updated["next_action"] = next_action
    updated["updated_at_epoch"] = time.time()
    return _write_state(run_root, updated)


def _load_frozen_control(run_root: Path, state: Mapping[str, Any] | None = None) -> dict:
    active_state = read_run_state(run_root) if state is None else state
    control = validate_control_manifest(
        load_json_object(run_root / "control_manifest.json")
    )
    if _canonical_digest(control) != active_state.get("control_digest"):
        raise ValidationError("frozen control manifest digest does not match state")
    return control


def _normalize_frozen_workload(control: Mapping[str, Any]):
    try:
        return _load_workload_module().normalize_workload(
            workload_manifest=control["workload_manifest"]
        )
    except (OSError, ValueError) as error:
        raise ValidationError(f"workload identity validation failed: {error}") from error


def _check_deadline(state: Mapping[str, Any]) -> None:
    if time.time() > state["deadline_epoch"]:
        raise ValidationError("workload optimization budget deadline has expired")


def _current_readiness_identity(control: Mapping[str, Any]) -> dict:
    environment_root = Path(control["mutation"]["environment_root"])
    if not environment_root.is_dir() or environment_root.is_symlink():
        raise ValidationError(
            "control-v2 environment_root must be an existing non-symlink directory"
        )
    inventory = _load_check_env_module().collect_identity_inventory()
    return _load_readiness_identity_module().build_identity(
        environment_root=environment_root,
        inventory=inventory,
        run=_load_check_env_module()._run,
    )


def _load_frozen_readiness_contract(
    control: Mapping[str, Any], run_root: Path, state: Mapping[str, Any]
) -> dict:
    module = _load_readiness_contract_module()
    path = run_root / "readiness_contract.json"
    value = module.load_contract(path)
    validated = module.validate_contract(
        value,
        project_root=Path(control["project_root"]),
        environment_root=Path(control["mutation"]["environment_root"]),
    )
    if module.contract_digest(validated) != state.get(
        "readiness_contract_digest"
    ):
        raise ValidationError("frozen readiness contract digest does not match state")
    return validated


def _run_readiness_gate(
    control: Mapping[str, Any], run_root: Path, state: Mapping[str, Any]
) -> dict:
    gate = _load_readiness_gate_module()
    contract = _load_frozen_readiness_contract(control, run_root, state)
    identity = _current_readiness_identity(control)
    return gate.run_gate(
        contract=contract,
        control={
            "project_root": control["project_root"],
            "environment_root": control["mutation"]["environment_root"],
            "environment_identity": identity,
        },
        run_dir=run_root,
        identity_provider=lambda: _current_readiness_identity(control),
    )


def _run_readiness_gate_checked(
    control: Mapping[str, Any], run_root: Path, state: Mapping[str, Any]
) -> dict:
    surface_before = (
        _project_surface_identity(Path(control["project_root"]))
        if "analysis_contract" in control
        else None
    )
    report = _run_readiness_gate(control, run_root, state)
    if (
        surface_before is not None
        and _project_surface_identity(Path(control["project_root"]))
        != surface_before
    ):
        raise ValidationError("readiness modified the complete project surface")
    return report


def _readiness_report_digest(run_root: Path, report: Mapping[str, Any]) -> str:
    path = run_root / "readiness" / "report.json"
    return _sha256_path(path) if path.is_file() else _canonical_digest(report)


def _verify_readiness_report(
    control: Mapping[str, Any], run_root: Path, state: Mapping[str, Any]
) -> bool:
    gate = _load_readiness_gate_module()
    try:
        report = gate._load_prior_report(run_root / "readiness")
    except ValueError as error:
        raise ValidationError(f"readiness report verification failed: {error}") from error
    if report is None:
        raise ValidationError("completed readiness stage is missing its report")
    if _readiness_report_digest(run_root, report) != state.get(
        "readiness_report_digest"
    ):
        raise ValidationError("readiness report digest does not match state")
    contract = _load_frozen_readiness_contract(control, run_root, state)
    contract_digest = _load_readiness_contract_module().contract_digest(contract)
    if report.get("contract_digest") != contract_digest:
        raise ValidationError("readiness report contract digest drifted")
    identity_digest = gate.environment_identity_digest(
        _current_readiness_identity(control)
    )
    if report.get("environment_identity_digest") != identity_digest:
        return False
    if not report.get("can_start_diagnosis"):
        return False
    now = time.time()
    required_ids = {
        item["id"] for item in contract["requirements"] if item["necessity"] == "required"
    }
    ready_ids = {
        item.get("requirement_id")
        for item in report.get("results", [])
        if type(item) is dict
        and item.get("necessity") == "required"
        and item.get("admission_status") == "ready"
        and isinstance(item.get("valid_until"), (int, float))
        and not isinstance(item.get("valid_until"), bool)
        and math.isfinite(float(item["valid_until"]))
        and float(item["valid_until"]) > now
    }
    return ready_ids == required_ids


def _load_frozen_analysis_contract(run_root: Path, state: Mapping[str, Any]) -> dict:
    contract = _validate_active_diagnosis_contract(
        load_json_object(run_root / "active_diagnosis" / "analysis_contract.json")
    )
    if _canonical_digest(contract) != state.get("analysis_contract_digest"):
        raise ValidationError("frozen analysis contract digest does not match state")
    return contract


def _adapter_execution_binding(
    adapter_path: Path, argv: Sequence[str], field: str
) -> dict:
    adapter = adapter_path.resolve(strict=True)
    adapter_text = str(adapter_path)
    if argv[0] == adapter_text:
        if not os.access(adapter, os.X_OK):
            raise ValidationError(f"{field} direct adapter must be executable")
        launcher = adapter
        mode = "direct"
        adapter_arg_index = 0
    elif len(argv) >= 2 and argv[1] == adapter_text:
        raw_launcher = Path(argv[0]).expanduser()
        if not raw_launcher.is_absolute():
            located = shutil.which(argv[0])
            if located is None:
                raise ValidationError(f"{field} interpreter cannot be resolved")
            raw_launcher = Path(located)
        launcher = raw_launcher.resolve(strict=True)
        basename = launcher.name.lower()
        suffix = adapter.suffix.lower()
        python_launcher = re.fullmatch(r"python(?:3(?:\.\d+)*)?", basename) is not None
        shell_launcher = basename in {"sh", "bash"}
        if not ((suffix == ".py" and python_launcher) or (suffix == ".sh" and shell_launcher)):
            raise ValidationError(
                f"{field} must execute the adapter directly or through a matching Python/shell interpreter"
            )
        mode = "interpreter"
        adapter_arg_index = 1
    else:
        raise ValidationError(
            f"{field} must place adapter_path at argv[0], or argv[1] after a matching interpreter"
        )
    if not launcher.is_file():
        raise ValidationError(f"{field} launcher must be a regular file")
    return {
        "adapter_path": adapter_text,
        "adapter_sha256": _sha256_path(adapter),
        "launcher_path": str(launcher),
        "launcher_sha256": _sha256_path(launcher),
        "mode": mode,
        "adapter_arg_index": adapter_arg_index,
    }


def _load_frozen_execution_bindings(
    run_root: Path, state: Mapping[str, Any]
) -> dict:
    bindings = load_json_object(
        run_root / "active_diagnosis" / "execution_bindings.json"
    )
    if _canonical_digest(bindings) != state.get("analysis_execution_bindings_digest"):
        raise ValidationError("frozen analysis execution bindings drifted from state")
    return bindings


def _verify_adapter_execution_binding(
    expected: Mapping[str, Any], adapter_path: Path, argv: Sequence[str], field: str
) -> None:
    actual = _adapter_execution_binding(adapter_path, argv, field)
    if actual != expected:
        raise ValidationError(f"{field} adapter or launcher identity drifted")


def _ready_capability_ids(report: Mapping[str, Any], *, now: float | None = None) -> list[str]:
    current = time.time() if now is None else float(now)
    identity_digest = report.get("environment_identity_digest")
    return sorted(
        item["requirement_id"]
        for item in report.get("results", [])
        if type(item) is dict
        and item.get("admission_status") == "ready"
        and item.get("identity_digest") == identity_digest
        and type(item.get("valid_until")) in {int, float}
        and float(item["valid_until"]) > current
    )


def _validate_global_scan_draft(value: Mapping[str, Any]) -> dict:
    draft = _object(value, "global_scan")
    fields = {
        "schema_version",
        "regime",
        "boundary_ambiguous",
        "window",
        "coverage",
        "nodes",
        "edges",
        "hot_path",
        "uncovered_intervals",
        "conclusion_level",
    }
    _closed(draft, fields, "global_scan")
    _required(draft, fields, "global_scan")
    if draft["schema_version"] != _GLOBAL_SCAN_DRAFT_SCHEMA:
        raise ValidationError(
            f"global_scan.schema_version must be {_GLOBAL_SCAN_DRAFT_SCHEMA}"
        )
    regime = _object(draft["regime"], "global_scan.regime")
    regime_fields = {
        "shape_distribution_sha256",
        "dynamic_branch_sha256",
        "execution_regime_sha256",
    }
    _closed(regime, regime_fields, "global_scan.regime")
    _required(regime, regime_fields, "global_scan.regime")
    for field in regime_fields:
        _sha256(regime[field], f"global_scan.regime.{field}")
    if type(draft["boundary_ambiguous"]) is not bool:
        raise ValidationError("global_scan.boundary_ambiguous must be a boolean")
    window = _object(draft["window"], "global_scan.window")
    _closed(window, {"start_us", "end_us"}, "global_scan.window")
    _required(window, {"start_us", "end_us"}, "global_scan.window")
    for field in ("start_us", "end_us"):
        value = window[field]
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            raise ValidationError(f"global_scan.window.{field} must be finite")
    if float(window["start_us"]) < 0 or float(window["end_us"]) <= float(
        window["start_us"]
    ):
        raise ValidationError("global_scan window must be positive")
    return _json_copy(draft, "global_scan", reject_sensitive=True)


def _verify_active_diagnosis_ledger(run_root: Path) -> list[dict]:
    ledger_dir = run_root / "active_diagnosis" / "ledger"
    if not ledger_dir.exists():
        return []
    events = []
    previous = None
    for expected_sequence, path in enumerate(sorted(ledger_dir.glob("*.json")), 1):
        event = load_json_object(path)
        fields = {
            "schema_version",
            "sequence",
            "event_type",
            "previous_event_sha256",
            "payload_sha256",
            "created_at_epoch",
        }
        _closed(event, fields, f"active diagnosis ledger event {path.name}")
        _required(event, fields, f"active diagnosis ledger event {path.name}")
        if event["schema_version"] != "cuda-optimizer/active-diagnosis-event-v1":
            raise ValidationError("active diagnosis ledger schema is invalid")
        if event["sequence"] != expected_sequence:
            raise ValidationError("active diagnosis ledger sequence is not contiguous")
        _identifier(event["event_type"], "active diagnosis event_type")
        if event["previous_event_sha256"] != previous:
            raise ValidationError("active diagnosis ledger hash chain is invalid")
        _sha256(event["payload_sha256"], "active diagnosis payload_sha256")
        created = event["created_at_epoch"]
        if type(created) not in {int, float} or not math.isfinite(float(created)):
            raise ValidationError("active diagnosis event time must be finite")
        previous = _canonical_digest(event)
        events.append(event)
    return events


def _active_ledger_binding(events: Sequence[Mapping[str, Any]]) -> dict:
    if not events:
        raise ValidationError("active diagnosis ledger is empty")
    return {
        "active_diagnosis_ledger_sequence": len(events),
        "active_diagnosis_ledger_head_sha256": _canonical_digest(events[-1]),
    }


def _verify_committed_active_ledger(
    state: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> None:
    sequence = state.get("active_diagnosis_ledger_sequence")
    head = state.get("active_diagnosis_ledger_head_sha256")
    if type(sequence) is not int or sequence < 1 or type(head) is not str:
        raise ValidationError("run state is missing its active diagnosis ledger binding")
    if len(events) < sequence:
        raise ValidationError("committed active diagnosis ledger tail is missing")
    if _canonical_digest(events[sequence - 1]) != head:
        raise ValidationError("committed active diagnosis ledger head drifted")


def _append_active_diagnosis_event(
    run_root: Path, event_type: str, payload: Mapping[str, Any]
) -> dict:
    event_type = _identifier(event_type, "active diagnosis event_type")
    payload_sha = _canonical_digest(_json_copy(payload, "active diagnosis payload"))
    events = _verify_active_diagnosis_ledger(run_root)
    if events and events[-1]["event_type"] == event_type and events[-1][
        "payload_sha256"
    ] == payload_sha:
        return events[-1]
    sequence = len(events) + 1
    event = {
        "schema_version": "cuda-optimizer/active-diagnosis-event-v1",
        "sequence": sequence,
        "event_type": event_type,
        "previous_event_sha256": (
            None if not events else _canonical_digest(events[-1])
        ),
        "payload_sha256": payload_sha,
        "created_at_epoch": time.time(),
    }
    path = (
        run_root
        / "active_diagnosis"
        / "ledger"
        / f"{sequence:06d}-{event_type}.json"
    )
    if path.exists():
        raise ValidationError("active diagnosis ledger event already exists")
    _atomic_json(path, event)
    return event


def _build_active_diagnosis_context(
    control: Mapping[str, Any], run_root: Path, state: Mapping[str, Any]
) -> dict:
    contract = _load_frozen_analysis_contract(run_root, state)
    active_root = run_root / "active_diagnosis"
    scan_path = active_root / "global_scan.json"
    draft = _validate_global_scan_draft(load_json_object(scan_path))
    identities = {
        "workload_contract_sha256": _sha256_path(Path(control["workload_manifest"])),
        "environment_sha256": _sha256(
            state.get("baseline_environment_identity_digest"),
            "baseline environment identity",
        ),
        "source_sha256": _sha256(
            state.get("baseline_identity_digest"), "baseline source identity"
        ),
        "analysis_policy_sha256": contract["analysis_policy_sha256"],
    }
    epoch_seed = {
        "identities": identities,
        "source": contract["source"],
        "regime": draft["regime"],
        "boundary_ambiguous": draft["boundary_ambiguous"],
    }
    epoch_id = f"epoch-{_canonical_digest(epoch_seed)[:16]}"
    epoch = {
        "schema_version": "cuda-optimizer/analysis-epoch-v1",
        "epoch_id": epoch_id,
        "sequence": 1,
        "trigger": "initial",
        "parent_epoch_id": None,
        "started_at": state["started_at_epoch"],
        "identities": identities,
        "source": copy.deepcopy(contract["source"]),
        "regime": copy.deepcopy(draft["regime"]),
        "boundary_ambiguous": draft["boundary_ambiguous"],
    }
    epoch_module = _load_analysis_epoch_module()
    try:
        epoch = epoch_module.validate_epoch(epoch, expected_identities=identities)
    except ValueError as error:
        raise ValidationError(f"invalid Controller analysis epoch: {error}") from error
    epoch_sha = epoch_module.epoch_digest(epoch)
    evidence_catalog = {
        "ev-global-scan": {
            "epoch_id": epoch_id,
            "kind": "nsys_timeline" if contract["source"]["profiler"] == "nsys" else "global_scan",
            "artifact_sha256": _sha256_path(scan_path),
        }
    }
    execution_map = {
        "schema_version": "cuda-optimizer/execution-map-v1",
        "map_id": f"map-{epoch_id.removeprefix('epoch-')}",
        "epoch_id": epoch_id,
        "epoch_sha256": epoch_sha,
        "identities": copy.deepcopy(identities),
        "window": {
            **copy.deepcopy(draft["window"]),
            "boundary_ambiguous": draft["boundary_ambiguous"],
        },
        "coverage": copy.deepcopy(draft["coverage"]),
        "nodes": copy.deepcopy(draft["nodes"]),
        "edges": copy.deepcopy(draft["edges"]),
        "hot_path": copy.deepcopy(draft["hot_path"]),
        "uncovered_intervals": copy.deepcopy(draft["uncovered_intervals"]),
        "conclusion_level": draft["conclusion_level"],
    }
    map_module = _load_execution_map_module()
    try:
        map_result = map_module.validate_execution_map(
            execution_map, epoch=epoch, evidence_catalog=evidence_catalog
        )
    except ValueError as error:
        raise ValidationError(f"invalid global scan execution map: {error}") from error
    execution_map = map_result["execution_map"]
    action_catalog = load_json_object(
        Path(__file__).resolve().parents[1]
        / "references"
        / "evidence_action_catalog.json"
    )
    enabled_action_ids = {item["action_id"] for item in contract["actions"]}
    action_catalog["actions"] = [
        item
        for item in action_catalog["actions"]
        if item.get("action_id") in enabled_action_ids
    ]
    if not action_catalog["actions"]:
        raise ValidationError("analysis contract enables no catalog evidence action")
    selection_policy = copy.deepcopy(contract["selection_policy"])
    readiness_report = _load_readiness_gate_module()._load_prior_report(
        run_root / "readiness"
    )
    if readiness_report is None:
        raise ValidationError("active diagnosis requires a completed readiness report")
    selection_policy["available_capability_ids"] = _ready_capability_ids(
        readiness_report
    )
    # Replay both Controller-owned inputs now, before an AI proposal exists.
    selector = _load_evidence_selector_module()
    try:
        action_catalog = selector._validate_catalog(action_catalog)[0]
        selection_policy = selector._validate_policy(selection_policy)
    except ValueError as error:
        raise ValidationError(f"invalid active diagnosis selection inputs: {error}") from error
    _atomic_json(active_root / "epoch.json", epoch)
    _atomic_json(active_root / "evidence_catalog.json", evidence_catalog)
    _atomic_json(active_root / "execution_map.json", execution_map)
    _atomic_json(active_root / "action_catalog.json", action_catalog)
    _atomic_json(active_root / "selection_policy.json", selection_policy)
    request_history = []
    _atomic_json(active_root / "request_history.json", request_history)
    completed_action_ids = (
        ["nsys-global-timeline"]
        if contract["source"]["profiler"] == "nsys"
        and "nsys-global-timeline" in enabled_action_ids
        else []
    )
    _atomic_json(
        active_root / "completed_action_ids.json",
        completed_action_ids,
    )
    diagnosis = load_json_object(run_root / "diagnosis.json")
    knowledge_context = _load_diagnostic_knowledge_module().route_cards(
        diagnosis, execution_map, limit=3
    )
    _atomic_json(active_root / "knowledge_context.json", knowledge_context)
    project_surface_identity = _project_surface_identity(Path(control["project_root"]))
    _atomic_json(
        active_root / "project_surface_identity.json", project_surface_identity
    )
    context = {
        "schema_version": "cuda-optimizer/diagnosis-context-v1",
        "epoch_id": epoch_id,
        "epoch_sha256": epoch_sha,
        "execution_map_sha256": map_module.execution_map_digest(
            execution_map, epoch=epoch, evidence_catalog=evidence_catalog
        ),
        "evidence_catalog_sha256": _canonical_digest(evidence_catalog),
        "action_catalog_sha256": _canonical_digest(action_catalog),
        "selection_policy_sha256": _canonical_digest(selection_policy),
        "request_history_sha256": _canonical_digest(request_history),
        "completed_action_ids_sha256": _canonical_digest(completed_action_ids),
        "diagnosis_sha256": _canonical_digest(diagnosis),
        "project_surface_identity_sha256": _canonical_digest(
            project_surface_identity
        ),
        "knowledge_context": knowledge_context,
        "requires_unmodeled_hypothesis": map_result[
            "requires_unmodeled_hypothesis"
        ],
        "evidence_results": [],
    }
    _atomic_json(run_root / "diagnosis_context.json", context)
    _append_active_diagnosis_event(run_root, "context", context)
    return context


def _load_active_diagnosis_context(
    control: Mapping[str, Any], run_root: Path, state: Mapping[str, Any]
) -> tuple[dict, dict, dict, dict, dict, dict]:
    if _identity(control, "project")["digest"] != state.get(
        "baseline_identity_digest"
    ):
        raise ValidationError("project identity drifted after diagnosis context")
    workload = _normalize_frozen_workload(control)
    if workload.source_hash != state.get("workload_source_hash"):
        raise ValidationError("workload identity drifted after diagnosis context")
    if _identity(control, "isolated_environment")["digest"] != state.get(
        "baseline_environment_identity_digest"
    ):
        raise ValidationError("environment identity drifted after diagnosis context")
    _load_frozen_analysis_contract(run_root, state)
    events = _verify_active_diagnosis_ledger(run_root)
    _verify_committed_active_ledger(state, events)
    active_root = run_root / "active_diagnosis"
    epoch = load_json_object(active_root / "epoch.json")
    evidence_catalog = load_json_object(active_root / "evidence_catalog.json")
    execution_map = load_json_object(active_root / "execution_map.json")
    action_catalog = load_json_object(active_root / "action_catalog.json")
    selection_policy = load_json_object(active_root / "selection_policy.json")
    request_history = json.loads(
        (active_root / "request_history.json").read_text(encoding="utf-8")
    )
    completed_action_ids = json.loads(
        (active_root / "completed_action_ids.json").read_text(encoding="utf-8")
    )
    project_surface_identity = load_json_object(
        active_root / "project_surface_identity.json"
    )
    context = load_json_object(run_root / "diagnosis_context.json")
    if _canonical_digest(context) != state.get("diagnosis_context_sha256"):
        raise ValidationError("diagnosis context digest does not match state")
    if _canonical_digest(project_surface_identity) != context.get(
        "project_surface_identity_sha256"
    ):
        raise ValidationError("diagnosis context project surface identity drifted")
    if _project_surface_identity(Path(control["project_root"])) != project_surface_identity:
        raise ValidationError("complete project surface drifted after diagnosis context")
    if type(request_history) is not list or type(completed_action_ids) is not list:
        raise ValidationError("active diagnosis histories are invalid")
    result_summaries = context.get("evidence_results", [])
    if type(result_summaries) is not list:
        raise ValidationError("diagnosis context evidence_results is invalid")
    for index, raw_summary in enumerate(result_summaries):
        summary = _object(raw_summary, f"evidence_results[{index}]")
        summary_fields = {
            "request_signature",
            "action_id",
            "evidence_id",
            "status",
            "outcome_id",
            "result_path",
            "result_sha256",
        }
        _closed(summary, summary_fields, f"evidence_results[{index}]")
        _required(summary, summary_fields, f"evidence_results[{index}]")
        relative = _relative(summary["result_path"], f"evidence_results[{index}].result_path")
        result_path = run_root / relative
        resolved_result = result_path.resolve(strict=False)
        if (
            not _is_within(resolved_result, run_root)
            or result_path.is_symlink()
            or not result_path.is_file()
        ):
            raise ValidationError("evidence result path is not a contained regular file")
        result = load_json_object(result_path)
        if _canonical_digest(result) != summary["result_sha256"]:
            raise ValidationError("evidence result content digest does not match context")
        if result.get("request_signature") != summary["request_signature"]:
            raise ValidationError("evidence result request signature does not match context")
        artifacts = result.get("artifacts")
        if type(artifacts) is not list:
            raise ValidationError("evidence result artifacts are invalid")
        for artifact_index, raw_artifact in enumerate(artifacts):
            artifact = _object(
                raw_artifact,
                f"evidence_results[{index}].artifacts[{artifact_index}]",
            )
            _closed(
                artifact,
                {"path", "sha256"},
                f"evidence_results[{index}].artifacts[{artifact_index}]",
            )
            _required(
                artifact,
                {"path", "sha256"},
                f"evidence_results[{index}].artifacts[{artifact_index}]",
            )
            relative_artifact = _relative(
                artifact["path"],
                f"evidence_results[{index}].artifacts[{artifact_index}].path",
            )
            artifact_path = result_path.parent / relative_artifact
            if (
                not _is_within(artifact_path.resolve(strict=False), result_path.parent)
                or artifact_path.is_symlink()
                or not artifact_path.is_file()
            ):
                raise ValidationError("evidence artifact path is not a contained regular file")
            if _sha256_path(artifact_path) != artifact["sha256"]:
                raise ValidationError("evidence artifact content digest does not match result")
        evidence_id = summary["evidence_id"]
        if evidence_id is not None:
            catalog_item = evidence_catalog.get(evidence_id)
            if type(catalog_item) is not dict:
                raise ValidationError("evidence result is missing from evidence catalog")
            if catalog_item.get("artifact_sha256") != _sha256_path(result_path):
                raise ValidationError("evidence result artifact digest does not match catalog")
    epoch_module = _load_analysis_epoch_module()
    expected_identities = {
        "workload_contract_sha256": _sha256_path(Path(control["workload_manifest"])),
        "environment_sha256": state["baseline_environment_identity_digest"],
        "source_sha256": state["baseline_identity_digest"],
        "analysis_policy_sha256": _load_frozen_analysis_contract(run_root, state)[
            "analysis_policy_sha256"
        ],
    }
    try:
        epoch = epoch_module.validate_epoch(
            epoch, expected_identities=expected_identities
        )
        execution_map = _load_execution_map_module().validate_execution_map(
            execution_map, epoch=epoch, evidence_catalog=evidence_catalog
        )["execution_map"]
    except ValueError as error:
        raise ValidationError(f"active diagnosis context validation failed: {error}") from error
    expected = {
        "epoch_sha256": epoch_module.epoch_digest(epoch),
        "execution_map_sha256": _load_execution_map_module().execution_map_digest(
            execution_map, epoch=epoch, evidence_catalog=evidence_catalog
        ),
        "evidence_catalog_sha256": _canonical_digest(evidence_catalog),
        "action_catalog_sha256": _canonical_digest(action_catalog),
        "selection_policy_sha256": _canonical_digest(selection_policy),
        "request_history_sha256": _canonical_digest(request_history),
        "completed_action_ids_sha256": _canonical_digest(completed_action_ids),
    }
    for field, digest in expected.items():
        if context.get(field) != digest:
            raise ValidationError(f"diagnosis context {field} drifted")
    return (
        context,
        epoch,
        execution_map,
        evidence_catalog,
        action_catalog,
        selection_policy,
    )


def start_run(
    control: Mapping[str, Any], run_dir: os.PathLike[str] | str
) -> dict:
    normalized = validate_control_manifest(control)
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    project_root = Path(normalized["project_root"])
    if _is_within(run_root, project_root):
        raise ValidationError("run_dir must be outside project_root")
    run_root.mkdir(parents=True, exist_ok=True)
    with _run_lock(run_root):
        return _start_run_unlocked(normalized, run_root)


def _start_run_unlocked(
    control: Mapping[str, Any], run_dir: os.PathLike[str] | str
) -> dict:
    """Initialize evidence, baseline, probes, and diagnosis up to the change boundary."""
    normalized = validate_control_manifest(control)
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    project_root = Path(normalized["project_root"])
    if _is_within(run_root, project_root):
        raise ValidationError("run_dir must be outside project_root")
    control_digest = _canonical_digest(normalized)
    state_path = run_root / "state.json"
    if state_path.exists():
        state = read_run_state(run_root)
        if state["control_digest"] != control_digest:
            raise ValidationError("control manifest drifted after run initialization")
        _load_frozen_control(run_root, state)
        if state["next_action"] in {
            "readiness_action",
            "propose_hypotheses",
            "collect_evidence",
            "evidence_gap",
            "register_change",
            "edit_then_evaluate",
            "done",
            "manual_recovery",
        }:
            return state
    else:
        run_root.mkdir(parents=True, exist_ok=True)
        readiness_contract = None
        readiness_contract_digest = None
        analysis_contract = None
        analysis_contract_digest = None
        analysis_execution_bindings = None
        if normalized["schema_version"] == CONTROL_SCHEMA_V2:
            readiness_module = _load_readiness_contract_module()
            readiness_contract = readiness_module.validate_contract(
                readiness_module.load_contract(normalized["readiness_contract"]),
                project_root=project_root,
                environment_root=Path(
                    normalized["mutation"]["environment_root"]
                ),
            )
            readiness_contract_digest = readiness_module.contract_digest(
                readiness_contract
            )
        if "analysis_contract" in normalized:
            analysis_contract = _validate_active_diagnosis_contract(
                load_json_object(normalized["analysis_contract"])
            )
            matching_global_probes = [
                item
                for item in normalized["probes"]
                if item["id"] == analysis_contract["global_scan_probe_id"]
            ]
            if len(matching_global_probes) != 1:
                raise ValidationError(
                    "analysis_contract global_scan_probe_id must name a control probe"
                )
            adapter_path = _absolute(
                analysis_contract["adapter_path"],
                "analysis_contract.adapter_path",
            )
            if not _is_within(adapter_path, project_root):
                raise ValidationError(
                    "analysis_contract adapter_path must be inside project_root"
                )
            if (
                not adapter_path.is_file()
                or adapter_path.is_symlink()
                or adapter_path.stat().st_uid != os.getuid()
            ):
                raise ValidationError(
                    "analysis_contract adapter must be a user-owned regular file"
                )
            if _sha256_path(adapter_path) != analysis_contract["source"][
                "adapter_sha256"
            ]:
                raise ValidationError(
                    "analysis_contract adapter digest does not match adapter_path"
                )
            analysis_execution_bindings = {
                "schema_version": "cuda-optimizer/analysis-execution-bindings-v1",
                "global_scan": _adapter_execution_binding(
                    adapter_path,
                    matching_global_probes[0]["argv"],
                    "analysis_contract global scan",
                ),
                "actions": {},
            }
            for action in analysis_contract["actions"]:
                action_adapter = Path(action["adapter_path"])
                if not _is_within(action_adapter, project_root):
                    raise ValidationError(
                        "analysis_contract action adapter_path must be inside project_root"
                    )
                if (
                    not action_adapter.is_file()
                    or action_adapter.is_symlink()
                    or action_adapter.stat().st_uid != os.getuid()
                ):
                    raise ValidationError(
                        "analysis_contract action adapter must be a user-owned regular file"
                    )
                if _sha256_path(action_adapter) != action["adapter_sha256"]:
                    raise ValidationError(
                        "analysis_contract action adapter digest does not match adapter_path"
                    )
                analysis_execution_bindings["actions"][action["action_id"]] = (
                    _adapter_execution_binding(
                        action_adapter,
                        action["argv"],
                        f"analysis_contract action {action['action_id']}",
                    )
                )
            analysis_contract_digest = _canonical_digest(analysis_contract)
        baseline_identity = _identity(normalized, "project")
        environment_root = Path(normalized["mutation"]["environment_root"])
        environment_identity = None
        if environment_root.exists() or environment_root.is_symlink():
            if (
                environment_root.is_symlink()
                or not environment_root.is_dir()
                or environment_root.stat().st_uid != os.getuid()
            ):
                raise ValidationError(
                    "existing environment_root must be a user-owned non-symlink directory"
                )
            environment_identity = _identity(normalized, "isolated_environment")
        workload = _normalize_frozen_workload(normalized)
        if _identity(normalized, "project")["digest"] != baseline_identity["digest"]:
            raise ValidationError(
                "declared project identity changed while loading the workload adapter"
            )
        if environment_identity is not None and _identity(
            normalized, "isolated_environment"
        )["digest"] != environment_identity["digest"]:
            raise ValidationError(
                "isolated environment changed while loading the workload adapter"
            )
        now = time.time()
        runtime = _BUDGET_RUNTIME[normalized["budget"]]
        state = {
            "schema_version": "cuda-workload-optimizer/state-v1",
            "status": "active",
            "stage": (
                "readiness"
                if normalized["schema_version"] == CONTROL_SCHEMA_V2
                else "baseline"
            ),
            "round": 1,
            "completed_stages": [],
            "next_action": (
                "readiness"
                if normalized["schema_version"] == CONTROL_SCHEMA_V2
                else "baseline"
            ),
            "control_digest": control_digest,
            "workload_source_hash": workload.source_hash,
            "started_at_epoch": now,
            "updated_at_epoch": now,
            "soft_target_epoch": now + runtime["soft_target_seconds"],
            "deadline_epoch": now + runtime["hard_ceiling_seconds"],
        }
        _atomic_json(run_root / "control_manifest.json", normalized)
        if readiness_contract is not None:
            _atomic_json(
                run_root / "readiness_contract.json", readiness_contract
            )
            state["readiness_contract_digest"] = readiness_contract_digest
            state["readiness_report_digest"] = None
        if analysis_contract is not None:
            _atomic_json(
                run_root / "active_diagnosis" / "analysis_contract.json",
                analysis_contract,
            )
            state["analysis_contract_digest"] = analysis_contract_digest
            _atomic_json(
                run_root / "active_diagnosis" / "execution_bindings.json",
                analysis_execution_bindings,
            )
            state["analysis_execution_bindings_digest"] = _canonical_digest(
                analysis_execution_bindings
            )
        _atomic_json(run_root / "baseline_identity.json", baseline_identity)
        state["baseline_identity_digest"] = baseline_identity["digest"]
        state["baseline_environment_identity_digest"] = (
            None if environment_identity is None else environment_identity["digest"]
        )
        if environment_identity is not None:
            _atomic_json(
                run_root / "baseline_environment_identity.json",
                environment_identity,
            )
        (run_root / "host_recommendations.md").write_text(
            "# Host recommendations\n\nNo host mutation was executed. Add evidence-backed suggestions here for manual review.\n",
            encoding="utf-8",
        )
        state = _write_state(run_root, state)

    _check_deadline(state)
    _load_frozen_control(run_root, state)
    workload = _normalize_frozen_workload(normalized)
    if workload.source_hash != state["workload_source_hash"]:
        raise ValidationError("workload identity drifted after run initialization")
    runtime = _BUDGET_RUNTIME[normalized["budget"]]

    if (
        normalized["schema_version"] == CONTROL_SCHEMA_V2
        and "readiness" not in state["completed_stages"]
    ):
        report = _run_readiness_gate_checked(normalized, run_root, state)
        state = copy.deepcopy(state)
        state["readiness_report_digest"] = _readiness_report_digest(
            run_root, report
        )
        state["readiness_environment_identity_digest"] = report.get(
            "environment_identity_digest"
        )
        if not report.get("can_start_diagnosis"):
            state["stage"] = "readiness"
            state["next_action"] = "readiness_action"
            state["updated_at_epoch"] = time.time()
            return _write_state(run_root, state)
        if _identity(normalized, "project")["digest"] != state.get(
            "baseline_identity_digest"
        ):
            raise ValidationError(
                "declared project identity drifted during readiness"
            )
        if _normalize_frozen_workload(normalized).source_hash != state.get(
            "workload_source_hash"
        ):
            raise ValidationError("workload identity drifted during readiness")
        refreshed_environment = _identity(
            normalized, "isolated_environment"
        )
        _atomic_json(
            run_root / "baseline_environment_identity.json",
            refreshed_environment,
        )
        state["baseline_environment_identity_digest"] = refreshed_environment[
            "digest"
        ]
        state = _advance(
            run_root,
            state,
            "readiness",
            stage="baseline",
            next_action="baseline",
        )

    if normalized["schema_version"] == CONTROL_SCHEMA_V2:
        if not _verify_readiness_report(normalized, run_root, state):
            report = _run_readiness_gate_checked(normalized, run_root, state)
            state = copy.deepcopy(state)
            state["readiness_report_digest"] = _readiness_report_digest(
                run_root, report
            )
            state["readiness_environment_identity_digest"] = report.get(
                "environment_identity_digest"
            )
            state["updated_at_epoch"] = time.time()
            state = _write_state(run_root, state)
            if not report.get("can_start_diagnosis"):
                state["stage"] = "readiness"
                state["next_action"] = "readiness_action"
                return _write_state(run_root, state)

    if "baseline" not in state["completed_stages"]:
        baseline_surface_before = (
            _project_surface_identity(Path(normalized["project_root"]))
            if "analysis_contract" in normalized
            else None
        )
        timeout = (
            None
            if workload.kind == "python"
            else min(120, max(0.001, state["deadline_epoch"] - time.time()))
        )
        baseline = _load_evaluate_module().measure_candidate(
            workload,
            normalized["baseline_candidate"],
            role="baseline",
            retries=runtime["retries"],
            timeout=timeout,
            deadline_epoch=state["deadline_epoch"],
        )
        if (
            baseline_surface_before is not None
            and _project_surface_identity(Path(normalized["project_root"]))
            != baseline_surface_before
        ):
            raise ValidationError("baseline modified the complete project surface")
        _atomic_json(run_root / "baseline" / "observation.json", baseline)
        if baseline["status"] != "measured":
            raise ValidationError("baseline workload failed; see baseline/observation.json")
        state = _advance(
            run_root, state, "baseline", stage="probes", next_action="probes"
        )
    _check_deadline(state)
    if "probes" not in state["completed_stages"]:
        if (
            normalized["schema_version"] == CONTROL_SCHEMA_V2
            and not _verify_readiness_report(normalized, run_root, state)
        ):
            report = _run_readiness_gate_checked(normalized, run_root, state)
            state = copy.deepcopy(state)
            state["readiness_report_digest"] = _readiness_report_digest(
                run_root, report
            )
            state["readiness_environment_identity_digest"] = report.get(
                "environment_identity_digest"
            )
            state["updated_at_epoch"] = time.time()
            if not report.get("can_start_diagnosis"):
                state["stage"] = "readiness"
                state["next_action"] = "readiness_action"
                return _write_state(run_root, state)
            state = _write_state(run_root, state)
        probe_surface_before = (
            _project_surface_identity(Path(normalized["project_root"]))
            if "analysis_contract" in normalized
            else None
        )
        probe_identity_before = _identity(normalized, "project")
        run_probes(
            normalized,
            run_root,
            deadline_epoch=state["deadline_epoch"],
        )
        if _identity(normalized, "project") != probe_identity_before:
            raise ValidationError("diagnosis probes modified declared project inputs")
        if (
            probe_surface_before is not None
            and _project_surface_identity(Path(normalized["project_root"]))
            != probe_surface_before
        ):
            raise ValidationError("diagnosis probes modified the complete project surface")
        state = _advance(
            run_root, state, "probes", stage="diagnosis", next_action="diagnosis"
        )
    _check_deadline(state)
    if "diagnosis" not in state["completed_stages"]:
        diagnose_run(run_root)
        if "analysis_contract" in normalized:
            state = _advance(
                run_root,
                state,
                "diagnosis",
                stage="active_diagnosis",
                next_action="diagnosis_context",
            )
        else:
            state = _advance(
                run_root,
                state,
                "diagnosis",
                stage="change",
                next_action="register_change",
            )
    if (
        "analysis_contract" in normalized
        and "diagnosis_context" not in state["completed_stages"]
    ):
        context = _build_active_diagnosis_context(normalized, run_root, state)
        state = _advance(
            run_root,
            state,
            "diagnosis_context",
            stage="active_diagnosis",
            next_action="propose_hypotheses",
        )
        updated = copy.deepcopy(state)
        updated["diagnosis_context_sha256"] = _canonical_digest(context)
        updated.update(
            _active_ledger_binding(_verify_active_diagnosis_ledger(run_root))
        )
        state = _write_state(run_root, updated)
    return state


def register_active_diagnosis_proposal(
    control: Mapping[str, Any],
    run_dir: os.PathLike[str] | str,
    hypothesis_set: Mapping[str, Any],
    request_set: Mapping[str, Any],
) -> dict:
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    with _run_lock(run_root):
        return _register_active_diagnosis_proposal_unlocked(
            control, run_root, hypothesis_set, request_set
        )


def _hypothesis_identity_registry(hypothesis_result: Mapping[str, Any]) -> dict:
    hypothesis_set = hypothesis_result["hypothesis_set"]
    return {
        "hypotheses": {
            item["hypothesis_id"]: {
                "kind": item["kind"],
                "scope_node_ids": item["scope_node_ids"],
                "statement": item["statement"],
                "mechanism": item["mechanism"],
            }
            for item in hypothesis_set["hypotheses"]
        },
        "relationships": copy.deepcopy(hypothesis_set["relationships"]),
    }


def _register_active_diagnosis_proposal_unlocked(
    control: Mapping[str, Any],
    run_dir: os.PathLike[str] | str,
    hypothesis_set: Mapping[str, Any],
    request_set: Mapping[str, Any],
) -> dict:
    """Replay an AI proposal against Controller-owned context and policy."""
    normalized = validate_control_manifest(control)
    if "analysis_contract" not in normalized:
        raise ValidationError("control does not enable active diagnosis")
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    state = read_run_state(run_root)
    if state["control_digest"] != _canonical_digest(normalized):
        raise ValidationError("control manifest drifted before diagnosis proposal")
    _load_frozen_control(run_root, state)
    if state["next_action"] != "propose_hypotheses":
        raise ValidationError("run is not ready for an active diagnosis proposal")
    _check_deadline(state)
    (
        context,
        epoch,
        execution_map,
        evidence_catalog,
        action_catalog,
        selection_policy,
    ) = _load_active_diagnosis_context(normalized, run_root, state)
    frozen_registry = None
    frozen_hypothesis_sha = state.get("hypothesis_set_sha256")
    if frozen_hypothesis_sha is not None:
        prior_result = load_json_object(
            run_root
            / "active_diagnosis"
            / "hypothesis_generations"
            / f"{frozen_hypothesis_sha}.json"
        )
        prior_set = prior_result.get("hypothesis_set")
        if (
            type(prior_set) is not dict
            or prior_result.get("hypothesis_set_sha256") != frozen_hypothesis_sha
            or _canonical_digest(prior_set) != frozen_hypothesis_sha
        ):
            raise ValidationError("frozen hypothesis result drifted from run state")
        frozen_registry = _hypothesis_identity_registry(prior_result)
    try:
        hypothesis_result = _load_hypothesis_space_module().validate_hypothesis_set(
            hypothesis_set,
            epoch=epoch,
            execution_map=execution_map,
            evidence_catalog=evidence_catalog,
        )
        proposed_registry = _hypothesis_identity_registry(hypothesis_result)
        if frozen_registry is not None and proposed_registry != frozen_registry:
            raise ValidationError(
                "hypothesis identity registry cannot change inside an analysis epoch"
            )
        selection = _load_evidence_selector_module().select_evidence_request(
            request_set,
            epoch=epoch,
            hypothesis_result=hypothesis_result,
            evidence_catalog=evidence_catalog,
            action_catalog=action_catalog,
            policy=selection_policy,
            request_history=json.loads(
                (run_root / "active_diagnosis" / "request_history.json").read_text(
                    encoding="utf-8"
                )
            ),
            completed_action_ids=json.loads(
                (run_root / "active_diagnosis" / "completed_action_ids.json").read_text(
                    encoding="utf-8"
                )
            ),
        )
    except ValueError as error:
        raise ValidationError(f"active diagnosis proposal rejected: {error}") from error
    active_root = run_root / "active_diagnosis"
    _atomic_json(
        active_root
        / "hypothesis_generations"
        / f"{hypothesis_result['hypothesis_set_sha256']}.json",
        hypothesis_result,
    )
    _atomic_json(active_root / "hypothesis_result.json", hypothesis_result)
    _atomic_json(
        active_root / "request_set.json",
        _json_copy(request_set, "request_set", reject_sensitive=True),
    )
    _atomic_json(active_root / "evidence_selection.json", selection)
    proposal_binding = {
        "context_sha256": _canonical_digest(context),
        "hypothesis_set_sha256": hypothesis_result["hypothesis_set_sha256"],
        "request_set_sha256": _canonical_digest(request_set),
        "selection_sha256": _canonical_digest(selection),
    }
    _append_active_diagnosis_event(run_root, "proposal", proposal_binding)
    next_action = {
        "selected": "collect_evidence",
        "sufficient": "register_change",
        "evidence_gap": "evidence_gap",
    }[selection["status"]]
    updated = copy.deepcopy(state)
    updated.update(
        {
            "stage": "active_diagnosis",
            "next_action": next_action,
            "updated_at_epoch": time.time(),
            "hypothesis_set_sha256": hypothesis_result["hypothesis_set_sha256"],
            "evidence_selection_sha256": _canonical_digest(selection),
            "diagnosis_context_sha256": _canonical_digest(context),
        }
    )
    updated.update(_active_ledger_binding(_verify_active_diagnosis_ledger(run_root)))
    if selection["selected_request"] is not None:
        updated["selected_request_signature"] = selection["selected_request"][
            "request_signature"
        ]
    if "diagnosis_proposal" not in updated["completed_stages"]:
        updated["completed_stages"].append("diagnosis_proposal")
    return _write_state(run_root, updated)


def _validate_evidence_result(
    value: Mapping[str, Any], selected: Mapping[str, Any], attempt_root: Path
) -> dict:
    result = _object(value, "evidence_result")
    fields = {
        "schema_version",
        "request_signature",
        "status",
        "outcome_id",
        "observations",
        "artifacts",
    }
    _closed(result, fields, "evidence_result")
    _required(result, fields, "evidence_result")
    if result["schema_version"] != "cuda-optimizer/evidence-result-v1":
        raise ValidationError("evidence_result schema_version is unsupported")
    if result["request_signature"] != selected["request_signature"]:
        raise ValidationError("evidence result request signature does not match selection")
    if result["status"] not in {"observed", "inconclusive", "unavailable", "failed"}:
        raise ValidationError("evidence_result.status is unsupported")
    outcome_ids = {item["outcome_id"] for item in selected["outcomes"]}
    outcome_id = result["outcome_id"]
    if result["status"] == "observed":
        if outcome_id not in outcome_ids:
            raise ValidationError("observed evidence must name a selected outcome")
    elif outcome_id is not None:
        raise ValidationError("non-observed evidence must use a null outcome_id")
    observations = _object(result["observations"], "evidence_result.observations")
    artifacts = result["artifacts"]
    if type(artifacts) is not list:
        raise ValidationError("evidence_result.artifacts must be an array")
    sealed_artifacts = []
    resolved_attempt = attempt_root.resolve(strict=True)
    reserved_artifact_paths = {
        ".output.json",
        "request.json",
        "intent.json",
        "execution.json",
        "result.json",
        "complete.json",
    }
    for index, item in enumerate(artifacts):
        artifact = _object(item, f"evidence_result.artifacts[{index}]")
        _closed(artifact, {"path", "sha256"}, f"evidence_result.artifacts[{index}]")
        _required(artifact, {"path"}, f"evidence_result.artifacts[{index}]")
        if "sha256" in artifact:
            _sha256(
                artifact["sha256"], f"evidence_result.artifacts[{index}].sha256"
            )
        raw_path = Path(
            _string(artifact["path"], f"evidence_result.artifacts[{index}].path")
        )
        artifact_path = raw_path if raw_path.is_absolute() else attempt_root / raw_path
        resolved_artifact = artifact_path.resolve(strict=False)
        if (
            not _is_within(resolved_artifact, resolved_attempt)
            or artifact_path.is_symlink()
            or not artifact_path.is_file()
        ):
            raise ValidationError("evidence artifact must be a contained regular file")
        relative = resolved_artifact.relative_to(resolved_attempt)
        if relative.as_posix() in reserved_artifact_paths:
            raise ValidationError(
                "evidence artifact cannot use a Controller-reserved path"
            )
        sealed_artifacts.append(
            {"path": str(relative), "sha256": _sha256_path(artifact_path)}
        )
    return {
        **copy.deepcopy(dict(result)),
        "observations": _json_copy(
            observations, "evidence_result.observations", reject_sensitive=True
        ),
        "artifacts": sealed_artifacts,
    }


def _run_active_evidence_adapter(
    control: Mapping[str, Any],
    run_root: Path,
    state: Mapping[str, Any],
    action: Mapping[str, Any],
    selected: Mapping[str, Any],
    attempt_root: Path,
) -> tuple[dict, dict]:
    output_path = attempt_root / ".output.json"
    request_path = attempt_root / "request.json"
    _atomic_json(request_path, selected)
    try:
        output_path.unlink()
    except FileNotFoundError:
        pass
    project_root = Path(control["project_root"])
    execution_root = project_root
    execution_argv = list(action["argv"])
    project_identity_before = _identity(control, "project")
    project_surface_before = _project_surface_identity(project_root)
    if selected["controller_action"]["control_scope"] == "project_copy":
        execution_root = attempt_root / "project_copy"
        if execution_root.exists() or execution_root.is_symlink():
            raise ValidationError("direction experiment project copy already exists")
        shutil.copytree(
            project_root,
            execution_root,
            symlinks=False,
            ignore=shutil.ignore_patterns(
                ".git", ".worktrees", "__pycache__", "*.pyc"
            ),
        )
        mapped_argv = []
        for token in execution_argv:
            token_path = Path(token)
            if token_path.is_absolute() and _is_within(
                token_path.resolve(strict=False), project_root
            ):
                mapped_argv.append(
                    str(execution_root / token_path.resolve(strict=False).relative_to(project_root))
                )
            else:
                mapped_argv.append(token)
        execution_argv = mapped_argv
    environment, secret_values = _probe_environment(
        {
            "CUDA_OPTIMIZER_EVIDENCE_OUTPUT": str(output_path),
            "CUDA_OPTIMIZER_EVIDENCE_REQUEST": str(request_path),
            "CUDA_OPTIMIZER_EVIDENCE_DIR": str(attempt_root),
            "CUDA_OPTIMIZER_RUN_DIR": str(run_root),
            "CUDA_OPTIMIZER_PROJECT_ROOT": str(execution_root),
        }
    )
    stdout = _BoundedLog(_DEFAULT_LOG_LIMIT)
    stderr = _BoundedLog(_DEFAULT_LOG_LIMIT)
    timeout = min(
        float(action["timeout_seconds"]),
        max(0.001, float(state["deadline_epoch"]) - time.time()),
    )
    started = time.monotonic()
    exit_code = None
    timed_out = False
    process = subprocess.Popen(
        execution_argv,
        cwd=execution_root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    readers = [
        threading.Thread(target=_drain, args=(process.stdout, stdout), daemon=True),
        threading.Thread(target=_drain, args=(process.stderr, stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()
    try:
        exit_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _stop_group(process)
        exit_code = process.returncode
    else:
        if _process_group_exists(process.pid):
            _stop_group(process)
    for reader in readers:
        reader.join(timeout=1)
    execution = {
        "schema_version": "cuda-optimizer/evidence-execution-v1",
        "action_id": action["action_id"],
        "argv_sha256": _canonical_digest(action["argv"]),
        "execution_argv_sha256": _canonical_digest(execution_argv),
        "adapter_sha256": action["adapter_sha256"],
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_seconds": time.monotonic() - started,
        "stdout": _redact_log(stdout.text(), secret_values),
        "stderr": _redact_log(stderr.text(), secret_values),
    }
    _atomic_json(attempt_root / "execution.json", execution)
    if _identity(control, "project")["digest"] != project_identity_before["digest"]:
        raise ValidationError("evidence action modified the frozen project")
    if _project_surface_identity(project_root) != project_surface_before:
        raise ValidationError("evidence action modified the complete project surface")
    if timed_out:
        result = {
            "schema_version": "cuda-optimizer/evidence-result-v1",
            "request_signature": selected["request_signature"],
            "status": "unavailable",
            "outcome_id": None,
            "observations": {"reason": "timeout"},
            "artifacts": [],
        }
    elif exit_code != 0:
        result = {
            "schema_version": "cuda-optimizer/evidence-result-v1",
            "request_signature": selected["request_signature"],
            "status": "failed",
            "outcome_id": None,
            "observations": {"reason": "nonzero_exit"},
            "artifacts": [],
        }
    else:
        result = _validate_evidence_result(
            _read_probe_output(output_path), selected, attempt_root
        )
    try:
        output_path.unlink()
    except FileNotFoundError:
        pass
    return result, execution


def _refresh_active_diagnosis_context(
    run_root: Path,
    context: Mapping[str, Any],
    epoch: Mapping[str, Any],
    execution_map: Mapping[str, Any],
    evidence_catalog: Mapping[str, Any],
    selection_policy: Mapping[str, Any],
    result_summary: Mapping[str, Any],
) -> dict:
    refreshed = copy.deepcopy(dict(context))
    refreshed["evidence_catalog_sha256"] = _canonical_digest(evidence_catalog)
    refreshed["selection_policy_sha256"] = _canonical_digest(selection_policy)
    refreshed["request_history_sha256"] = _canonical_digest(
        json.loads(
            (run_root / "active_diagnosis" / "request_history.json").read_text(
                encoding="utf-8"
            )
        )
    )
    refreshed["completed_action_ids_sha256"] = _canonical_digest(
        json.loads(
            (run_root / "active_diagnosis" / "completed_action_ids.json").read_text(
                encoding="utf-8"
            )
        )
    )
    refreshed["execution_map_sha256"] = (
        _load_execution_map_module().execution_map_digest(
            execution_map, epoch=epoch, evidence_catalog=evidence_catalog
        )
    )
    evidence_results = refreshed.get("evidence_results", [])
    if type(evidence_results) is not list:
        raise ValidationError("diagnosis context evidence_results is invalid")
    evidence_results.append(copy.deepcopy(dict(result_summary)))
    refreshed["evidence_results"] = evidence_results
    _atomic_json(run_root / "diagnosis_context.json", refreshed)
    return refreshed


def _recover_or_block_active_evidence_attempt(
    control: Mapping[str, Any],
    run_root: Path,
    state: Mapping[str, Any],
    selected: Mapping[str, Any],
    attempt_root: Path,
) -> dict | None:
    intent_path = attempt_root / "intent.json"
    complete_path = attempt_root / "complete.json"
    if not intent_path.exists() and not complete_path.exists():
        return None
    if intent_path.is_symlink() or not intent_path.is_file():
        raise ValidationError("evidence intent is not a regular file")
    if not complete_path.exists():
        updated = copy.deepcopy(dict(state))
        updated.update(
            {
                "status": "blocked",
                "stage": "active_diagnosis",
                "next_action": "manual_recovery",
                "updated_at_epoch": time.time(),
                "manual_recovery_reason": "evidence_action_interrupted_not_reexecuted",
            }
        )
        return _write_state(run_root, updated)
    if complete_path.is_symlink() or not complete_path.is_file():
        raise ValidationError("evidence completion is not a regular file")
    completion = load_json_object(complete_path)
    fields = {
        "schema_version",
        "request_signature",
        "result_sha256",
        "execution_sha256",
        "context_sha256",
        "completed_at_epoch",
    }
    _closed(completion, fields, "evidence_completion")
    _required(completion, fields, "evidence_completion")
    if completion["schema_version"] != "cuda-optimizer/evidence-completion-v1":
        raise ValidationError("evidence completion schema_version is unsupported")
    signature = selected["request_signature"]
    if completion["request_signature"] != signature:
        raise ValidationError("evidence completion request signature drifted")
    result = load_json_object(attempt_root / "result.json")
    execution = load_json_object(attempt_root / "execution.json")
    context = load_json_object(run_root / "diagnosis_context.json")
    expected_digests = {
        "result_sha256": _canonical_digest(result),
        "execution_sha256": _canonical_digest(execution),
        "context_sha256": _canonical_digest(context),
    }
    for field, digest in expected_digests.items():
        if completion[field] != digest:
            raise ValidationError(f"evidence completion {field} drifted")
    event_payload = {
        "request_signature": signature,
        **expected_digests,
    }
    payload_sha = _canonical_digest(event_payload)
    events = _verify_active_diagnosis_ledger(run_root)
    if not any(
        event["event_type"] == "evidence"
        and event["payload_sha256"] == payload_sha
        for event in events
    ):
        raise ValidationError("evidence completion has no matching ledger event")
    recovered = copy.deepcopy(dict(state))
    recovered.update(
        {
            "stage": "active_diagnosis",
            "next_action": "propose_hypotheses",
            "updated_at_epoch": time.time(),
            "diagnosis_context_sha256": completion["context_sha256"],
            "last_request_signature": signature,
            "active_diagnosis_round": int(state.get("active_diagnosis_round", 1)) + 1,
        }
    )
    recovered.update(_active_ledger_binding(events))
    _load_active_diagnosis_context(control, run_root, recovered)
    return _write_state(run_root, recovered)


def collect_active_diagnosis_evidence(
    control: Mapping[str, Any], run_dir: os.PathLike[str] | str
) -> dict:
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    with _run_lock(run_root):
        return _collect_active_diagnosis_evidence_unlocked(control, run_root)


def _collect_active_diagnosis_evidence_unlocked(
    control: Mapping[str, Any], run_dir: os.PathLike[str] | str
) -> dict:
    """Execute one frozen evidence action and checkpoint the next diagnosis round."""
    normalized = validate_control_manifest(control)
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    state = read_run_state(run_root)
    if state["control_digest"] != _canonical_digest(normalized):
        raise ValidationError("control manifest drifted before evidence collection")
    if (
        state["next_action"] == "propose_hypotheses"
        and state.get("last_request_signature")
    ):
        return state
    if state["next_action"] != "collect_evidence":
        raise ValidationError("run is not ready to collect active diagnosis evidence")
    selection = load_json_object(
        run_root / "active_diagnosis" / "evidence_selection.json"
    )
    if _canonical_digest(selection) != state.get("evidence_selection_sha256"):
        raise ValidationError("evidence selection digest drifted before collection")
    selected = selection.get("selected_request")
    if type(selected) is not dict or selection.get("status") != "selected":
        raise ValidationError("evidence selection contains no executable request")
    signature = selected["request_signature"]
    attempt_root = run_root / "active_diagnosis" / "evidence" / signature
    recovered = _recover_or_block_active_evidence_attempt(
        normalized, run_root, state, selected, attempt_root
    )
    if recovered is not None:
        return recovered
    _check_deadline(state)
    if not _verify_readiness_report(normalized, run_root, state):
        report = _run_readiness_gate_checked(normalized, run_root, state)
        refreshed_state = copy.deepcopy(state)
        refreshed_state["readiness_report_digest"] = _readiness_report_digest(
            run_root, report
        )
        refreshed_state["readiness_environment_identity_digest"] = report.get(
            "environment_identity_digest"
        )
        refreshed_state["updated_at_epoch"] = time.time()
        if not report.get("can_start_diagnosis"):
            refreshed_state["stage"] = "readiness"
            refreshed_state["next_action"] = "readiness_action"
            return _write_state(run_root, refreshed_state)
        if report.get("environment_identity_digest") != state.get(
            "baseline_environment_identity_digest"
        ):
            raise ValidationError(
                "environment identity changed after baseline; create a child run"
            )
        state = _write_state(run_root, refreshed_state)
    (
        context,
        epoch,
        execution_map,
        evidence_catalog,
        _action_catalog,
        selection_policy,
    ) = _load_active_diagnosis_context(normalized, run_root, state)
    contract = _load_frozen_analysis_contract(run_root, state)
    action_by_id = {item["action_id"]: item for item in contract["actions"]}
    action = action_by_id.get(selected["action_id"])
    if action is None:
        raise ValidationError("selected evidence action has no frozen adapter")
    required = set(selected["controller_action"]["required_capability_ids"])
    readiness_report = _load_readiness_gate_module()._load_prior_report(
        run_root / "readiness"
    )
    if readiness_report is None:
        raise ValidationError("evidence collection requires a readiness report")
    available = set(_ready_capability_ids(readiness_report))
    if not required.issubset(available):
        updated = copy.deepcopy(state)
        updated.update(
            {
                "stage": "active_diagnosis",
                "next_action": "evidence_gap",
                "updated_at_epoch": time.time(),
                "missing_capability_ids": sorted(required - available),
            }
        )
        return _write_state(run_root, updated)
    adapter_path = Path(action["adapter_path"])
    if _sha256_path(adapter_path) != action["adapter_sha256"]:
        raise ValidationError("evidence action adapter drifted before execution")
    bindings = _load_frozen_execution_bindings(run_root, state)
    expected_binding = bindings.get("actions", {}).get(action["action_id"])
    if type(expected_binding) is not dict:
        raise ValidationError("evidence action has no frozen execution binding")
    _verify_adapter_execution_binding(
        expected_binding,
        adapter_path,
        action["argv"],
        f"analysis_contract action {action['action_id']}",
    )

    intent_path = attempt_root / "intent.json"
    complete_path = attempt_root / "complete.json"
    intent = {
        "schema_version": "cuda-optimizer/evidence-intent-v1",
        "request_signature": signature,
        "selection_sha256": _canonical_digest(selection),
        "action_sha256": _canonical_digest(action),
        "created_at_epoch": time.time(),
    }
    _atomic_json(intent_path, intent)
    result, execution = _run_active_evidence_adapter(
        normalized, run_root, state, action, selected, attempt_root
    )
    _atomic_json(attempt_root / "result.json", result)
    evidence_id = None
    if result["status"] == "observed":
        evidence_id = f"ev-{signature[:16]}"
        outcome = next(
            item for item in selected["outcomes"] if item["outcome_id"] == result["outcome_id"]
        )
        evidence_catalog[evidence_id] = {
            "epoch_id": epoch["epoch_id"],
            "kind": selected["controller_action"]["evidence_kind"],
            "artifact_sha256": _sha256_path(attempt_root / "result.json"),
            "supports_hypothesis_ids": sorted(outcome["supports"]),
            "opposes_hypothesis_ids": sorted(outcome["opposes"]),
        }
    _atomic_json(run_root / "active_diagnosis" / "evidence_catalog.json", evidence_catalog)
    history_path = run_root / "active_diagnosis" / "request_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if type(history) is not list:
        raise ValidationError("active diagnosis request history is invalid")
    if signature not in history:
        history.append(signature)
    history.sort()
    _atomic_json(history_path, history)
    completed_path = run_root / "active_diagnosis" / "completed_action_ids.json"
    completed_actions = json.loads(completed_path.read_text(encoding="utf-8"))
    if type(completed_actions) is not list:
        raise ValidationError("active diagnosis completed action history is invalid")
    if selected["action_id"] not in completed_actions:
        completed_actions.append(selected["action_id"])
    completed_actions.sort()
    _atomic_json(completed_path, completed_actions)
    selection_policy = copy.deepcopy(selection_policy)
    selection_policy["remaining_profile_actions"] = max(
        0, int(selection_policy["remaining_profile_actions"]) - 1
    )
    _atomic_json(
        run_root / "active_diagnosis" / "selection_policy.json", selection_policy
    )
    refreshed_context = _refresh_active_diagnosis_context(
        run_root,
        context,
        epoch,
        execution_map,
        evidence_catalog,
        selection_policy,
        {
            "request_signature": signature,
            "action_id": selected["action_id"],
            "evidence_id": evidence_id,
            "status": result["status"],
            "outcome_id": result["outcome_id"],
            "result_path": str(
                (attempt_root / "result.json").relative_to(run_root)
            ),
            "result_sha256": _canonical_digest(result),
        },
    )
    event_payload = {
        "request_signature": signature,
        "result_sha256": _canonical_digest(result),
        "execution_sha256": _canonical_digest(execution),
        "context_sha256": _canonical_digest(refreshed_context),
    }
    _append_active_diagnosis_event(run_root, "evidence", event_payload)
    _atomic_json(
        complete_path,
        {
            "schema_version": "cuda-optimizer/evidence-completion-v1",
            **event_payload,
            "completed_at_epoch": time.time(),
        },
    )
    updated = copy.deepcopy(state)
    updated.update(
        {
            "stage": "active_diagnosis",
            "next_action": "propose_hypotheses",
            "updated_at_epoch": time.time(),
            "diagnosis_context_sha256": _canonical_digest(refreshed_context),
            "last_request_signature": signature,
            "active_diagnosis_round": int(state.get("active_diagnosis_round", 1)) + 1,
        }
    )
    updated.update(_active_ledger_binding(_verify_active_diagnosis_ledger(run_root)))
    return _write_state(run_root, updated)


def register_change(
    control: Mapping[str, Any],
    run_dir: os.PathLike[str] | str,
    change_set: Mapping[str, Any],
) -> dict:
    normalized = validate_control_manifest(control)
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    state = read_run_state(run_root)
    if state["control_digest"] != _canonical_digest(normalized):
        raise ValidationError("control manifest drifted before ChangeSet registration")
    _load_frozen_control(run_root, state)
    change = validate_change_set(change_set, normalized)
    change_digest = _canonical_digest(change)
    pending_path = run_root / "registration_pending.json"
    if state["next_action"] == "edit_then_evaluate":
        if state.get("change_set_digest") != change_digest:
            raise ValidationError("a different ChangeSet is already registered")
        try:
            pending_path.unlink()
        except FileNotFoundError:
            pass
        return state
    if state["next_action"] != "register_change":
        raise ValidationError("run is not ready to register a ChangeSet")
    _check_deadline(state)
    if change["scope"] == "project":
        current_identity = _identity(normalized, "project")
        if current_identity["digest"] != state.get("baseline_identity_digest"):
            raise ValidationError(
                "declared project identity drifted after baseline capture"
            )
    else:
        expected_environment = state.get("baseline_environment_identity_digest")
        if expected_environment is None:
            raise ValidationError(
                "environment_root must exist before baseline capture for isolated changes"
            )
        if _identity(normalized, "isolated_environment")["digest"] != expected_environment:
            raise ValidationError(
                "isolated environment drifted after baseline capture"
            )
    pending = {
        "schema_version": "cuda-workload-optimizer/registration-pending-v1",
        "change_set_digest": change_digest,
        "scope": change["scope"],
    }
    if pending_path.exists():
        if load_json_object(pending_path) != pending:
            raise ValidationError("pending ChangeSet registration does not match retry")
        before_path = run_root / "rounds" / "round-1" / "before_identity.json"
        if before_path.exists():
            before_value = load_json_object(before_path)
            before = _validated_identity_artifact(
                before_value, before_value.get("digest", "")
            )
        else:
            snapshot_name = "project" if change["scope"] == "project" else "environment"
            incomplete_snapshot = run_root / "snapshot" / snapshot_name
            if incomplete_snapshot.exists() or incomplete_snapshot.is_symlink():
                _remove_path(incomplete_snapshot)
            before = _snapshot_scope(normalized, run_root, change["scope"])
    else:
        _atomic_json(pending_path, pending)
        before = _snapshot_scope(normalized, run_root, change["scope"])
    _atomic_json(run_root / "change_set.json", change)
    _atomic_json(run_root / "rounds" / "round-1" / "change_set.json", change)
    updated = copy.deepcopy(state)
    if "change" not in updated["completed_stages"]:
        updated["completed_stages"].append("change")
    updated.update(
        {
            "stage": "review",
            "next_action": "edit_then_evaluate",
            "updated_at_epoch": time.time(),
            "before_identity_digest": before["digest"],
            "change_set_digest": change_digest,
            "change_scope": change["scope"],
        }
    )
    committed = _write_state(run_root, updated)
    try:
        pending_path.unlink()
    except FileNotFoundError:
        pass
    return committed


def _run_correctness_commands(
    control: Mapping[str, Any],
    change: Mapping[str, Any],
    run_root: Path,
    *,
    timeout_seconds: float = 300,
) -> dict:
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise ValidationError("correctness timeout must be a positive finite number")
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 300:
        raise ValidationError("correctness timeout must be positive and at most 300 seconds")
    records = []
    environment, secrets = _probe_environment({})
    for index, argv in enumerate(change["commands"]):
        started = time.monotonic()
        stdout = _BoundedLog(_DEFAULT_LOG_LIMIT)
        stderr = _BoundedLog(_DEFAULT_LOG_LIMIT)
        process = None
        returncode = None
        failure = None
        try:
            process = subprocess.Popen(
                argv,
                cwd=control["project_root"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            readers = [
                threading.Thread(target=_drain, args=(process.stdout, stdout), daemon=True),
                threading.Thread(target=_drain, args=(process.stderr, stderr), daemon=True),
            ]
            for reader in readers:
                reader.start()
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                failure = "TimeoutExpired"
                _stop_group(process)
                returncode = process.returncode
            else:
                if _process_group_exists(process.pid):
                    _stop_group(process)
            for reader in readers:
                reader.join(timeout=1)
        except OSError as error:
            failure = type(error).__name__
        records.append(
            {
                "index": index,
                "argv_sha256": _canonical_digest(argv),
                "returncode": returncode,
                "duration_seconds": time.monotonic() - started,
                "stdout": _redact_log(stdout.text(), secrets),
                "stderr": _redact_log(stderr.text(), secrets),
                "failure": failure,
            }
        )
        if failure is not None or returncode != 0:
            break
    result = {
        "schema_version": "cuda-workload-optimizer/correctness-v1",
        "status": (
            "passed"
            if len(records) == len(change["commands"])
            and all(record["returncode"] == 0 for record in records)
            else "failed"
        ),
        "commands": records,
    }
    _atomic_json(run_root / "correctness.json", result)
    return result


def _finish_rejected(
    run_root: Path,
    state: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    scope: str,
    reason: str,
    primary_status: str | None,
    time_gate: Mapping[str, Any] | None = None,
) -> dict:
    try:
        _restore_snapshot(
            control,
            run_root,
            scope,
            state["before_identity_digest"],
        )
    except (OSError, ValidationError) as error:
        decision = {
            "schema_version": "cuda-workload-optimizer/decision-v1",
            "status": "manual_recovery_required",
            "reason": "rollback_failed",
            "rejected_reason": reason,
            "primary_status": primary_status,
            "rolled_back": False,
            "error": f"{type(error).__name__}: {error}",
            "snapshot": str(run_root / "snapshot" / ("project" if scope == "project" else "environment")),
        }
        _atomic_json(run_root / "decision.json", decision)
        updated = copy.deepcopy(state)
        updated.update(
            {
                "status": "manual_recovery_required",
                "stage": "decision",
                "next_action": "manual_recovery",
                "updated_at_epoch": time.time(),
                "decision_digest": _canonical_digest(decision),
            }
        )
        _write_state(run_root, updated)
        return decision
    decision = {
        "schema_version": "cuda-workload-optimizer/decision-v1",
        "status": "rejected",
        "reason": reason,
        "primary_status": primary_status,
        "rolled_back": True,
    }
    if time_gate is not None:
        decision.update(
            {
                "elapsed_seconds": time_gate["elapsed_seconds"],
                "stop_reason": time_gate["stop_reason"],
                "skipped_expensive_stages": time_gate[
                    "skipped_expensive_stages"
                ],
            }
        )
    _atomic_json(run_root / "decision.json", decision)
    updated = copy.deepcopy(state)
    for stage in ("review", "evaluation", "decision"):
        if stage not in updated["completed_stages"]:
            updated["completed_stages"].append(stage)
    updated.update(
        {
            "status": "completed",
            "stage": "decision",
            "next_action": "done",
            "updated_at_epoch": time.time(),
            "decision_digest": _canonical_digest(decision),
        }
    )
    _write_state(run_root, updated)
    return decision


def _validated_identity_artifact(value: Mapping[str, Any], expected_digest: str) -> dict:
    identity = _object(value, "identity artifact")
    fields = {"schema_version", "scope", "roots", "missing_roots", "files", "digest"}
    _closed(identity, fields, "identity artifact")
    _required(identity, fields, "identity artifact")
    if identity["schema_version"] != "cuda-workload-optimizer/project-identity-v1":
        raise ValidationError("identity artifact schema is invalid")
    computed = _canonical_digest(
        {
            "missing_roots": identity["missing_roots"],
            "files": identity["files"],
        }
    )
    if identity["digest"] != computed or computed != expected_digest:
        raise ValidationError("frozen identity artifact digest does not match state")
    return copy.deepcopy(identity)


def evaluate_change(run_dir: os.PathLike[str] | str) -> dict:
    """Verify the bounded diff, review it, run paired evaluation, and decide."""
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    state = read_run_state(run_root)
    if state["status"] == "completed":
        decision = load_json_object(run_root / "decision.json")
        if _canonical_digest(decision) != state.get("decision_digest"):
            raise ValidationError("decision artifact digest does not match state")
        return decision
    if state["next_action"] != "edit_then_evaluate":
        raise ValidationError("run is not ready to evaluate a ChangeSet")
    control = _load_frozen_control(run_root, state)
    change = validate_change_set(
        load_json_object(run_root / "change_set.json"), control
    )
    frozen_change = validate_change_set(
        load_json_object(run_root / "rounds" / "round-1" / "change_set.json"),
        control,
    )
    change_digest = _canonical_digest(change)
    if (
        change != frozen_change
        or change_digest != state.get("change_set_digest")
        or change["scope"] != state.get("change_scope")
    ):
        return _finish_rejected(
            run_root,
            state,
            control,
            scope=state["change_scope"],
            reason="frozen_artifact_drift",
            primary_status=None,
        )
    if time.time() > state["deadline_epoch"]:
        _atomic_json(
            run_root / "review.json",
            {
                "schema_version": "cuda-workload-optimizer/review-artifact-v1",
                "status": "skipped",
                "request_digest": None,
                "response": None,
                "execution": {"reason": "budget_expired"},
            },
        )
        _atomic_json(
            run_root / "evaluation.json",
            {
                "schema_version": "cuda-workload-optimizer/evaluation-v1",
                "status": "budget_expired",
            },
        )
        return _finish_rejected(
            run_root,
            state,
            control,
            scope=change["scope"],
            reason="budget_expired",
            primary_status=None,
        )
    workload = _normalize_frozen_workload(control)
    if workload.source_hash != state["workload_source_hash"]:
        decision = _finish_rejected(
            run_root,
            state,
            control,
            scope=change["scope"],
            reason="workload_identity_drift",
            primary_status=None,
        )
        if decision["status"] == "manual_recovery_required":
            return decision
        raise ValidationError("workload identity drifted before evaluation")
    try:
        before = _validated_identity_artifact(
            load_json_object(
                run_root / "rounds" / "round-1" / "before_identity.json"
            ),
            state["before_identity_digest"],
        )
    except (OSError, ValidationError, KeyError):
        return _finish_rejected(
            run_root,
            state,
            control,
            scope=change["scope"],
            reason="frozen_artifact_drift",
            primary_status=None,
        )
    after = _identity(control, change["scope"])
    changed = _changed_paths(before, after)
    if not changed:
        return _finish_rejected(
            run_root,
            state,
            control,
            scope=change["scope"],
            reason="no_scoped_changes",
            primary_status=None,
        )
    outside = [path for path in changed if not _path_allowed(path, change["paths"])]
    if outside:
        decision = _finish_rejected(
            run_root,
            state,
            control,
            scope=change["scope"],
            reason="change_set_path_escape",
            primary_status=None,
        )
        if decision["status"] == "manual_recovery_required":
            return decision
        raise ValidationError(
            "actual scoped diff is outside ChangeSet paths: " + ", ".join(outside)
        )
    _atomic_json(run_root / "rounds" / "round-1" / "after_identity.json", after)
    bound_candidate = copy.deepcopy(change["candidate"])
    candidate_binding = {
        "schema_version": "cuda-workload-optimizer/candidate-binding-v1",
        "candidate": bound_candidate,
        "candidate_digest": _canonical_digest(bound_candidate),
        "change_set_digest": change_digest,
        "after_identity_digest": after["digest"],
    }
    candidate_binding["digest"] = _canonical_digest(candidate_binding)
    _atomic_json(run_root / "candidate_binding.json", candidate_binding)
    (run_root / "candidate.diff").write_text(
        _candidate_diff(control, run_root, changed, change["scope"]), encoding="utf-8"
    )

    runtime = _BUDGET_RUNTIME[control["budget"]]
    evaluations: dict[str, dict] = {}

    def evaluate_pairs(stage: str, blocks: int) -> dict:
        timeout = (
            None
            if workload.kind == "python"
            else min(120, max(0.001, state["deadline_epoch"] - time.time()))
        )
        evaluation = _load_evaluate_module().evaluate_pairs(
            workload,
            control["baseline_candidate"],
            bound_candidate,
            blocks=blocks,
            retries=runtime["retries"],
            seed=0,
            timeout=timeout,
            deadline_epoch=state["deadline_epoch"],
            bootstrap_samples=runtime["bootstrap"],
        )
        evaluations[stage] = evaluation
        _atomic_json(run_root / f"{stage}_evaluation.json", evaluation)
        primary = evaluation.get("primary", {})
        constraints_passed = all(
            item.get("status") == "passed"
            for item in evaluation.get("constraints", [])
        )
        passed = evaluation.get("status") == "evaluated"
        if stage == "formal_paired":
            passed = (
                passed
                and primary.get("status") == "confirmed_win"
                and constraints_passed
            )
        return {
            "status": "passed" if passed else "failed",
            "estimate": primary.get("estimate_pct"),
            "lower_bound": primary.get("ci_low_pct"),
            "upper_bound": primary.get("ci_high_pct"),
        }

    remaining = max(0.001, state["deadline_epoch"] - time.time())
    soft_remaining = max(
        0.001,
        min(remaining, state.get("soft_target_epoch", time.time()) - time.time()),
    )
    gate_contract = {
        "soft_target_seconds": soft_remaining,
        "hard_ceiling_seconds": remaining,
        "minimum_effect": {
            "mechanism_us": 1.0,
            "service_pct": max(0.5, float(workload.objective["min_effect_pct"])),
        },
    }
    gate = _load_budget_module().CandidateGate(
        gate_contract,
        bound_candidate,
    )
    gate_result = gate.run(
        {
            "static_review": lambda: {"status": "passed"},
            "build_correctness": lambda: _run_correctness_commands(
                control, change, run_root
            ),
            "short_paired": lambda: evaluate_pairs(
                "short_paired", min(2, runtime["blocks"])
            ),
            "formal_paired": lambda: evaluate_pairs(
                "formal_paired", runtime["blocks"]
            ),
        }
    )
    evaluation = evaluations.get(
        "formal_paired",
        evaluations.get(
            "short_paired",
            {"schema_version": "cuda-workload-optimizer/evaluation-v1", "status": gate_result["stop_reason"]},
        ),
    )
    _atomic_json(run_root / "evaluation.json", evaluation)
    _atomic_json(run_root / "time_gate.json", gate_result)
    if gate_result["decision"] != "PROMOTE":
        rejection_reason = gate_result["stop_reason"]
        if rejection_reason == "hard_ceiling_admission_failed":
            rejection_reason = "budget_expired"
        if evaluations and evaluation.get("status") != "evaluated":
            rejection_reason = "workload_failed"
        _atomic_json(
            run_root / "review.json",
            {
                "schema_version": "cuda-workload-optimizer/review-artifact-v1",
                "status": "skipped",
                "request_digest": None,
                "response": None,
                "execution": {"reason": gate_result["stop_reason"]},
            },
        )
        return _finish_rejected(
            run_root,
            state,
            control,
            scope=change["scope"],
            reason=rejection_reason,
            primary_status=evaluation.get("primary", {}).get("status"),
            time_gate=gate_result,
        )

    review_change(
        control,
        run_root,
        change,
        deadline_epoch=min(state["deadline_epoch"], time.time() + 180),
    )
    if time.time() > state["deadline_epoch"]:
        return _finish_rejected(
            run_root,
            state,
            control,
            scope=change["scope"],
            reason="budget_expired",
            primary_status=evaluation.get("primary", {}).get("status"),
        )
    if _identity(control, change["scope"])["digest"] != after["digest"]:
        decision = _finish_rejected(
            run_root,
            state,
            control,
            scope=change["scope"],
            reason="scoped_identity_drift",
            primary_status=evaluation.get("primary", {}).get("status"),
        )
        if decision["status"] == "manual_recovery_required":
            return decision
        raise ValidationError("scoped identity drifted during paired evaluation")
    primary_status = evaluation.get("primary", {}).get("status")
    constraints = evaluation.get("constraints", [])
    promoted = (
        evaluation.get("status") == "evaluated"
        and primary_status == "confirmed_win"
        and all(item.get("status") == "passed" for item in constraints)
        and control.get("evaluation_gate", "promotion") == "promotion"
    )
    if not promoted:
        if evaluation.get("status") != "evaluated":
            reason = "workload_failed"
        elif primary_status != "confirmed_win":
            reason = "primary_not_confirmed"
        elif control.get("evaluation_gate", "promotion") == "reject_only":
            reason = "reject_only_stage_cannot_promote"
        else:
            reason = "constraint_failed"
        return _finish_rejected(
            run_root,
            state,
            control,
            scope=change["scope"],
            reason=reason,
            primary_status=primary_status,
        )

    decision = {
        "schema_version": "cuda-workload-optimizer/decision-v1",
        "status": "promoted",
        "reason": "paired_workload_win",
        "primary_status": primary_status,
        "rolled_back": False,
        "change_set_digest": change_digest,
        "candidate_binding_digest": candidate_binding["digest"],
        "after_identity_digest": after["digest"],
        "evaluation_digest": _canonical_digest(evaluation),
        "elapsed_seconds": gate_result["elapsed_seconds"],
        "stop_reason": gate_result["stop_reason"],
        "skipped_expensive_stages": gate_result["skipped_expensive_stages"],
    }
    _atomic_json(run_root / "decision.json", decision)
    updated = copy.deepcopy(state)
    for stage in ("review", "evaluation", "decision"):
        if stage not in updated["completed_stages"]:
            updated["completed_stages"].append(stage)
    updated.update(
        {
            "status": "completed",
            "stage": "decision",
            "next_action": "done",
            "updated_at_epoch": time.time(),
            "decision_digest": _canonical_digest(decision),
        }
    )
    _write_state(run_root, updated)
    return decision


def resume_run(run_dir: os.PathLike[str] | str) -> dict:
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    state = read_run_state(run_root)
    _load_frozen_control(run_root, state)
    if state["next_action"] == "collect_evidence":
        return collect_active_diagnosis_evidence(
            _load_frozen_control(run_root), run_root
        )
    if state["next_action"] in {
        "propose_hypotheses",
        "evidence_gap",
        "register_change",
        "edit_then_evaluate",
        "done",
        "manual_recovery",
    }:
        return state
    return start_run(_load_frozen_control(run_root), run_root)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and run bounded GPU workload optimization rounds."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate controller JSON")
    validate.add_argument("--control", required=True)
    validate.add_argument("--change-set")
    probe = subparsers.add_parser("probe", help="collect normalized probe evidence")
    probe.add_argument("--control", required=True)
    probe.add_argument("--run-dir", required=True)
    diagnose_parser = subparsers.add_parser(
        "diagnose", help="classify stored normalized probe evidence"
    )
    diagnose_parser.add_argument("--run-dir", required=True)
    review = subparsers.add_parser("review", help="request optional advisory review")
    review.add_argument("--control", required=True)
    review.add_argument("--run-dir", required=True)
    review.add_argument("--change-set", required=True)
    run = subparsers.add_parser("run", help="collect baseline evidence and diagnosis")
    run.add_argument("--control", required=True)
    run.add_argument("--run-dir", required=True)
    status = subparsers.add_parser("status", help="read the current run checkpoint")
    status.add_argument("--run-dir", required=True)
    register = subparsers.add_parser(
        "register-change", help="freeze and register a bounded ChangeSet"
    )
    register.add_argument("--control", required=True)
    register.add_argument("--run-dir", required=True)
    register.add_argument("--change-set", required=True)
    diagnosis_proposal = subparsers.add_parser(
        "register-diagnosis",
        help="validate and freeze an active diagnosis proposal",
    )
    diagnosis_proposal.add_argument("--control", required=True)
    diagnosis_proposal.add_argument("--run-dir", required=True)
    diagnosis_proposal.add_argument("--hypothesis-set", required=True)
    diagnosis_proposal.add_argument("--request-set", required=True)
    collect_evidence = subparsers.add_parser(
        "collect-evidence",
        help="execute the selected frozen active-diagnosis evidence action",
    )
    collect_evidence.add_argument("--control", required=True)
    collect_evidence.add_argument("--run-dir", required=True)
    evaluate = subparsers.add_parser(
        "evaluate", help="verify, evaluate, promote, or roll back a candidate"
    )
    evaluate.add_argument("--run-dir", required=True)
    resume = subparsers.add_parser("resume", help="resume from the last checkpoint")
    resume.add_argument("--run-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            control = validate_control_manifest(
                load_json_object(args.control), args.control
            )
            if args.change_set:
                validate_change_set(load_json_object(args.change_set), control)
            print(json.dumps({"status": "valid"}, sort_keys=True))
            return 0
        if args.command == "probe":
            values = run_probes(load_json_object(args.control), args.run_dir)
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "probe_count": len(values),
                        "available_probe_count": sum(
                            item["status"] in {"ok", "degraded"} for item in values
                        ),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "diagnose":
            print(json.dumps(diagnose_run(args.run_dir), sort_keys=True))
            return 0
        if args.command == "review":
            artifact = review_change(
                load_json_object(args.control),
                args.run_dir,
                load_json_object(args.change_set),
            )
            print(json.dumps(artifact, sort_keys=True))
            return 0
        if args.command == "run":
            control = validate_control_manifest(load_json_object(args.control))
            if control["schema_version"] != CONTROL_SCHEMA_V2:
                raise ValidationError(
                    "new controller runs require control-v2 with readiness_contract; "
                    "control-v1 remains available for validate and historical resume"
                )
            print(
                json.dumps(
                    start_run(control, args.run_dir),
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "status":
            print(json.dumps(read_run_state(args.run_dir), sort_keys=True))
            return 0
        if args.command == "register-change":
            print(
                json.dumps(
                    register_change(
                        load_json_object(args.control),
                        args.run_dir,
                        load_json_object(args.change_set),
                    ),
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "register-diagnosis":
            print(
                json.dumps(
                    register_active_diagnosis_proposal(
                        load_json_object(args.control),
                        args.run_dir,
                        load_json_object(args.hypothesis_set),
                        load_json_object(args.request_set),
                    ),
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "collect-evidence":
            print(
                json.dumps(
                    collect_active_diagnosis_evidence(
                        load_json_object(args.control), args.run_dir
                    ),
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "evaluate":
            print(json.dumps(evaluate_change(args.run_dir), sort_keys=True))
            return 0
        if args.command == "resume":
            print(json.dumps(resume_run(args.run_dir), sort_keys=True))
            return 0
    except ValidationError as error:
        print(f"validation error: {error}", file=sys.stderr)
        return 2
    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
