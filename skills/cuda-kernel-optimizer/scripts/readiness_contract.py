#!/usr/bin/env python3
"""Validate the bounded capability contract used before workload diagnosis."""

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


SCHEMA_VERSION = "cuda-workload-optimizer/readiness-contract-v1"
REQUESTED_CLAIMS = {"kernel", "workload", "serving"}
NECESSITIES = {"required", "diagnostic", "optional"}
CONTROL_SCOPES = {"project", "isolated_environment", "host"}
PHASES = {"foundation", "workload"}
KINDS = {
    "target_compile",
    "gpu_execute",
    "nsys_trace",
    "ncu_counters",
    "sanitizer",
    "sass",
    "benchmark_noise",
    "workload_smoke",
    "rollback",
}
REMEDIATION_MODES = {"none", "user_action", "isolated_pip"}

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _load_artifact_store():
    path = Path(__file__).with_name("artifact_store.py")
    spec = importlib.util.spec_from_file_location(
        "cuda_readiness_artifact_store", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_ARTIFACT_STORE = _load_artifact_store()


class ValidationError(ValueError):
    """Raised when a readiness contract is open, unsafe, or inconsistent."""


def _pairs_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_number(token: str):
    raise ValidationError(f"JSON number must be finite: {token}")


def load_contract(path: str | os.PathLike) -> dict:
    """Read one strict JSON object without following symlinks."""
    try:
        raw = _ARTIFACT_STORE.read_regular_bytes(path)
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_invalid_number,
        )
    except ValidationError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValidationError(
            f"invalid or unsafe readiness contract {path}: {error}"
        ) from error
    if type(value) is not dict:
        raise ValidationError("readiness contract root must be an object")
    return value


def _closed(
    value: Any,
    required: set[str],
    name: str,
    *,
    optional: set[str] | None = None,
) -> dict:
    if type(value) is not dict:
        raise ValidationError(f"{name} must be an object")
    allowed = required | (optional or set())
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValidationError(
            f"{name} contains unknown fields: {', '.join(unknown)}"
        )
    missing = sorted(required - set(value))
    if missing:
        raise ValidationError(
            f"{name} is missing required fields: {', '.join(missing)}"
        )
    return value


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


def _enum(value: Any, allowed: set[str], field: str) -> str:
    if type(value) is not str or value not in allowed:
        raise ValidationError(
            f"{field} must be one of: {', '.join(sorted(allowed))}"
        )
    return value


def _finite_positive(value: Any, field: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a positive finite number")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValidationError(f"{field} must be a positive finite number")
    return value


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{field} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{field} must be a non-negative integer")
    return value


def _safe_root(value: str | os.PathLike, field: str) -> Path:
    raw = Path(os.path.expanduser(os.fspath(value)))
    if not raw.is_absolute():
        raise ValidationError(f"{field} must be an absolute path")
    root = Path(os.path.abspath(raw))
    marker = root / ".readiness-root-check"
    try:
        directory_fd, _leaf, _target = _ARTIFACT_STORE._open_parent_directory(
            marker, create=False
        )
    except (OSError, ValueError) as error:
        raise ValidationError(f"{field} contains a symlink or is unsafe") from error
    else:
        os.close(directory_fd)
    if not root.is_dir():
        raise ValidationError(f"{field} must be an existing directory")
    return root


def _absolute_inside(value: Any, root: Path, field: str) -> Path:
    text = _string(value, field)
    expanded = Path(os.path.expanduser(text))
    if not expanded.is_absolute():
        raise ValidationError(f"{field} must be an absolute path")
    path = Path(os.path.abspath(expanded))
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValidationError(f"{field} must be inside {root.name} root") from error
    return path


def _safe_regular(path: Path, field: str, *, executable: bool = False) -> None:
    fd = None
    parent_fd = None
    try:
        parent_fd, leaf, _target = _ARTIFACT_STORE._open_parent_directory(
            path, create=False
        )
        fd = os.open(
            leaf,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValidationError(f"{field} must be a regular file")
        if executable and not os.access(path, os.X_OK):
            raise ValidationError(f"{field} must be executable")
    except ValidationError:
        raise
    except (OSError, ValueError) as error:
        raise ValidationError(f"{field} contains a symlink or is unsafe") from error
    finally:
        if fd is not None:
            os.close(fd)
        if parent_fd is not None:
            os.close(parent_fd)


def _safe_isolated_python(path: Path, field: str) -> None:
    """Allow only the leaf symlink shape used by standard isolated venvs."""
    parent_fd = None
    try:
        parent_fd, leaf, _target = _ARTIFACT_STORE._open_parent_directory(
            path, create=False
        )
        metadata = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISREG(metadata.st_mode):
            _safe_regular(path, field, executable=True)
            return
        if not stat.S_ISLNK(metadata.st_mode):
            raise ValidationError(f"{field} must be a regular file or venv symlink")
        resolved = Path(os.path.realpath(path))
        _safe_regular(resolved, f"{field} target", executable=True)
        if not os.access(path, os.X_OK):
            raise ValidationError(f"{field} must resolve to an executable")
    except ValidationError:
        raise
    except (OSError, ValueError) as error:
        raise ValidationError(f"{field} contains an unsafe symlink") from error
    finally:
        if parent_fd is not None:
            os.close(parent_fd)


def _validate_probe(value: Any, field: str) -> dict:
    probe = _closed(value, {"argv", "timeout_seconds"}, field)
    argv = probe["argv"]
    if type(argv) is not list or not argv:
        raise ValidationError(f"{field}.argv must be a non-empty array")
    if len(argv) > 128:
        raise ValidationError(f"{field}.argv exceeds 128 entries")
    for index, argument in enumerate(argv):
        _string(argument, f"{field}.argv[{index}]", max_length=16384)
    _finite_positive(probe["timeout_seconds"], f"{field}.timeout_seconds")
    return probe


def _validate_remediation(
    value: Any,
    *,
    field: str,
    control_scope: str,
    project_root: Path,
    environment_root: Path,
) -> dict:
    if type(value) is not dict:
        raise ValidationError(f"{field} must be an object")
    mode = _enum(value.get("mode"), REMEDIATION_MODES, f"{field}.mode")
    if mode == "none":
        _closed(value, {"mode"}, field)
    elif mode == "user_action":
        _closed(value, {"mode", "message"}, field)
        _string(value["message"], f"{field}.message", max_length=4096)
    else:
        _closed(
            value,
            {
                "mode",
                "authorization_id",
                "python",
                "requirements_file",
                "requirements_sha256",
                "timeout_seconds",
            },
            field,
        )
        if control_scope != "isolated_environment":
            if control_scope == "host":
                raise ValidationError(
                    "host remediation must remain none or user_action"
                )
            raise ValidationError(
                "isolated_pip requires isolated_environment control_scope"
            )
        _identifier(value["authorization_id"], f"{field}.authorization_id")
        python = _absolute_inside(
            value["python"], environment_root, f"{field}.python"
        )
        requirements = _absolute_inside(
            value["requirements_file"], project_root, f"{field}.requirements_file"
        )
        if _SHA256.fullmatch(
            _string(value["requirements_sha256"], f"{field}.requirements_sha256")
        ) is None:
            raise ValidationError(f"{field}.requirements_sha256 must be SHA-256")
        _finite_positive(value["timeout_seconds"], f"{field}.timeout_seconds")
        _safe_isolated_python(python, f"{field}.python")
        _safe_regular(requirements, f"{field}.requirements_file")
    return value


def validate_contract(
    value: Mapping[str, Any], *, project_root: Path, environment_root: Path
) -> dict:
    """Return a detached validated contract without running any capability."""
    project = _safe_root(project_root, "project_root")
    environment = _safe_root(environment_root, "environment_root")
    contract = _closed(
        value,
        {"schema_version", "requested_claim", "budget", "requirements"},
        "readiness contract",
    )
    if contract["schema_version"] != SCHEMA_VERSION:
        raise ValidationError(f"schema_version must be {SCHEMA_VERSION}")
    _enum(contract["requested_claim"], REQUESTED_CLAIMS, "requested_claim")

    budget = _closed(
        contract["budget"], {"max_seconds", "max_repairs"}, "budget"
    )
    _finite_positive(budget["max_seconds"], "budget.max_seconds")
    _nonnegative_integer(budget["max_repairs"], "budget.max_repairs")

    requirements = contract["requirements"]
    if type(requirements) is not list or not requirements:
        raise ValidationError("requirements must be a non-empty array")
    if len(requirements) > 128:
        raise ValidationError("requirements exceeds 128 entries")
    seen_ids = set()
    seen_authorization_ids = set()
    for index, item in enumerate(requirements):
        field = f"requirements[{index}]"
        requirement = _closed(
            item,
            {
                "id",
                "necessity",
                "control_scope",
                "phase",
                "kind",
                "max_age_seconds",
                "probe",
                "remediation",
            },
            field,
        )
        requirement_id = _identifier(requirement["id"], f"{field}.id")
        if requirement_id in seen_ids:
            raise ValidationError(f"duplicate requirement id: {requirement_id}")
        seen_ids.add(requirement_id)
        _enum(requirement["necessity"], NECESSITIES, f"{field}.necessity")
        scope = _enum(
            requirement["control_scope"], CONTROL_SCOPES, f"{field}.control_scope"
        )
        _enum(requirement["phase"], PHASES, f"{field}.phase")
        _enum(requirement["kind"], KINDS, f"{field}.kind")
        _positive_integer(
            requirement["max_age_seconds"], f"{field}.max_age_seconds"
        )
        _validate_probe(requirement["probe"], f"{field}.probe")
        _validate_remediation(
            requirement["remediation"],
            field=f"{field}.remediation",
            control_scope=scope,
            project_root=project,
            environment_root=environment,
        )
        remediation = requirement["remediation"]
        if remediation["mode"] == "isolated_pip":
            authorization_id = remediation["authorization_id"]
            if authorization_id in seen_authorization_ids:
                raise ValidationError(
                    f"duplicate authorization_id: {authorization_id}"
                )
            seen_authorization_ids.add(authorization_id)

    try:
        detached = copy.deepcopy(contract)
        json.dumps(detached, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"contract must contain finite JSON values: {error}") from error
    return detached


def contract_digest(value: Mapping[str, Any]) -> str:
    """Return the SHA-256 of canonical strict JSON contract bytes."""
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValidationError(f"contract is not strict JSON: {error}") from error
    return hashlib.sha256(payload).hexdigest()
