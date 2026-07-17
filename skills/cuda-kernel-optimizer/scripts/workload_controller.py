#!/usr/bin/env python3
"""Strict contracts and orchestration entry point for workload optimization."""

from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


CONTROL_SCHEMA = "cuda-workload-optimizer/control-v1"
CHANGE_SCHEMA = "cuda-workload-optimizer/change-v1"
_BUDGETS = {"fast", "balanced", "thorough"}
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
_BUDGET_RUNTIME = {
    "fast": {"deadline_seconds": 900, "blocks": 3, "retries": 0, "bootstrap": 200},
    "balanced": {
        "deadline_seconds": 3600,
        "blocks": 5,
        "retries": 1,
        "bootstrap": 1000,
    },
    "thorough": {
        "deadline_seconds": 14400,
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


def validate_control_manifest(value: Mapping[str, Any], source_path=None) -> dict:
    """Validate and detach the closed v2.4 controller manifest."""
    control = _object(value, "control")
    allowed = {
        "schema_version",
        "project_root",
        "workload_manifest",
        "baseline_candidate",
        "budget",
        "mutation",
        "probes",
        "reviewer",
    }
    required = allowed - {"reviewer"}
    _closed(control, allowed, "control")
    _required(control, required, "control")
    if control["schema_version"] != CONTROL_SCHEMA:
        raise ValidationError(f"schema_version must be {CONTROL_SCHEMA}")

    project_root = _absolute(control["project_root"], "project_root")
    workload_manifest = _absolute(
        control["workload_manifest"], "workload_manifest"
    )
    if not _is_within(workload_manifest, project_root):
        raise ValidationError("workload_manifest must be inside project_root")
    baseline = _object(control["baseline_candidate"], "baseline_candidate")
    if not baseline:
        raise ValidationError("baseline_candidate must not be empty")
    _json_copy(baseline, "baseline_candidate", reject_sensitive=True)
    if control["budget"] not in _BUDGETS:
        raise ValidationError("budget must be fast, balanced, or thorough")

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

    return _json_copy(control, "control", reject_sensitive=True)


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
        _load_evaluate_module()
        _WORKLOAD_MODULE = sys.modules["workload_adapter"]
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
    try:
        output_path.unlink()
    except FileNotFoundError:
        pass
    environment, secret_values = _probe_environment(
        {
            "CUDA_OPTIMIZER_OUTPUT": str(output_path),
            "CUDA_OPTIMIZER_RUN_DIR": str(run_root),
            "CUDA_OPTIMIZER_PROJECT_ROOT": normalized_control["project_root"],
        }
    )
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
    blocks = {"fast": 3, "balanced": 5, "thorough": 9}[normalized["budget"]]
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


def start_run(
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
            "register_change",
            "edit_then_evaluate",
            "done",
            "manual_recovery",
        }:
            return state
    else:
        run_root.mkdir(parents=True, exist_ok=True)
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
            "stage": "baseline",
            "round": 1,
            "completed_stages": [],
            "next_action": "baseline",
            "control_digest": control_digest,
            "workload_source_hash": workload.source_hash,
            "started_at_epoch": now,
            "updated_at_epoch": now,
            "deadline_epoch": now + runtime["deadline_seconds"],
        }
        _atomic_json(run_root / "control_manifest.json", normalized)
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

    if "baseline" not in state["completed_stages"]:
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
        _atomic_json(run_root / "baseline" / "observation.json", baseline)
        if baseline["status"] != "measured":
            raise ValidationError("baseline workload failed; see baseline/observation.json")
        state = _advance(
            run_root, state, "baseline", stage="probes", next_action="probes"
        )
    _check_deadline(state)
    if "probes" not in state["completed_stages"]:
        run_probes(
            normalized,
            run_root,
            deadline_epoch=state["deadline_epoch"],
        )
        state = _advance(
            run_root, state, "probes", stage="diagnosis", next_action="diagnosis"
        )
    _check_deadline(state)
    if "diagnosis" not in state["completed_stages"]:
        diagnose_run(run_root)
        state = _advance(
            run_root,
            state,
            "diagnosis",
            stage="change",
            next_action="register_change",
        )
    return state


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

    correctness = _run_correctness_commands(control, change, run_root)
    if correctness["status"] != "passed":
        _atomic_json(
            run_root / "review.json",
            {
                "schema_version": "cuda-workload-optimizer/review-artifact-v1",
                "status": "skipped",
                "request_digest": None,
                "response": None,
                "execution": {"reason": "correctness_failed"},
            },
        )
        _atomic_json(
            run_root / "evaluation.json",
            {
                "schema_version": "cuda-workload-optimizer/evaluation-v1",
                "status": "correctness_failed",
            },
        )
        return _finish_rejected(
            run_root,
            state,
            control,
            scope=change["scope"],
            reason="correctness_failed",
            primary_status=None,
        )

    review_change(
        control,
        run_root,
        change,
        deadline_epoch=state["deadline_epoch"],
    )
    if time.time() > state["deadline_epoch"]:
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
    runtime = _BUDGET_RUNTIME[control["budget"]]
    timeout = (
        None
        if workload.kind == "python"
        else min(120, max(0.001, state["deadline_epoch"] - time.time()))
    )
    evaluation = _load_evaluate_module().evaluate_pairs(
        workload,
        control["baseline_candidate"],
        bound_candidate,
        blocks=runtime["blocks"],
        retries=runtime["retries"],
        seed=0,
        timeout=timeout,
        deadline_epoch=state["deadline_epoch"],
        bootstrap_samples=runtime["bootstrap"],
    )
    _atomic_json(run_root / "evaluation.json", evaluation)
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
    )
    if not promoted:
        if evaluation.get("status") != "evaluated":
            reason = "workload_failed"
        elif primary_status != "confirmed_win":
            reason = "primary_not_confirmed"
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
    if state["next_action"] in {
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
            print(
                json.dumps(
                    start_run(load_json_object(args.control), args.run_dir),
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
