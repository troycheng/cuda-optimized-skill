#!/usr/bin/env python3
"""Strict contracts and orchestration entry point for workload optimization."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
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
    r"(?i)([A-Z0-9_]*(?:API[_-]?KEY|AUTH|COOKIE|CREDENTIAL|PASSWORD|SECRET|TOKEN)[A-Z0-9_]*)\s*[:=]\s*([^\s,;]+)"
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
    environment_root = _absolute(mutation["environment_root"], "environment_root")
    if _is_within(environment_root, project_root):
        raise ValidationError("environment_root must be outside project_root")
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
        fields = {"argv", "timeout_seconds"}
        _closed(reviewer, fields, "reviewer")
        _required(reviewer, fields, "reviewer")
        _argv(reviewer["argv"], "reviewer")
        _timeout(reviewer["timeout_seconds"], "reviewer.timeout_seconds")

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


def _stop_group(process: Any) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.wait(timeout=0.25)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
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
    result = _LOG_SECRET.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
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
            exit_code = process.wait(timeout=float(selected["timeout_seconds"]))
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_group(process)
            exit_code = process.returncode
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
                f"probe exceeded {selected['timeout_seconds']} seconds",
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


def run_probes(control: Mapping[str, Any], run_dir: os.PathLike[str] | str) -> list[dict]:
    normalized = validate_control_manifest(control)
    return [run_probe(probe, normalized, run_dir) for probe in normalized["probes"]]


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
    except ValidationError as error:
        print(f"validation error: {error}", file=sys.stderr)
        return 2
    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
