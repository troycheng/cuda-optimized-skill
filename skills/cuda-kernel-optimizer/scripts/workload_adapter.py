#!/usr/bin/env python3
"""Normalize and execute user-owned end-to-end workloads.

This module deliberately has no workload discovery or download behavior.  A
full-mode run exists only when the user supplies a complete Python adapter,
command, or manifest; otherwise callers remain in kernel-only mode.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import shlex
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


REQUIRED_ADAPTER_CALLS = (
    "prepare",
    "validate",
    "benchmark",
    "metrics",
    "cleanup",
)

_OBJECTIVE_FIELDS = {"primary_metric", "min_effect_pct", "constraints"}
_PRIMARY_FIELDS = {"name", "direction"}
_CONSTRAINT_FIELDS = {"name", "max_regression_pct"}
_MANIFEST_FIELDS = {"kind", "source", "objective", "cases"}
_DIAGNOSTIC_LIMIT = 4096
_OUTPUT_LIMIT_BYTES = 1024 * 1024


class _FrozenDict(dict):
    """JSON-compatible immutable dict used inside a frozen WorkloadSpec."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("WorkloadSpec values are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, memo):
        return self


class _FrozenList(list):
    """JSON-compatible immutable list used inside a frozen WorkloadSpec."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("WorkloadSpec values are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable

    def __deepcopy__(self, memo):
        return self


def _freeze_json(value):
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze_json(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json(item) for item in value)
    return copy.deepcopy(value)


@dataclass(frozen=True)
class WorkloadSpec:
    kind: str
    source: str | list[str]
    objective: dict
    cases: tuple[dict, ...]
    source_hash: str

    def __post_init__(self) -> None:
        if isinstance(self.source, list) and not isinstance(self.source, _FrozenList):
            object.__setattr__(self, "source", _freeze_json(self.source))
        object.__setattr__(self, "objective", _freeze_json(self.objective))
        object.__setattr__(
            self,
            "cases",
            tuple(_freeze_json(case) for case in self.cases),
        )


def _nonempty_string(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _finite_nonnegative(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return number


def _unknown_fields(value: Mapping, allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{field} contains unknown fields: {', '.join(unknown)}")


def validate_objective(value) -> dict:
    """Return a strict, normalized deep copy of an objective JSON object."""
    if not isinstance(value, dict):
        raise ValueError("objective must be a JSON object")
    _unknown_fields(value, _OBJECTIVE_FIELDS, "objective")
    missing = sorted(_OBJECTIVE_FIELDS - set(value))
    if missing:
        raise ValueError(f"objective missing required fields: {', '.join(missing)}")

    primary = value["primary_metric"]
    if not isinstance(primary, dict):
        raise ValueError("primary_metric must be an object")
    _unknown_fields(primary, _PRIMARY_FIELDS, "primary_metric")
    missing_primary = sorted(_PRIMARY_FIELDS - set(primary))
    if missing_primary:
        raise ValueError(
            "primary_metric missing required fields: " + ", ".join(missing_primary)
        )
    primary_name = _nonempty_string(primary["name"], "primary_metric.name")
    direction = primary["direction"]
    if direction not in ("lower", "higher"):
        raise ValueError("primary_metric.direction must be 'lower' or 'higher'")

    constraints = value["constraints"]
    if not isinstance(constraints, list):
        raise ValueError("constraints must be a list")
    normalized_constraints = []
    seen = set()
    for index, constraint in enumerate(constraints):
        field = f"constraints[{index}]"
        if not isinstance(constraint, dict):
            raise ValueError(f"{field} must be an object")
        _unknown_fields(constraint, _CONSTRAINT_FIELDS, field)
        missing_constraint = sorted(_CONSTRAINT_FIELDS - set(constraint))
        if missing_constraint:
            raise ValueError(
                f"{field} missing required fields: {', '.join(missing_constraint)}"
            )
        name = _nonempty_string(constraint["name"], f"{field}.name")
        if name in seen:
            raise ValueError(f"constraints contains duplicate name: {name}")
        seen.add(name)
        normalized_constraints.append(
            {
                "name": name,
                "max_regression_pct": _finite_nonnegative(
                    constraint["max_regression_pct"],
                    f"{field}.max_regression_pct",
                ),
            }
        )

    return {
        "primary_metric": {"name": primary_name, "direction": direction},
        "min_effect_pct": _finite_nonnegative(
            value["min_effect_pct"], "min_effect_pct"
        ),
        "constraints": normalized_constraints,
    }


def _read_json_object(path, field: str) -> tuple[Path, dict]:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError) as error:
        raise ValueError(f"{field} file does not exist: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"{field} must be a regular file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} must contain valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return resolved, value


def load_objective(path) -> dict:
    _, value = _read_json_object(path, "objective")
    return validate_objective(value)


def _regular_file(path, field: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError) as error:
        raise ValueError(f"{field} file does not exist: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"{field} must be a regular file: {resolved}")
    return resolved


def load_python_adapter(path) -> ModuleType:
    """Load a user adapter after validating its complete lifecycle surface."""
    resolved = _regular_file(path, "workload adapter")
    module_name = "_cuda_optimizer_user_workload_" + hashlib.sha256(
        str(resolved).encode("utf-8")
    ).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load workload adapter: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except KeyboardInterrupt:
        sys.modules.pop(module_name, None)
        raise
    except BaseException as error:
        sys.modules.pop(module_name, None)
        raise ValueError(
            f"failed to import workload adapter {resolved}: "
            f"{type(error).__name__}: {error}"
        ) from error

    missing = [
        name
        for name in REQUIRED_ADAPTER_CALLS
        if not callable(getattr(module, name, None))
    ]
    if missing:
        sys.modules.pop(module_name, None)
        raise ValueError(
            "workload adapter missing required callables: " + ", ".join(missing)
        )
    return module


def _command_argv(command) -> list[str]:
    if isinstance(command, str):
        try:
            argv = shlex.split(command)
        except ValueError as error:
            raise ValueError(f"workload command has invalid quoting: {error}") from error
    elif isinstance(command, Sequence) and not isinstance(
        command, (str, bytes, bytearray)
    ):
        argv = list(command)
    else:
        raise ValueError("workload command must be a string or sequence of strings")
    if not argv:
        raise ValueError("workload command must not be empty")
    for index, argument in enumerate(argv):
        if not isinstance(argument, str) or not argument.strip():
            raise ValueError(
                f"workload command argument {index} must be a non-empty string"
            )
    return argv


def _canonical_json(value) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"workload content must be JSON-compatible: {error}") from error


def _file_fingerprints(paths: Sequence[Path]) -> list[dict]:
    fingerprints = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        fingerprints.append({"path": str(resolved), "sha256": digest})
    return sorted(fingerprints, key=lambda item: item["path"])


def _referenced_command_files(argv: Sequence[str], base_dir: Path) -> list[Path]:
    files = []
    for argument in argv:
        if argument.startswith("-"):
            continue
        candidate = Path(argument).expanduser()
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        try:
            if candidate.is_file():
                files.append(candidate.resolve())
        except OSError:
            continue
    return files


def _source_hash(payload: dict, paths: Sequence[Path]) -> str:
    frozen = copy.deepcopy(payload)
    frozen["source_files"] = _file_fingerprints(paths)
    return hashlib.sha256(_canonical_json(frozen)).hexdigest()


def _normalized_source_hash(
    kind: str,
    source: str | Sequence[str],
    objective: Mapping,
    cases: Sequence[Mapping],
) -> str:
    """Hash the executable contract and the current referenced source bytes."""
    if kind == "python":
        if not isinstance(source, str):
            raise ValueError("Python workload source must be a file path")
        source_files = [_regular_file(source, "workload adapter")]
    elif kind == "command":
        if isinstance(source, str):
            raise ValueError("command workload source must be an argv list")
        source_files = _referenced_command_files(source, Path.cwd())
    else:
        raise ValueError(f"unknown workload kind: {kind}")
    payload = {
        "executor_kind": kind,
        "source": source,
        "objective": objective,
        "cases": cases,
    }
    return _source_hash(payload, source_files)


def _adapter_objective(adapter: ModuleType, path: Path) -> dict:
    try:
        value = adapter.metrics()
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        raise ValueError(
            f"workload adapter metrics() failed for {path}: "
            f"{type(error).__name__}: {error}"
        ) from error
    return validate_objective(value)


def _normalize_cases(value) -> tuple[dict, ...]:
    if not isinstance(value, list):
        raise ValueError("manifest cases must be a list of objects")
    normalized = []
    for index, case in enumerate(value):
        if not isinstance(case, dict):
            raise ValueError(f"manifest cases[{index}] must be an object")
        normalized.append(copy.deepcopy(case))
    _canonical_json(normalized)
    return tuple(normalized)


def _resolve_manifest_command(argv: list[str], base_dir: Path) -> list[str]:
    resolved = []
    for argument in argv:
        if argument.startswith("-"):
            resolved.append(argument)
            continue
        path = Path(argument).expanduser()
        candidate = path if path.is_absolute() else base_dir / path
        try:
            exists = candidate.exists()
        except OSError:
            exists = False
        if exists:
            resolved.append(str(candidate.resolve()))
        else:
            resolved.append(argument)
    return resolved


def _normalize_manifest(manifest_path, objective_path) -> WorkloadSpec:
    resolved_manifest, manifest = _read_json_object(manifest_path, "workload manifest")
    _unknown_fields(manifest, _MANIFEST_FIELDS, "workload manifest")
    missing = sorted({"kind", "source", "cases"} - set(manifest))
    if missing:
        raise ValueError(
            "workload manifest missing required fields: " + ", ".join(missing)
        )
    kind = manifest["kind"]
    if kind not in ("python", "command"):
        raise ValueError("workload manifest kind must be 'python' or 'command'")
    cases = _normalize_cases(manifest["cases"])

    embedded = manifest.get("objective")
    if embedded is not None and objective_path is not None:
        raise ValueError("conflicting objective: manifest and --objective both provided")
    if embedded is None and objective_path is None:
        raise ValueError("workload manifest requires an objective")
    objective = (
        validate_objective(embedded)
        if embedded is not None
        else load_objective(objective_path)
    )

    base_dir = resolved_manifest.parent
    # The normalized payload below covers every allowed manifest field.  Do
    # not hash raw JSON bytes: insignificant key order and whitespace must not
    # change the frozen workload identity.
    if kind == "python":
        raw_source = _nonempty_string(manifest["source"], "manifest source")
        source_path = Path(raw_source).expanduser()
        if not source_path.is_absolute():
            source_path = base_dir / source_path
        source_path = _regular_file(source_path, "manifest Python source")
        adapter = load_python_adapter(source_path)
        adapter_objective = _adapter_objective(adapter, source_path)
        if adapter_objective != objective:
            raise ValueError(
                "conflicting objective: manifest objective does not match "
                "Python adapter metrics()"
            )
        objective = adapter_objective
        source: str | list[str] = str(source_path)
    else:
        source = _resolve_manifest_command(
            _command_argv(manifest["source"]), base_dir
        )

    return WorkloadSpec(
        kind=kind,
        source=source,
        objective=objective,
        cases=cases,
        source_hash=_normalized_source_hash(kind, source, objective, cases),
    )


def normalize_workload(
    *,
    workload=None,
    workload_cmd=None,
    workload_manifest=None,
    objective=None,
) -> WorkloadSpec | None:
    """Normalize exactly one user-provided workload form, or return None."""
    selected = [
        name
        for name, value in (
            ("--workload", workload),
            ("--workload-cmd", workload_cmd),
            ("--workload-manifest", workload_manifest),
        )
        if value is not None
    ]
    if len(selected) > 1:
        raise ValueError(
            "exactly one workload form may be provided: " + ", ".join(selected)
        )
    if not selected:
        if objective is not None:
            raise ValueError("--objective cannot be used without a workload")
        return None

    if workload is not None:
        if objective is not None:
            raise ValueError(
                "conflicting objective: Python workload objective comes from metrics()"
            )
        source_path = _regular_file(workload, "workload adapter")
        adapter = load_python_adapter(source_path)
        normalized_objective = _adapter_objective(adapter, source_path)
        cases: tuple[dict, ...] = ()
        return WorkloadSpec(
            kind="python",
            source=str(source_path),
            objective=normalized_objective,
            cases=cases,
            source_hash=_normalized_source_hash(
                "python", str(source_path), normalized_objective, cases
            ),
        )

    if workload_cmd is not None:
        if objective is None:
            raise ValueError("command workload requires external --objective")
        argv = _command_argv(workload_cmd)
        normalized_objective = load_objective(objective)
        cases = ()
        return WorkloadSpec(
            kind="command",
            source=argv,
            objective=normalized_objective,
            cases=cases,
            source_hash=_normalized_source_hash(
                "command", argv, normalized_objective, cases
            ),
        )

    return _normalize_manifest(workload_manifest, objective)


def _validation_failed(result) -> bool:
    if result is False:
        return True
    if isinstance(result, Mapping) and result.get("valid") is False:
        return True
    return False


def _json_value_copy(value, field: str):
    """Return a detached JSON value, rejecting ambiguous Python-only data."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} numbers must be finite")
        return value
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} object must use string keys")
            normalized[key] = _json_value_copy(item, f"{field}.{key}")
        return normalized
    if isinstance(value, list):
        return [
            _json_value_copy(item, f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(
        f"{field} must contain only JSON-compatible values, got "
        f"{type(value).__name__}"
    )


def _validate_validation_result(validation) -> bool | dict:
    if isinstance(validation, bool):
        return validation
    elif isinstance(validation, Mapping):
        normalized_validation = _json_value_copy(validation, "validation")
        if not isinstance(normalized_validation.get("valid"), bool):
            raise ValueError("validation object requires literal boolean valid")
        return normalized_validation
    raise ValueError(
        "validation must be a literal boolean or object with literal boolean valid"
    )


def _validate_benchmark_result(benchmark) -> dict:
    if not isinstance(benchmark, Mapping):
        raise ValueError("benchmark must be a JSON object")
    return _json_value_copy(benchmark, "benchmark")


def _validate_observation(validation, benchmark) -> tuple[bool | dict, dict]:
    """Normalize the lifecycle evidence shared by Python and command paths."""
    return (
        _validate_validation_result(validation),
        _validate_benchmark_result(benchmark),
    )


def _record_cleanup_failure(primary: BaseException, cleanup: BaseException) -> None:
    note = f"workload cleanup failed: {type(cleanup).__name__}: {cleanup}"
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        add_note(note)
    else:
        notes = list(getattr(primary, "__notes__", []))
        notes.append(note)
        try:
            primary.__notes__ = notes
        except Exception:
            pass


def run_once(adapter, *, candidate, role: str, case: dict) -> dict:
    """Run one Python-adapter observation and always clean up exactly once."""
    _nonempty_string(role, "role")
    if not isinstance(case, dict):
        raise ValueError("case must be an object")
    lifecycle_candidate = copy.deepcopy(candidate)
    case_copy = copy.deepcopy(case)
    primary = None
    try:
        adapter.prepare(lifecycle_candidate)
        raw_validation = adapter.validate(lifecycle_candidate)
        validation = _validate_validation_result(raw_validation)
        if _validation_failed(validation):
            raise ValueError("workload validation failed")
        raw_benchmark = adapter.benchmark(lifecycle_candidate)
        benchmark = _validate_benchmark_result(raw_benchmark)
        objective = validate_objective(adapter.metrics())
        return {
            "role": role,
            "case": case_copy,
            "validation": validation,
            "benchmark": benchmark,
            "objective": objective,
        }
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            adapter.cleanup()
        except BaseException as cleanup_error:
            if primary is None:
                raise
            _record_cleanup_failure(primary, cleanup_error)


def _clip_diagnostic(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value)
    if len(text) <= _DIAGNOSTIC_LIMIT:
        return text
    return text[:_DIAGNOSTIC_LIMIT] + "...[truncated]"


def _diagnostics(stdout, stderr) -> str:
    return (
        f"; stdout={_clip_diagnostic(stdout)!r}"
        f"; stderr={_clip_diagnostic(stderr)!r}"
    )


def _read_command_output(path: Path) -> dict:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except FileNotFoundError as error:
        raise RuntimeError("workload command did not create its output file") from error
    except OSError as error:
        raise RuntimeError(f"cannot open workload command output: {error}") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError("workload command output must be a regular file")
        if info.st_size > _OUTPUT_LIMIT_BYTES:
            raise RuntimeError(
                f"workload command output exceeds {_OUTPUT_LIMIT_BYTES} bytes"
            )
        chunks = []
        remaining = _OUTPUT_LIMIT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if len(raw) > _OUTPUT_LIMIT_BYTES:
        raise RuntimeError(
            f"workload command output exceeds {_OUTPUT_LIMIT_BYTES} bytes"
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"workload command output must be valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("workload command output must be a single JSON object")
    required = {"validation", "benchmark"}
    missing = sorted(required - set(value))
    if missing:
        raise RuntimeError(
            "workload command output missing required fields: "
            + ", ".join(missing)
        )
    unknown = sorted(set(value) - {"validation", "benchmark", "diagnostics"})
    if unknown:
        shown = ", ".join(unknown[:8])
        suffix = " ..." if len(unknown) > 8 else ""
        raise RuntimeError(
            f"workload command output contains unknown fields: {shown}{suffix}"
        )
    try:
        validation, benchmark = _validate_observation(
            value["validation"], value["benchmark"]
        )
    except ValueError as error:
        raise RuntimeError(f"workload command {error}") from error
    value["validation"] = validation
    value["benchmark"] = benchmark
    if "diagnostics" in value and not isinstance(value["diagnostics"], dict):
        raise RuntimeError("workload command diagnostics must be a JSON object")
    return value


def run_command_once(
    spec: WorkloadSpec,
    *,
    candidate,
    role: str,
    case: dict,
    timeout: float | None = None,
) -> dict:
    """Execute one command observation using only the output-file protocol."""
    if not isinstance(spec, WorkloadSpec) or not isinstance(spec.source, list):
        raise ValueError("command workload spec must contain an argv source")
    normalized_role = _nonempty_string(role, "role")
    if not isinstance(case, dict):
        raise ValueError("case must be an object")
    case_json = _canonical_json(copy.deepcopy(case)).decode("utf-8")
    if timeout is not None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
        ):
            raise ValueError("timeout must be a positive finite number")
        timeout = float(timeout)

    with tempfile.TemporaryDirectory(prefix="cuda-optimizer-workload-") as tmp:
        output_path = Path(tmp) / "observation.json"
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_OPTIMIZER_CANDIDATE": str(candidate),
                "CUDA_OPTIMIZER_ROLE": normalized_role,
                "CUDA_OPTIMIZER_OUTPUT": str(output_path),
                "CUDA_OPTIMIZER_CASE": case_json,
            }
        )
        try:
            completed = subprocess.run(
                list(spec.source),
                shell=False,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                f"workload command not found: {spec.source[0]}"
            ) from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"workload command timed out{_diagnostics(error.stdout, error.stderr)}"
            ) from error
        except OSError as error:
            raise RuntimeError(f"failed to execute workload command: {error}") from error

        if completed.returncode != 0:
            raise RuntimeError(
                f"workload command exited with exit {completed.returncode}"
                f"{_diagnostics(completed.stdout, completed.stderr)}"
            )
        try:
            return _read_command_output(output_path)
        except RuntimeError as error:
            raise RuntimeError(
                f"{error}{_diagnostics(completed.stdout, completed.stderr)}"
            ) from error


def _verify_source_hash(spec: WorkloadSpec) -> None:
    try:
        current = _normalized_source_hash(
            spec.kind,
            spec.source,
            spec.objective,
            spec.cases,
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"workload source_hash verification failed: {error}") from error
    if current != spec.source_hash:
        raise ValueError(
            "workload source_hash mismatch; source or frozen contract changed "
            "after normalization"
        )


def run_spec_once(
    spec: WorkloadSpec,
    *,
    candidate,
    role: str,
    case: dict | None = None,
    timeout: float | None = None,
) -> dict:
    """Run any normalized workload through one lifecycle result contract."""
    if not isinstance(spec, WorkloadSpec):
        raise ValueError("spec must be a WorkloadSpec")
    normalized_role = _nonempty_string(role, "role")
    if case is None:
        normalized_case = {}
    elif isinstance(case, dict):
        normalized_case = copy.deepcopy(case)
    else:
        raise ValueError("case must be an object")

    frozen_objective = validate_objective(spec.objective)
    _verify_source_hash(spec)

    if spec.kind == "python":
        if not isinstance(spec.source, str):
            raise ValueError("Python workload source must be a file path")
        if timeout is not None:
            raise ValueError(
                "timeout is only runner-enforced for command workloads; "
                "Python adapters must raise TimeoutError themselves"
            )
        source_path = _regular_file(spec.source, "workload adapter")
        adapter = load_python_adapter(source_path)
        current_objective = _adapter_objective(adapter, source_path)
        if current_objective != frozen_objective:
            raise ValueError(
                "conflicting objective: Python adapter metrics() changed after "
                "normalization"
            )
        result = run_once(
            adapter,
            candidate=candidate,
            role=normalized_role,
            case=normalized_case,
        )
        if result["objective"] != frozen_objective:
            raise ValueError(
                "conflicting objective: Python adapter metrics() changed during run"
            )
        return result

    if spec.kind == "command":
        if not isinstance(spec.source, list):
            raise ValueError("command workload source must be an argv list")
        command_result = run_command_once(
            spec,
            candidate=candidate,
            role=normalized_role,
            case=normalized_case,
            timeout=timeout,
        )
        validation = command_result["validation"]
        if _validation_failed(validation):
            raise ValueError("workload validation failed")
        result = {
            "role": normalized_role,
            "case": normalized_case,
            "validation": validation,
            "benchmark": command_result["benchmark"],
            "objective": frozen_objective,
        }
        if "diagnostics" in command_result:
            result["diagnostics"] = command_result["diagnostics"]
        return result

    raise ValueError(f"unknown workload kind: {spec.kind}")
