#!/usr/bin/env python3
"""Normalize and execute user-owned end-to-end workloads.

This module deliberately has no workload discovery or download behavior.  A
full-mode run exists only when the user supplies a complete Python adapter,
command, or manifest; otherwise callers remain in kernel-only mode.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
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
_PROCESS_GRACE_SECONDS = 0.5
_SECRET_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "KEY",
    "COOKIE",
    "CREDENTIAL",
    "AUTH",
)


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    data: bytes
    device: int
    inode: int
    size: int
    mtime_ns: int
    mode: int
    sha256: str


@dataclass(frozen=True)
class _PythonBundle:
    source: _FileSnapshot
    dependencies: tuple[tuple[str, _FileSnapshot], ...]


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


def _strict_json_loads(text: str, field: str):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{field} contains duplicate key: {key}")
            result[key] = value
        return result

    def non_finite(token):
        raise ValueError(f"{field} contains non-finite JSON constant: {token}")

    try:
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=non_finite,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} must contain valid strict JSON: {error}") from error


def _read_json_object(path, field: str) -> tuple[Path, dict]:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError) as error:
        raise ValueError(f"{field} file does not exist: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"{field} must be a regular file: {resolved}")
    try:
        value = _strict_json_loads(resolved.read_text(encoding="utf-8"), field)
    except (OSError, UnicodeError) as error:
        raise ValueError(f"{field} must contain valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return resolved, value


def load_objective(path) -> dict:
    _, value = _read_json_object(path, "objective")
    return validate_objective(value)


def _stable_file_snapshot(path, field: str) -> _FileSnapshot:
    """Read one non-symlink regular file through a stable descriptor."""
    try:
        absolute = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
        before = os.lstat(absolute)
    except (OSError, TypeError) as error:
        raise ValueError(f"{field} file does not exist: {path}") from error
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{field} must not be a symlink: {absolute}")
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{field} must be a regular file: {absolute}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise ValueError(f"cannot open {field}: {absolute}: {error}") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{field} must be a regular file: {absolute}")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"{field} changed while opening: {absolute}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = os.lstat(absolute)
    except OSError as error:
        raise ValueError(f"{field} changed while reading: {absolute}") from error
    identity_before = (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        getattr(opened, "st_mtime_ns", int(opened.st_mtime * 1e9)),
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        getattr(after, "st_mtime_ns", int(after.st_mtime * 1e9)),
    )
    path_identity = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        getattr(path_after, "st_mtime_ns", int(path_after.st_mtime * 1e9)),
    )
    data = b"".join(chunks)
    if identity_before != identity_after or identity_after != path_identity:
        raise ValueError(f"{field} changed while reading: {absolute}")
    if len(data) != after.st_size:
        raise ValueError(f"{field} size changed while reading: {absolute}")
    canonical = Path(os.path.realpath(absolute.parent)) / absolute.name
    return _FileSnapshot(
        path=canonical,
        data=data,
        device=after.st_dev,
        inode=after.st_ino,
        size=after.st_size,
        mtime_ns=identity_after[3],
        mode=stat.S_IMODE(after.st_mode),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _declared_dependencies(source: _FileSnapshot) -> tuple[str, ...]:
    try:
        tree = compile(
            source.data,
            str(source.path),
            "exec",
            flags=ast.PyCF_ONLY_AST,
            dont_inherit=True,
        )
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"invalid workload adapter source: {error}") from error
    declarations = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "WORKLOAD_DEPENDENCIES"
            for target in node.targets
        ):
            declarations.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "WORKLOAD_DEPENDENCIES"
        ):
            declarations.append(node.value)
    if not declarations:
        return ()
    if len(declarations) != 1 or declarations[0] is None:
        raise ValueError("WORKLOAD_DEPENDENCIES must have one literal declaration")
    try:
        value = ast.literal_eval(declarations[0])
    except (ValueError, SyntaxError) as error:
        raise ValueError(
            "WORKLOAD_DEPENDENCIES must be a literal list or tuple"
        ) from error
    if not isinstance(value, (list, tuple)):
        raise ValueError("WORKLOAD_DEPENDENCIES must be a literal list or tuple")
    dependencies = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"WORKLOAD_DEPENDENCIES[{index}] must be a relative file string"
            )
        dependencies.append(item)
    if len(set(dependencies)) != len(dependencies):
        raise ValueError("WORKLOAD_DEPENDENCIES contains duplicate paths")
    return tuple(dependencies)


def _read_python_bundle(path) -> _PythonBundle:
    source = _stable_file_snapshot(path, "workload adapter")
    root = source.path.parent.resolve(strict=True)
    dependencies = []
    for relative in _declared_dependencies(source):
        dependency_path = Path(relative)
        if dependency_path.is_absolute():
            raise ValueError("workload dependency must be relative")
        if ".." in dependency_path.parts:
            raise ValueError(
                f"workload dependency escapes adapter directory: {relative}"
            )
        candidate = root
        try:
            for component in dependency_path.parts:
                if component in ("", "."):
                    continue
                candidate = candidate / component
                info = os.lstat(candidate)
                if stat.S_ISLNK(info.st_mode):
                    raise ValueError(
                        f"workload dependency must not contain symlinks: {relative}"
                    )
            resolved = candidate.resolve(strict=True)
        except ValueError:
            raise
        except (OSError, RuntimeError) as error:
            raise ValueError(
                f"workload dependency file does not exist: {relative}"
            ) from error
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"workload dependency escapes adapter directory: {relative}"
            ) from error
        snapshot = _stable_file_snapshot(resolved, "workload dependency")
        dependencies.append((relative, snapshot))
    return _PythonBundle(source=source, dependencies=tuple(dependencies))


def _dependency_module_name(relative: str) -> str | None:
    path = Path(relative)
    if path.suffix != ".py":
        return None
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    if not parts or not all(part.isidentifier() for part in parts):
        return None
    return ".".join(parts)


@contextmanager
def _installed_modules(modules: Mapping[str, ModuleType]):
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, old in previous.items():
            if old is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _fresh_module(snapshot: _FileSnapshot, name: str) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = str(snapshot.path)
    module.__package__ = name.rpartition(".")[0]
    code = compile(snapshot.data, str(snapshot.path), "exec", dont_inherit=True)
    exec(code, module.__dict__)
    return module


def _load_python_bundle(bundle: _PythonBundle) -> ModuleType:
    dependency_modules = {}
    dependency_snapshots = {}
    for relative, snapshot in bundle.dependencies:
        name = _dependency_module_name(relative)
        if name is None:
            continue
        if name in dependency_modules:
            raise ValueError(f"declared dependency module name collision: {name}")
        dependency_snapshots[name] = snapshot
        dependency_modules[name] = ModuleType(name)
        dependency_modules[name].__file__ = str(snapshot.path)
        dependency_modules[name].__package__ = name.rpartition(".")[0]
    try:
        with _installed_modules(dependency_modules):
            for name, module in dependency_modules.items():
                code = compile(
                    dependency_snapshots[name].data,
                    str(dependency_snapshots[name].path),
                    "exec",
                    dont_inherit=True,
                )
                exec(code, module.__dict__)
            module_name = "_cuda_optimizer_user_workload_" + bundle.source.sha256
            module = _fresh_module(bundle.source, module_name)
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        raise ValueError(
            f"failed to import workload adapter {bundle.source.path}: "
            f"{type(error).__name__}: {error}"
        ) from error
    runtime_dependencies = getattr(module, "WORKLOAD_DEPENDENCIES", ())
    declared = tuple(relative for relative, _ in bundle.dependencies)
    if tuple(runtime_dependencies) != declared:
        raise ValueError("WORKLOAD_DEPENDENCIES must remain the declared literal value")
    module.__cuda_optimizer_bundle__ = bundle
    module.__cuda_optimizer_dependency_modules__ = dependency_modules
    missing_calls = [
        name
        for name in REQUIRED_ADAPTER_CALLS
        if not callable(getattr(module, name, None))
    ]
    if missing_calls:
        raise ValueError(
            "workload adapter missing required callables: "
            + ", ".join(missing_calls)
        )
    return module


def load_python_adapter(path) -> ModuleType:
    """Load exactly the bytes read from a stable user-owned source bundle."""
    return _load_python_bundle(_read_python_bundle(path))


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


def _normalize_command_source(
    command, *, base_dir: Path | None = None
) -> tuple[list[str], tuple[_FileSnapshot, ...]]:
    argv = _command_argv(command)
    executable_name = argv[0]
    executable_path = Path(executable_name).expanduser()
    if base_dir is not None and not executable_path.is_absolute() and (
        executable_path.parent != Path(".") or executable_name.startswith(".")
    ):
        executable_name = str(base_dir / executable_path)
    search = shutil.which(executable_name)
    if search is None:
        raise ValueError(f"workload command executable not found: {argv[0]}")
    executable = _stable_file_snapshot(search, "workload command executable")
    if not executable.mode & 0o111:
        raise ValueError(
            f"workload command executable is not executable: {executable.path}"
        )
    normalized = list(argv)
    normalized[0] = str(executable.path)
    snapshots = [executable]
    root = Path.cwd() if base_dir is None else base_dir
    for index, argument in enumerate(normalized[1:], start=1):
        if argument.startswith("-"):
            continue
        candidate = Path(argument).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            info = os.lstat(candidate)
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            snapshot = _stable_file_snapshot(candidate, "workload command script")
            normalized[index] = str(snapshot.path)
            snapshots.append(snapshot)
    return normalized, tuple(snapshots)


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


def _file_fingerprints(snapshots: Sequence[_FileSnapshot]) -> list[dict]:
    fingerprints = []
    seen = set()
    for snapshot in snapshots:
        if snapshot.path in seen:
            continue
        seen.add(snapshot.path)
        fingerprints.append(
            {
                "path": str(snapshot.path),
                "sha256": snapshot.sha256,
                "mode": snapshot.mode,
            }
        )
    return sorted(fingerprints, key=lambda item: item["path"])


def _source_hash(payload: dict, snapshots: Sequence[_FileSnapshot]) -> str:
    frozen = copy.deepcopy(payload)
    frozen["source_files"] = _file_fingerprints(snapshots)
    return hashlib.sha256(_canonical_json(frozen)).hexdigest()


def _normalized_source_hash(
    kind: str,
    source: str | Sequence[str],
    objective: Mapping,
    cases: Sequence[Mapping],
    *,
    snapshots: Sequence[_FileSnapshot] | None = None,
) -> str:
    """Hash the executable contract and the current referenced source bytes."""
    if kind not in ("python", "command"):
        raise ValueError(f"unknown workload kind: {kind}")
    if snapshots is None:
        if kind == "python":
            if not isinstance(source, str):
                raise ValueError("Python workload source must be a file path")
            bundle = _read_python_bundle(source)
            snapshots = (bundle.source,) + tuple(
                snapshot for _, snapshot in bundle.dependencies
            )
        else:
            if isinstance(source, str):
                raise ValueError("command workload source must be an argv list")
            normalized, snapshots = _normalize_command_source(source)
            if list(normalized) != list(source):
                raise ValueError("command workload source normalization changed")
    payload = {
        "executor_kind": kind,
        "source": source,
        "objective": objective,
        "cases": cases,
    }
    return _source_hash(payload, snapshots)


def _adapter_objective(adapter: ModuleType, path: Path) -> dict:
    try:
        with _adapter_runtime_scope(adapter):
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
        bundle = _read_python_bundle(source_path)
        source_path = bundle.source.path
        adapter = _load_python_bundle(bundle)
        adapter_objective = _adapter_objective(adapter, source_path)
        if adapter_objective != objective:
            raise ValueError(
                "conflicting objective: manifest objective does not match "
                "Python adapter metrics()"
            )
        objective = adapter_objective
        source: str | list[str] = str(source_path)
        snapshots = (bundle.source,) + tuple(
            snapshot for _, snapshot in bundle.dependencies
        )
    else:
        source, snapshots = _normalize_command_source(
            manifest["source"], base_dir=base_dir
        )

    return WorkloadSpec(
        kind=kind,
        source=source,
        objective=objective,
        cases=cases,
        source_hash=_normalized_source_hash(
            kind, source, objective, cases, snapshots=snapshots
        ),
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
        bundle = _read_python_bundle(workload)
        source_path = bundle.source.path
        adapter = _load_python_bundle(bundle)
        normalized_objective = _adapter_objective(adapter, source_path)
        cases: tuple[dict, ...] = ()
        return WorkloadSpec(
            kind="python",
            source=str(source_path),
            objective=normalized_objective,
            cases=cases,
            source_hash=_normalized_source_hash(
                "python",
                str(source_path),
                normalized_objective,
                cases,
                snapshots=(bundle.source,)
                + tuple(snapshot for _, snapshot in bundle.dependencies),
            ),
        )

    if workload_cmd is not None:
        if objective is None:
            raise ValueError("command workload requires external --objective")
        argv, snapshots = _normalize_command_source(workload_cmd)
        normalized_objective = load_objective(objective)
        cases = ()
        return WorkloadSpec(
            kind="command",
            source=argv,
            objective=normalized_objective,
            cases=cases,
            source_hash=_normalized_source_hash(
                "command",
                argv,
                normalized_objective,
                cases,
                snapshots=snapshots,
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


def _normalize_case(case) -> dict:
    if case is None:
        return {}
    if not isinstance(case, Mapping):
        raise ValueError("case must be a JSON object")
    return _json_value_copy(case, "case")


@contextmanager
def _adapter_runtime_scope(adapter, *, role: str | None = None, case=None):
    modules = getattr(adapter, "__cuda_optimizer_dependency_modules__", {})
    had_context = hasattr(adapter, "CUDA_OPTIMIZER_CONTEXT")
    previous_context = getattr(adapter, "CUDA_OPTIMIZER_CONTEXT", None)
    if role is not None:
        adapter.CUDA_OPTIMIZER_CONTEXT = {
            "role": role,
            "case": _normalize_case(case),
        }
    try:
        with _installed_modules(modules):
            yield
    finally:
        if role is not None:
            if had_context:
                adapter.CUDA_OPTIMIZER_CONTEXT = previous_context
            else:
                try:
                    delattr(adapter, "CUDA_OPTIMIZER_CONTEXT")
                except AttributeError:
                    pass


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
    normalized_role = _nonempty_string(role, "role")
    normalized_case = _normalize_case(case)
    lifecycle_candidate = copy.deepcopy(candidate)
    primary = None
    with _adapter_runtime_scope(
        adapter, role=normalized_role, case=normalized_case
    ):
        try:
            adapter.prepare(lifecycle_candidate)
            raw_validation = adapter.validate(lifecycle_candidate)
            validation = _validate_validation_result(raw_validation)
            if _validation_failed(validation):
                raise ValueError("workload validation failed")
            raw_benchmark = adapter.benchmark(lifecycle_candidate)
            benchmark = _validate_benchmark_result(raw_benchmark)
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
    with _adapter_runtime_scope(adapter):
        objective = validate_objective(adapter.metrics())
    return {
        "role": normalized_role,
        "case": copy.deepcopy(normalized_case),
        "validation": validation,
        "benchmark": benchmark,
        "objective": objective,
    }


def _is_secret_name(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in _SECRET_MARKERS)


def _redact_diagnostic(text: str, secret_values: Sequence[str]) -> str:
    for value in sorted(
        {value for value in secret_values if value}, key=len, reverse=True
    ):
        text = text.replace(value, "[REDACTED]")
    marker_pattern = "|".join(_SECRET_MARKERS)
    return re.sub(
        rf"(?i)([A-Z0-9_]*(?:{marker_pattern})[A-Z0-9_]*)\s*[:=]\s*([^\s,;]+)",
        lambda match: f"{match.group(1)}=[REDACTED]",
        text,
    )


def _clip_diagnostic(value, *, secret_values: Sequence[str] = ()) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = _redact_diagnostic(str(value), secret_values)
    if len(text) <= _DIAGNOSTIC_LIMIT:
        return text
    return text[:_DIAGNOSTIC_LIMIT] + "...[truncated]"


def _diagnostics(stdout, stderr, *, secret_values: Sequence[str] = ()) -> str:
    return (
        f"; stdout={_clip_diagnostic(stdout, secret_values=secret_values)!r}"
        f"; stderr={_clip_diagnostic(stderr, secret_values=secret_values)!r}"
    )


class _BoundedCapture:
    def __init__(self, limit: int = _DIAGNOSTIC_LIMIT) -> None:
        self.limit = limit
        self.data = bytearray()
        self.truncated = False

    def append(self, chunk: bytes) -> None:
        self.data.extend(chunk)
        if len(self.data) > self.limit:
            del self.data[: len(self.data) - self.limit]
            self.truncated = True

    def text(self) -> str:
        value = bytes(self.data).decode("utf-8", errors="replace")
        return ("...[truncated]" if self.truncated else "") + value


def _drain_pipe(stream, capture: _BoundedCapture) -> None:
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            capture.append(chunk)
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _signal_process_group(process_group: int, signal_number: int) -> None:
    try:
        os.killpg(process_group, signal_number)
    except (ProcessLookupError, PermissionError):
        pass


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_process_group(process, process_group: int) -> None:
    _signal_process_group(process_group, signal.SIGTERM)
    deadline = time.monotonic() + _PROCESS_GRACE_SECONDS
    while time.monotonic() < deadline:
        process.poll()
        if not _process_group_exists(process_group):
            break
        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
    if _process_group_exists(process_group):
        _signal_process_group(process_group, signal.SIGKILL)
    try:
        process.wait(timeout=_PROCESS_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        _signal_process_group(process_group, signal.SIGKILL)


def _finish_readers(threads, streams) -> None:
    deadline = time.monotonic() + _PROCESS_GRACE_SECONDS
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    readers_alive = any(thread.is_alive() for thread in threads)
    for stream in streams:
        if stream is not None and not stream.closed:
            try:
                if readers_alive:
                    os.close(stream.fileno())
                else:
                    stream.close()
            except Exception:
                pass
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))


def _command_environment(overrides: Mapping[str, str]) -> tuple[dict, tuple[str, ...]]:
    inherited = dict(os.environ)
    allow = {
        name.strip()
        for name in inherited.get("CUDA_OPTIMIZER_PASS_ENV", "").split(",")
        if name.strip()
    }
    secret_values = tuple(
        value
        for name, value in inherited.items()
        if _is_secret_name(name) and value
    )
    environment = {
        name: value
        for name, value in inherited.items()
        if not _is_secret_name(name) or name in allow
    }
    environment.update(overrides)
    return environment, secret_values


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
        value = _strict_json_loads(
            raw.decode("utf-8"), "workload command output"
        )
    except (UnicodeError, ValueError) as error:
        raise RuntimeError(
            f"workload command output must be valid strict JSON: {error}"
        ) from error
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
    if "diagnostics" in value:
        if not isinstance(value["diagnostics"], Mapping):
            raise RuntimeError("workload command diagnostics must be a JSON object")
        try:
            value["diagnostics"] = _json_value_copy(
                value["diagnostics"], "diagnostics"
            )
        except ValueError as error:
            raise RuntimeError(f"workload command {error}") from error
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
    _verify_command_source(spec)
    normalized_role = _nonempty_string(role, "role")
    normalized_case = _normalize_case(case)
    case_json = _canonical_json(normalized_case).decode("utf-8")
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
        environment, secret_values = _command_environment(
            {
                "CUDA_OPTIMIZER_CANDIDATE": str(candidate),
                "CUDA_OPTIMIZER_ROLE": normalized_role,
                "CUDA_OPTIMIZER_OUTPUT": str(output_path),
                "CUDA_OPTIMIZER_CASE": case_json,
            }
        )
        try:
            process = subprocess.Popen(
                list(spec.source),
                shell=False,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                text=False,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                f"workload command not found: {spec.source[0]}"
            ) from error
        except OSError as error:
            raise RuntimeError(f"failed to execute workload command: {error}") from error
        process_group = process.pid
        stdout_capture = _BoundedCapture()
        stderr_capture = _BoundedCapture()
        streams = (process.stdout, process.stderr)
        reader_threads = (
            threading.Thread(
                target=_drain_pipe,
                args=(process.stdout, stdout_capture),
                daemon=True,
            ),
            threading.Thread(
                target=_drain_pipe,
                args=(process.stderr, stderr_capture),
                daemon=True,
            ),
        )
        started_threads = []
        timeout_error = None
        try:
            for thread in reader_threads:
                thread.start()
                started_threads.append(thread)
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                timeout_error = error
        finally:
            _stop_process_group(process, process_group)
            _finish_readers(started_threads, streams)
        if timeout_error is not None:
            raise RuntimeError(
                "workload command timed out"
                + _diagnostics(
                    stdout_capture.text(),
                    stderr_capture.text(),
                    secret_values=secret_values,
                )
            ) from timeout_error
        stdout = stdout_capture.text()
        stderr = stderr_capture.text()
        if returncode != 0:
            raise RuntimeError(
                f"workload command exited with exit {returncode}"
                f"{_diagnostics(stdout, stderr, secret_values=secret_values)}"
            )
        try:
            return _read_command_output(output_path)
        except RuntimeError as error:
            raise RuntimeError(
                f"{error}{_diagnostics(stdout, stderr, secret_values=secret_values)}"
            ) from error


def _verify_source_hash(
    spec: WorkloadSpec, snapshots: Sequence[_FileSnapshot]
) -> None:
    try:
        current = _normalized_source_hash(
            spec.kind,
            spec.source,
            spec.objective,
            spec.cases,
            snapshots=snapshots,
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"workload source_hash verification failed: {error}") from error
    if current != spec.source_hash:
        raise ValueError(
            "workload source_hash mismatch; source or frozen contract changed "
            "after normalization"
        )


def _verify_command_source(spec: WorkloadSpec) -> None:
    try:
        normalized, snapshots = _normalize_command_source(spec.source)
    except (OSError, ValueError) as error:
        raise ValueError(f"workload source_hash verification failed: {error}") from error
    if normalized != list(spec.source):
        raise ValueError(
            "workload source_hash mismatch; command resolution changed after "
            "normalization"
        )
    _verify_source_hash(spec, snapshots)


def verify_frozen_spec(spec: WorkloadSpec) -> None:
    """Verify frozen source bytes and contract without executing user code."""
    if not isinstance(spec, WorkloadSpec):
        raise ValueError("spec must be a WorkloadSpec")
    if spec.kind == "command":
        _verify_command_source(spec)
        return
    if spec.kind != "python" or not isinstance(spec.source, str):
        raise ValueError("frozen workload kind or source is invalid")
    try:
        bundle = _read_python_bundle(spec.source)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"workload source_hash verification failed: {error}"
        ) from error
    snapshots = (bundle.source,) + tuple(
        snapshot for _, snapshot in bundle.dependencies
    )
    _verify_source_hash(spec, snapshots)


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
    normalized_case = _normalize_case(case)

    frozen_objective = validate_objective(spec.objective)

    if spec.kind == "python":
        if not isinstance(spec.source, str):
            raise ValueError("Python workload source must be a file path")
        if timeout is not None:
            raise ValueError(
                "timeout is only runner-enforced for command workloads; "
                "Python adapters must raise TimeoutError themselves"
            )
        try:
            bundle = _read_python_bundle(spec.source)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"workload source_hash verification failed: {error}"
            ) from error
        snapshots = (bundle.source,) + tuple(
            snapshot for _, snapshot in bundle.dependencies
        )
        _verify_source_hash(spec, snapshots)
        source_path = bundle.source.path
        adapter = _load_python_bundle(bundle)
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
        return result

    raise ValueError(f"unknown workload kind: {spec.kind}")
