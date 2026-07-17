#!/usr/bin/env python3
"""Workload-scoped, advisory strategy memory primitives."""

from __future__ import annotations

import copy
import ctypes
import errno
import fcntl
import hashlib
import json
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
import workload_adapter  # noqa: E402


MEMORY_SCHEMA = "cuda-kernel-optimizer/strategy-memory-v1"
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


def _input_identity(manifest: Mapping, name: str) -> tuple[str, int]:
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
    current = artifact_store.read_regular_bytes(path)
    actual = hashlib.sha256(current).hexdigest()
    if actual != expected:
        raise ValueError(f"manifest.inputs.{name}.sha256 does not match current content")
    if len(current) != record["size_bytes"]:
        raise ValueError(f"manifest.inputs.{name}.size_bytes does not match current content")
    return actual, len(current)


def scope_document(manifest_path: str | os.PathLike) -> dict:
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
    baseline_sha, _ = _input_identity(manifest, "baseline")
    ref_sha, _ = _input_identity(manifest, "ref")
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
    return _validate_scope_document(scope)


def scope_key(manifest_or_scope: str | os.PathLike | Mapping) -> str:
    """Return the canonical SHA-256 key for a manifest path or scope document."""
    if isinstance(manifest_or_scope, Mapping):
        scope = manifest_or_scope
    else:
        scope = scope_document(manifest_or_scope)
    return _scope_key_from_document(scope)


def _validate_record(value: Any) -> dict:
    record = _strict_copy(value, "strategy run")
    _exact_fields(record, _RECORD_FIELDS, "strategy run")
    for field in sorted(_RECORD_FIELDS):
        _sha(record[field], f"strategy run.{field}")
    return record


def _record_key(record: Mapping) -> str:
    identity = {field: record[field] for field in sorted(_RECORD_FIELDS)}
    return hashlib.sha256(_canonical_bytes(identity)).hexdigest()


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
        for record in entry["runs"]:
            clean_record = _validate_record(record)
            if clean_record["input_hash"] != clean_scope["input_hash"]:
                raise ValueError(
                    "strategy run.input_hash does not match its scope.input_hash"
                )
            identity = _record_key(clean_record)
            if identity in identities:
                raise ValueError("scope entry contains duplicate strategy runs")
            identities.add(identity)
        for field in ("methods", "bundles"):
            if not isinstance(entry[field], dict):
                raise ValueError(f"scope entry.{field} must be a JSON object")
            if entry[field]:
                raise ValueError(f"scope entry.{field} is unsupported before record import")
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
        inserted = True
        return memory

    _locked_memory_update(memory_path, update)
    return inserted
