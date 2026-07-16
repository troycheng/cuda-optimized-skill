#!/usr/bin/env python3
"""Collect durable, content-addressed compiler artifacts without fabrication."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path


SCHEMA_VERSION = 2
STAGES = ("source", "ttir", "ttgir", "llvm_ir", "ptx", "sass", "binary")
TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "compile_command", "backend", "arch", "binary_sha256", *STAGES}
)
STAGE_FIELDS = frozenset({"status", "path", "sha256", "size_bytes"})
BACKENDS = frozenset({"cuda", "cutlass", "triton"})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

_UNAVAILABLE = {
    "status": "unavailable",
    "path": None,
    "sha256": None,
    "size_bytes": None,
}
_CACHE_SUFFIX_STAGE = {
    ".ttir": "ttir",
    ".ttgir": "ttgir",
    ".llir": "llvm_ir",
    ".llvm": "llvm_ir",
    ".ptx": "ptx",
    ".cubin": "binary",
    ".hsaco": "binary",
}


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(str(path), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _artifact_directory(path) -> Path:
    directory = Path(path).expanduser()
    if directory.is_symlink():
        raise ValueError(f"compiler evidence directory must not be a symlink: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"compiler evidence path is not a real directory: {directory}")
    return directory


def _atomic_write(path: Path, data: bytes) -> None:
    directory = _artifact_directory(path.parent)
    if path.is_symlink():
        raise ValueError(f"compiler evidence output must not be a symlink: {path}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(directory)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
        _fsync_directory(directory)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path, text: str) -> Path:
    """Atomically persist UTF-8 compiler text evidence."""
    if type(text) is not str:
        raise ValueError("compiler evidence text must be a string")
    target = Path(path)
    _atomic_write(target, text.encode("utf-8"))
    return target


def _atomic_write_json(path: Path, payload: dict) -> None:
    data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    _atomic_write(path, data)


def artifact_identity(path) -> dict | None:
    """Return a stable identity only while *path* remains bound to the open file."""
    candidate = Path(path).expanduser().absolute()
    try:
        file_stat = candidate.lstat()
        resolved_before = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        return None
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        return None

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(resolved_before), flags)
    except OSError:
        return None
    digest = hashlib.sha256()
    try:
        opened_stat = os.fstat(fd)
        if not stat.S_ISREG(opened_stat.st_mode):
            return None
        with os.fdopen(fd, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        closed_stat = os.fstat(fd)
        if (
            opened_stat.st_dev != closed_stat.st_dev
            or opened_stat.st_ino != closed_stat.st_ino
            or opened_stat.st_size != closed_stat.st_size
            or opened_stat.st_mtime_ns != closed_stat.st_mtime_ns
        ):
            return None
        resolved_after = candidate.resolve(strict=True)
        path_stat = candidate.lstat()
        if (
            resolved_after != resolved_before
            or stat.S_ISLNK(path_stat.st_mode)
            or path_stat.st_dev != opened_stat.st_dev
            or path_stat.st_ino != opened_stat.st_ino
        ):
            return None
        return {
            "path": resolved_before,
            "dev": opened_stat.st_dev,
            "ino": opened_stat.st_ino,
            "size_bytes": opened_stat.st_size,
            "mtime_ns": opened_stat.st_mtime_ns,
            "sha256": digest.hexdigest(),
        }
    except (OSError, RuntimeError):
        return None
    finally:
        os.close(fd)


def _record(path) -> dict:
    if path is None:
        return dict(_UNAVAILABLE)
    identity = artifact_identity(path)
    if identity is None:
        return dict(_UNAVAILABLE)
    return {
        "status": "available",
        "path": str(identity["path"]),
        "sha256": identity["sha256"],
        "size_bytes": identity["size_bytes"],
    }


def _command(value) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError("compile_command must be a sequence of strings")
    command = list(value)
    if not command or any(type(item) is not str or not item for item in command):
        raise ValueError("compile_command must contain non-empty strings")
    return command


def _backend(value) -> str | None:
    if value is None:
        return None
    if type(value) is not str or value not in BACKENDS:
        raise ValueError("backend must be cuda, cutlass, triton, or null")
    return value


def _arch(value) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value.strip():
        raise ValueError("arch must be a non-empty string or null")
    return value


def _binary_sha256(value) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise ValueError("binary_sha256 must be a lowercase SHA-256 or null")
    return value


def _validate_stage(stage: str, record) -> None:
    if not isinstance(record, Mapping) or set(record) != STAGE_FIELDS:
        raise ValueError(f"compiler evidence stage {stage} has invalid fields")
    status = record.get("status")
    if status == "unavailable":
        if any(record[name] is not None for name in ("path", "sha256", "size_bytes")):
            raise ValueError(f"compiler evidence stage {stage} is incoherent")
        return
    if status != "available":
        raise ValueError(f"compiler evidence stage {stage} has invalid status")
    path = record.get("path")
    digest = record.get("sha256")
    size_bytes = record.get("size_bytes")
    if type(path) is not str or not path or not Path(path).is_absolute():
        raise ValueError(f"compiler evidence stage {stage} has invalid path")
    if type(digest) is not str or not _SHA256.fullmatch(digest):
        raise ValueError(f"compiler evidence stage {stage} has invalid sha256")
    if type(size_bytes) is not int or size_bytes < 0:
        raise ValueError(f"compiler evidence stage {stage} has invalid size_bytes")


def validate_manifest(payload) -> dict:
    """Validate the exact manifest schema without trusting stored file claims."""
    if not isinstance(payload, Mapping) or set(payload) != TOP_LEVEL_FIELDS:
        raise ValueError("compiler evidence manifest has invalid top-level fields")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"compiler evidence schema_version must be {SCHEMA_VERSION}")
    _command(payload.get("compile_command"))
    _backend(payload.get("backend"))
    _arch(payload.get("arch"))
    binding = _binary_sha256(payload.get("binary_sha256"))
    for stage in STAGES:
        _validate_stage(stage, payload[stage])
    if binding is not None:
        if payload["binary"]["status"] != "available":
            raise ValueError("binary_sha256 requires available binary evidence")
        if payload["sass"]["status"] != "available":
            raise ValueError("binary_sha256 requires available SASS evidence")
        if payload["binary"]["sha256"] != binding:
            raise ValueError("binary_sha256 does not match binary evidence")
    elif payload["sass"]["status"] == "available":
        raise ValueError("available SASS evidence requires binary_sha256")
    return dict(payload)


def collect(
    *,
    source=None,
    binary=None,
    discovered=None,
    compile_command=None,
    backend=None,
    arch=None,
    binary_sha256=None,
) -> dict:
    """Build a complete stage manifest from files that exist right now."""
    discovered = {} if discovered is None else discovered
    if not isinstance(discovered, Mapping):
        raise ValueError("discovered compiler artifacts must be a mapping")
    unknown = sorted(set(discovered) - set(STAGES))
    if unknown:
        raise ValueError(f"unknown compiler stage: {unknown[0]}")
    command = _command(compile_command)
    backend = _backend(backend)
    arch = _arch(arch)

    paths = dict(discovered)
    if source is not None:
        paths["source"] = source
    if binary is not None:
        paths["binary"] = binary

    result = {
        "schema_version": SCHEMA_VERSION,
        "compile_command": command,
        "backend": backend,
        "arch": arch,
        "binary_sha256": _binary_sha256(binary_sha256),
    }
    for stage in STAGES:
        result[stage] = _record(paths.get(stage))
    if result["sass"]["status"] == "available" and result["binary_sha256"] is None:
        if result["binary"]["status"] != "available":
            raise ValueError("SASS evidence requires an available binary")
        result["binary_sha256"] = result["binary"]["sha256"]
    return validate_manifest(result)


def _load_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"compiler evidence manifest must be a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid compiler evidence manifest: {path}") from error
    try:
        return validate_manifest(payload)
    except ValueError as error:
        raise ValueError(f"invalid compiler evidence manifest {path}: {error}") from error


def load_manifest(evidence_dir) -> dict:
    path = Path(evidence_dir) / "manifest.json"
    payload = _load_manifest(path)
    if payload is None:
        raise ValueError(f"compiler evidence manifest not found: {path}")
    return payload


def write_fresh_manifest(
    evidence_dir,
    *,
    source=None,
    binary=None,
    discovered=None,
    compile_command=None,
    backend=None,
    arch=None,
    binary_sha256=None,
) -> dict:
    """Atomically replace prior evidence without reading stale or corrupt state."""
    directory = _artifact_directory(evidence_dir)
    result = collect(
        source=source,
        binary=binary,
        discovered=discovered,
        compile_command=compile_command,
        backend=backend,
        arch=arch,
        binary_sha256=binary_sha256,
    )
    _atomic_write_json(directory / "manifest.json", result)
    return result


def _revalidate_record(record: Mapping) -> dict:
    if record.get("status") != "available":
        return dict(_UNAVAILABLE)
    current = _record(record.get("path"))
    if current != dict(record):
        return dict(_UNAVAILABLE)
    return current


def update_manifest(
    evidence_dir,
    *,
    source=None,
    binary=None,
    discovered=None,
    compile_command=None,
    backend=None,
    arch=None,
    binary_sha256=None,
) -> dict:
    """Atomically create or merge compiler evidence for a resumable run."""
    directory = _artifact_directory(evidence_dir)
    manifest_path = directory / "manifest.json"
    existing = _load_manifest(manifest_path)
    supplied = {} if discovered is None else dict(discovered)
    if source is not None:
        supplied["source"] = source
    if binary is not None:
        supplied["binary"] = binary

    if existing is None:
        return write_fresh_manifest(
            directory,
            discovered=supplied,
            compile_command=compile_command,
            backend=backend,
            arch=arch,
            binary_sha256=binary_sha256,
        )
    else:
        unknown = sorted(set(supplied) - set(STAGES))
        if unknown:
            raise ValueError(f"unknown compiler stage: {unknown[0]}")
        result = dict(existing)
        for stage in STAGES:
            result[stage] = _revalidate_record(existing[stage])
        for stage, path in supplied.items():
            result[stage] = _record(path)
        if compile_command is not None:
            result["compile_command"] = _command(compile_command)
        if backend is not None:
            result["backend"] = _backend(backend)
        if arch is not None:
            result["arch"] = _arch(arch)

        if result["sass"]["status"] == "available" and result["binary"]["status"] == "available":
            requested_binding = _binary_sha256(binary_sha256)
            if "sass" in supplied:
                result["binary_sha256"] = requested_binding or result["binary"]["sha256"]
            elif existing.get("binary_sha256") == result["binary"]["sha256"]:
                result["binary_sha256"] = existing["binary_sha256"]
            else:
                result["sass"] = dict(_UNAVAILABLE)
                result["binary_sha256"] = None
        else:
            result["sass"] = dict(_UNAVAILABLE)
            result["binary_sha256"] = None

    validate_manifest(result)
    _atomic_write_json(manifest_path, result)
    return result


def same_artifact(first, second) -> bool:
    """Return true only when two real files have identical size and SHA-256."""
    left = artifact_identity(first)
    right = artifact_identity(second)
    if left is None or right is None:
        return False
    return (
        left["size_bytes"], left["sha256"]
    ) == (
        right["size_bytes"], right["sha256"]
    )


def _cache_files(cache_root) -> dict[str, dict]:
    root = Path(cache_root).expanduser()
    if root.is_symlink() or not root.is_dir():
        return {}
    files = {}
    try:
        candidates = root.rglob("*")
        for candidate in candidates:
            identity = artifact_identity(candidate)
            if identity is None:
                continue
            files[str(identity["path"])] = identity
    except OSError:
        return files
    return files


def _snapshot_identity(identity: Mapping) -> tuple:
    return (
        identity["dev"],
        identity["ino"],
        identity["size_bytes"],
        identity["mtime_ns"],
        identity["sha256"],
    )


def snapshot_cache(cache_root) -> dict[str, tuple]:
    """Freeze real-file identity data before a Triton kernel triggers compilation."""
    return {
        path: _snapshot_identity(identity)
        for path, identity in _cache_files(cache_root).items()
    }


def discover_triton_cache(cache_root, before=None) -> dict[str, Path]:
    """Return only recognized cache files created or changed since *before*."""
    previous = {} if before is None else dict(before)
    groups = {}
    for path, identity in _cache_files(cache_root).items():
        if previous.get(path) == _snapshot_identity(identity):
            continue
        resolved = identity["path"]
        stage = _CACHE_SUFFIX_STAGE.get(resolved.suffix.lower())
        if stage is None:
            continue
        unit = str(resolved.parent)
        stage_files = groups.setdefault(unit, {})
        current = stage_files.get(stage)
        order_key = (identity["mtime_ns"], path)
        if current is None or order_key > current[0]:
            stage_files[stage] = (order_key, resolved)
    if not groups:
        return {}
    selected_unit, selected = max(
        groups.items(),
        key=lambda item: (
            len(item[1]),
            max(value[0] for value in item[1].values()),
            item[0],
        ),
    )
    del selected_unit
    return {stage: value[1] for stage, value in selected.items()}
