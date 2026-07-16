#!/usr/bin/env python3
"""Collect durable, content-addressed compiler artifacts without fabrication."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path


SCHEMA_VERSION = 1
STAGES = ("source", "ttir", "ttgir", "llvm_ir", "ptx", "sass", "binary")

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


def _regular_file_digest(path) -> tuple[Path, int, str] | None:
    candidate = Path(path).expanduser()
    try:
        file_stat = candidate.lstat()
    except (FileNotFoundError, OSError):
        return None
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        return None

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(candidate), flags)
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
        resolved = candidate.resolve(strict=True)
        return resolved, opened_stat.st_size, digest.hexdigest()
    except OSError:
        return None
    finally:
        os.close(fd)


def _record(path) -> dict:
    if path is None:
        return dict(_UNAVAILABLE)
    artifact = _regular_file_digest(path)
    if artifact is None:
        return dict(_UNAVAILABLE)
    resolved, size_bytes, digest = artifact
    return {
        "status": "available",
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": size_bytes,
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


def collect(
    *,
    source=None,
    binary=None,
    discovered=None,
    compile_command=None,
    backend=None,
    arch=None,
) -> dict:
    """Build a complete stage manifest from files that exist right now."""
    discovered = {} if discovered is None else discovered
    if not isinstance(discovered, Mapping):
        raise ValueError("discovered compiler artifacts must be a mapping")
    unknown = sorted(set(discovered) - set(STAGES))
    if unknown:
        raise ValueError(f"unknown compiler stage: {unknown[0]}")
    command = _command(compile_command)
    if backend is not None and (type(backend) is not str or not backend):
        raise ValueError("backend must be a non-empty string")
    if arch is not None and (type(arch) is not str or not arch):
        raise ValueError("arch must be a non-empty string")

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
    }
    for stage in STAGES:
        result[stage] = _record(paths.get(stage))
    return result


def _load_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"compiler evidence manifest must be a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid compiler evidence manifest: {path}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported compiler evidence manifest: {path}")
    for stage in STAGES:
        if stage not in payload or not isinstance(payload[stage], dict):
            raise ValueError(f"compiler evidence manifest is missing stage: {stage}")
    return payload


def update_manifest(
    evidence_dir,
    *,
    source=None,
    binary=None,
    discovered=None,
    compile_command=None,
    backend=None,
    arch=None,
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
        result = collect(
            discovered=supplied,
            compile_command=compile_command,
            backend=backend,
            arch=arch,
        )
    else:
        unknown = sorted(set(supplied) - set(STAGES))
        if unknown:
            raise ValueError(f"unknown compiler stage: {unknown[0]}")
        result = dict(existing)
        for stage in STAGES:
            previous = existing[stage]
            if previous.get("status") == "available":
                result[stage] = _record(previous.get("path"))
            else:
                result[stage] = dict(_UNAVAILABLE)
        for stage, path in supplied.items():
            result[stage] = _record(path)
        if compile_command is not None:
            result["compile_command"] = _command(compile_command)
        if backend is not None:
            if type(backend) is not str or not backend:
                raise ValueError("backend must be a non-empty string")
            result["backend"] = backend
        if arch is not None:
            if type(arch) is not str or not arch:
                raise ValueError("arch must be a non-empty string")
            result["arch"] = arch

    _atomic_write_json(manifest_path, result)
    return result


def same_artifact(first, second) -> bool:
    """Return true only when two real files have identical size and SHA-256."""
    left = _regular_file_digest(first)
    right = _regular_file_digest(second)
    if left is None or right is None:
        return False
    return left[1:] == right[1:]


def _cache_files(cache_root) -> dict[str, tuple[int, int, Path]]:
    root = Path(cache_root).expanduser()
    if root.is_symlink() or not root.is_dir():
        return {}
    files = {}
    try:
        candidates = root.rglob("*")
        for candidate in candidates:
            try:
                details = candidate.lstat()
            except OSError:
                continue
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
                continue
            files[str(candidate.resolve())] = (
                details.st_mtime_ns,
                details.st_size,
                candidate.resolve(),
            )
    except OSError:
        return files
    return files


def snapshot_cache(cache_root) -> dict[str, tuple[int, int]]:
    """Freeze real-file identity data before a Triton kernel triggers compilation."""
    return {
        path: (details[0], details[1])
        for path, details in _cache_files(cache_root).items()
    }


def discover_triton_cache(cache_root, before=None) -> dict[str, Path]:
    """Return only recognized cache files created or changed since *before*."""
    previous = {} if before is None else dict(before)
    newest = {}
    for path, details in _cache_files(cache_root).items():
        mtime_ns, size_bytes, resolved = details
        if previous.get(path) == (mtime_ns, size_bytes):
            continue
        stage = _CACHE_SUFFIX_STAGE.get(resolved.suffix.lower())
        if stage is None:
            continue
        current = newest.get(stage)
        order_key = (mtime_ns, path)
        if current is None or order_key > current[0]:
            newest[stage] = (order_key, resolved)
    return {stage: value[1] for stage, value in newest.items()}
