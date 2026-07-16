#!/usr/bin/env python3
"""Versioned, traversal-safe storage for optimizer run artifacts."""

from __future__ import annotations

import copy
import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Any, Optional, Union


CURRENT_SCHEMA_VERSION = 2
_CANDIDATE_ID = re.compile(r"[A-Za-z0-9._-]+")
_PathLike = Union[str, os.PathLike]


def sha256_file(path: _PathLike) -> str:
    """Return a stable SHA-256 digest without following path symlinks."""
    return hashlib.sha256(read_regular_bytes(path)).hexdigest()


def _open_parent_directory(
    path: _PathLike, *, create: bool
) -> tuple[int, str, Path]:
    """Open a stable parent dirfd without following any path component."""
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if not target.name:
        raise ValueError("artifact path must name a file")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(target.anchor, flags)
    try:
        for index, component in enumerate(target.parts[1:-1]):
            # macOS exposes root-owned compatibility aliases such as /var ->
            # /private/var.  Permit only that filesystem-root boundary; every
            # user-controlled descendant remains no-follow.
            component_flags = flags if index == 0 else flags | nofollow
            try:
                child_fd = os.open(
                    component, component_flags, dir_fd=directory_fd
                )
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, 0o755, dir_fd=directory_fd)
                child_fd = os.open(
                    component, component_flags, dir_fd=directory_fd
                )
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ValueError(
                        f"artifact parent path contains a symlink or non-directory: {target}"
                    ) from error
                raise
            os.close(directory_fd)
            directory_fd = child_fd
        return directory_fd, target.name, target
    except BaseException:
        os.close(directory_fd)
        raise


def _read_regular_leaf(
    directory_fd: int, leaf: str, target: Path, *, missing_ok: bool
) -> Optional[bytes]:
    """Read one no-follow regular leaf from an already stable parent dirfd."""
    fd = None
    try:
        try:
            fd = os.open(
                leaf,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
        except OSError as error:
            if missing_ok and error.errno == errno.ENOENT:
                return None
            if error.errno in {errno.ELOOP, errno.ENOENT, errno.ENOTDIR}:
                raise ValueError(
                    f"artifact file is missing, a symlink, or unsafe: {target}"
                ) from error
            raise
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"artifact path is not a regular file: {target}")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        if fd is not None:
            os.close(fd)


def read_regular_bytes(path: _PathLike) -> bytes:
    """Read regular-file bytes through no-follow parent and leaf descriptors."""
    try:
        directory_fd, leaf, target = _open_parent_directory(path, create=False)
    except FileNotFoundError as error:
        raise ValueError(f"artifact file does not exist: {path}") from error
    try:
        return _read_regular_leaf(
            directory_fd, leaf, target, missing_ok=False
        )
    finally:
        os.close(directory_fd)


def read_regular_with_optional_sibling(
    path: _PathLike, sibling_name: str
) -> tuple[bytes, Optional[bytes]]:
    """Read a regular file and optional sibling through one stable parent dirfd."""
    sibling = Path(sibling_name)
    if (
        not sibling_name
        or sibling.is_absolute()
        or len(sibling.parts) != 1
        or sibling_name in {".", ".."}
    ):
        raise ValueError("sibling_name must name one relative file")
    try:
        directory_fd, leaf, target = _open_parent_directory(path, create=False)
    except FileNotFoundError as error:
        raise ValueError(f"artifact file does not exist: {path}") from error

    try:
        primary = _read_regular_leaf(
            directory_fd, leaf, target, missing_ok=False
        )
        sibling_payload = _read_regular_leaf(
            directory_fd,
            sibling_name,
            target.with_name(sibling_name),
            missing_ok=True,
        )
        return primary, sibling_payload
    finally:
        os.close(directory_fd)


def _validate_leaf_name(value: str) -> str:
    if type(value) is not str or not value or value in {".", ".."}:
        raise ValueError("artifact bundle names must be non-empty file names")
    path = Path(value)
    windows_path = PureWindowsPath(value)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or len(path.parts) != 1
        or len(windows_path.parts) != 1
    ):
        raise ValueError("artifact bundle names must contain one relative component")
    return value


def _atomic_write_leaf(
    directory_fd: int, leaf: str, target: Path, payload: bytes
) -> None:
    if not isinstance(payload, bytes):
        raise TypeError("atomic payload must be bytes")
    temporary_leaf = f".{leaf}.{secrets.token_hex(8)}.tmp"
    fd = None
    try:
        try:
            existing = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and stat.S_ISLNK(existing.st_mode):
            raise ValueError(f"artifact target path contains a symlink: {target}")
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise ValueError(f"artifact target must be a regular file: {target}")
        fd = os.open(
            temporary_leaf,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError("atomic artifact write made no progress")
            offset += written
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(
            temporary_leaf,
            leaf,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
    except BaseException:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(temporary_leaf, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        raise


def _remove_regular_leaf(
    directory_fd: int, leaf: str, target: Path, *, missing_ok: bool
) -> bool:
    try:
        metadata = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError as error:
        if missing_ok:
            return False
        raise ValueError(f"artifact file does not exist: {target}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"artifact path is not a regular file: {target}")
    os.unlink(leaf, dir_fd=directory_fd)
    return True


def remove_regular_file(path: _PathLike, *, missing_ok: bool = True) -> bool:
    """Remove one regular file relative to a stable, no-follow parent dirfd."""
    try:
        directory_fd, leaf, target = _open_parent_directory(path, create=False)
    except FileNotFoundError as error:
        if missing_ok:
            return False
        raise ValueError(f"artifact file does not exist: {path}") from error
    try:
        removed = _remove_regular_leaf(
            directory_fd, leaf, target, missing_ok=missing_ok
        )
        os.fsync(directory_fd)
        return removed
    finally:
        os.close(directory_fd)


def atomic_write_bytes(path: _PathLike, payload: bytes) -> None:
    """Atomically replace a file relative to one stable, no-follow parent dirfd."""
    if not isinstance(payload, bytes):
        raise TypeError("atomic payload must be bytes")
    directory_fd, leaf, target = _open_parent_directory(path, create=True)
    try:
        _atomic_write_leaf(directory_fd, leaf, target, payload)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def publish_regular_bundle(directory: _PathLike, writes, removals=()) -> dict:
    """Publish regular-file writes/removals through one stable directory fd."""
    if not isinstance(writes, Mapping):
        raise ValueError("artifact bundle writes must be a mapping")
    if isinstance(removals, (str, bytes, bytearray, Mapping)):
        raise ValueError("artifact bundle removals must be a sequence")
    try:
        removal_names = list(removals)
    except TypeError as error:
        raise ValueError("artifact bundle removals must be a sequence") from error
    clean_writes = []
    expected_hashes = {}
    for name, payload in writes.items():
        leaf = _validate_leaf_name(name)
        if not isinstance(payload, bytes):
            raise TypeError("artifact bundle payloads must be bytes")
        clean_writes.append((leaf, payload))
        expected_hashes[leaf] = hashlib.sha256(payload).hexdigest()
    clean_removals = [_validate_leaf_name(name) for name in removal_names]
    if len(set(clean_removals)) != len(clean_removals):
        raise ValueError("artifact bundle removals must be unique")
    if set(expected_hashes).intersection(clean_removals):
        raise ValueError("artifact bundle cannot write and remove the same file")

    directory_path = Path(
        os.path.abspath(os.path.expanduser(os.fspath(directory)))
    )
    marker = directory_path / ".publish-bundle"
    try:
        directory_fd, _leaf, _target = _open_parent_directory(
            marker, create=False
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            f"artifact publish directory is missing, a symlink, or unsafe: {directory_path}"
        ) from error
    original = os.fstat(directory_fd)
    if not stat.S_ISDIR(original.st_mode):
        os.close(directory_fd)
        raise ValueError(f"artifact publish path is not a directory: {directory_path}")
    try:
        for leaf, payload in clean_writes:
            target = directory_path / leaf
            _atomic_write_leaf(directory_fd, leaf, target, payload)
            published = _read_regular_leaf(
                directory_fd, leaf, target, missing_ok=False
            )
            if hashlib.sha256(published).hexdigest() != expected_hashes[leaf]:
                raise ValueError(
                    f"published artifact does not match captured payload: {target}"
                )
        for leaf in clean_removals:
            _remove_regular_leaf(
                directory_fd,
                leaf,
                directory_path / leaf,
                missing_ok=True,
            )
        os.fsync(directory_fd)

        identity_fd = None
        try:
            identity_fd, _leaf, _target = _open_parent_directory(
                marker, create=False
            )
            current = os.fstat(identity_fd)
        except (OSError, ValueError) as error:
            raise ValueError(
                "artifact publish directory was replaced, is a symlink, or unsafe: "
                f"{directory_path}"
            ) from error
        finally:
            if identity_fd is not None:
                os.close(identity_fd)
        if (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino):
            raise ValueError(
                f"artifact publish directory was replaced: {directory_path}"
            )
        return expected_hashes
    finally:
        os.close(directory_fd)


def atomic_write_json(path: _PathLike, payload: Any) -> None:
    """Atomically replace *path* with a formatted UTF-8 JSON document."""
    try:
        encoded = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    except (ValueError, OverflowError) as error:
        raise ValueError("JSON document is not serializable") from error
    atomic_write_bytes(path, encoded)


def atomic_write_jsonl(path: _PathLike, records) -> None:
    """Atomically replace *path* with strict JSON Lines and durable metadata."""
    if isinstance(records, (str, bytes, bytearray, dict)):
        raise ValueError("JSONL records must be a sequence")
    try:
        snapshot = list(records)
    except TypeError as error:
        raise ValueError("JSONL records must be a sequence") from error
    encoded = []
    for index, record in enumerate(snapshot):
        try:
            encoded.append(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"JSONL record {index} is not strict JSON") from error

    document = ("\n".join(encoded) + ("\n" if encoded else "")).encode("utf-8")
    atomic_write_bytes(path, document)


def write_paired_samples(
    path: _PathLike,
    pairs,
    *,
    kind: str,
    input_hash: str,
    iteration: int,
    candidate_id,
    candidate_file: _PathLike,
    classifier_config=None,
) -> dict:
    """Persist raw paired observations with candidate/input/iteration bindings."""
    if kind not in {"kernel", "workload"}:
        raise ValueError("paired sample kind must be kernel or workload")
    if not isinstance(input_hash, str) or not input_hash.strip():
        raise ValueError("paired sample input_hash must be non-empty")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration <= 0:
        raise ValueError("paired sample iteration must be positive")
    if candidate_id is None or isinstance(candidate_id, bool):
        raise ValueError("paired sample candidate_id must be non-empty")
    candidate_name = str(candidate_id).strip()
    if not candidate_name:
        raise ValueError("paired sample candidate_id must be non-empty")
    if not isinstance(classifier_config, dict) or not classifier_config:
        raise ValueError("paired sample classifier_config must be a non-empty mapping")
    try:
        classifier = json.loads(
            json.dumps(classifier_config, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("paired sample classifier_config must be strict JSON") from error
    candidate = Path(candidate_file).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("paired sample candidate must be a regular non-symlink file")
    candidate = candidate.resolve(strict=True)
    candidate_sha256 = sha256_file(candidate)
    if isinstance(pairs, (str, bytes, bytearray, dict)):
        raise ValueError("paired samples must be a sequence")
    try:
        raw_pairs = copy.deepcopy(list(pairs))
    except TypeError as error:
        raise ValueError("paired samples must be a sequence") from error
    records = [
        {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "kind": kind,
            "input_hash": input_hash,
            "iteration": iteration,
            "candidate_id": candidate_name,
            "candidate_file": str(candidate),
            "candidate_sha256": candidate_sha256,
            "classifier": copy.deepcopy(classifier),
            "pair_index": index,
            "pair": pair,
        }
        for index, pair in enumerate(raw_pairs)
    ]
    target = Path(path).expanduser().absolute()
    atomic_write_jsonl(target, records)
    target = target.resolve(strict=True)
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "kind": kind,
        "path": str(target),
        "sha256": hashlib.sha256(read_regular_bytes(target)).hexdigest(),
        "pairs": len(records),
        "input_hash": input_hash,
        "iteration": iteration,
        "candidate_id": candidate_name,
        "candidate_file": str(candidate),
        "candidate_sha256": candidate_sha256,
        "classifier": classifier,
    }


class ArtifactStore:
    """Own the durable artifacts beneath one optimizer run directory."""

    def __init__(self, root: _PathLike) -> None:
        self.root = Path(root).expanduser().resolve()

    def initialize(
        self,
        *,
        inputs: dict,
        budget: dict,
        environment: Optional[dict] = None,
    ) -> dict:
        if not isinstance(inputs, dict):
            raise ValueError("inputs must be a dict containing baseline and ref")
        missing = [name for name in ("baseline", "ref") if name not in inputs]
        if missing:
            raise ValueError(
                "inputs must contain baseline and ref; missing: " + ", ".join(missing)
            )

        for directory in (
            self.root,
            self.root / "workload",
            self.root / "baseline",
            self.root / "candidates",
        ):
            directory.mkdir(parents=True, exist_ok=True)

        input_records = {}
        for name, value in inputs.items():
            if not isinstance(name, str) or not name:
                raise ValueError("input keys must be non-empty strings")
            source = Path(value).expanduser().resolve()
            digest = sha256_file(source)
            input_records[name] = {
                "path": str(source),
                "sha256": digest,
                "size_bytes": source.stat().st_size,
            }

        sha_mapping = {
            name: input_records[name]["sha256"] for name in sorted(input_records)
        }
        stable_json = json.dumps(
            sha_mapping, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        input_hash = hashlib.sha256(stable_json.encode("utf-8")).hexdigest()
        manifest = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "inputs": input_records,
            "budget": copy.deepcopy(budget),
            "environment": copy.deepcopy(environment) if environment is not None else {},
            "input_hash": input_hash,
        }
        atomic_write_json(self.root / "manifest.json", manifest)
        return manifest

    def candidate_dir(self, candidate_id: str) -> Path:
        if (
            not isinstance(candidate_id, str)
            or candidate_id in {".", ".."}
            or not _CANDIDATE_ID.fullmatch(candidate_id)
        ):
            raise ValueError(
                "candidate_id must match [A-Za-z0-9._-]+ and cannot be '.' or '..'"
            )
        path = self._resolve_relative(Path("candidates") / candidate_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_relative(self, relative_path: _PathLike) -> Path:
        text = os.fspath(relative_path)
        if not text:
            raise ValueError("artifact path must be a non-empty relative path")
        relative = Path(text)
        if relative.is_absolute() or PureWindowsPath(text).is_absolute():
            raise ValueError(f"artifact path must be relative to {self.root}: {text}")

        target = (self.root / relative).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as error:
            raise ValueError(
                f"artifact path escapes run root {self.root}: {text}"
            ) from error
        if target == self.root:
            raise ValueError(f"artifact path must name a file below {self.root}: {text}")
        return target

    def write_json(self, relative_path: _PathLike, payload: Any) -> Path:
        target = self._resolve_relative(relative_path)
        atomic_write_json(target, payload)
        return target

    def append_jsonl(self, relative_path: _PathLike, payload: Any) -> Path:
        target = self._resolve_relative(relative_path)
        line = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        directory_fd, leaf, stable_target = _open_parent_directory(
            target, create=True
        )
        fd = None
        try:
            try:
                fd = os.open(
                    leaf,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_APPEND
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o644,
                    dir_fd=directory_fd,
                )
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOENT, errno.ENOTDIR}:
                    raise ValueError(
                        f"JSONL target is missing, a symlink, or unsafe: {stable_target}"
                    ) from error
                raise
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError(
                    f"JSONL target is not a regular file: {stable_target}"
                )
            fcntl.flock(fd, fcntl.LOCK_EX)
            offset = 0
            while offset < len(line):
                offset += os.write(fd, line[offset:])
            os.fsync(fd)
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.fsync(directory_fd)
            return stable_target
        finally:
            if fd is not None:
                os.close(fd)
            os.close(directory_fd)

    def read_jsonl(self, relative_path: _PathLike) -> list:
        target = self._resolve_relative(relative_path)
        if not target.exists():
            return []
        records = []
        try:
            text = read_regular_bytes(target).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"JSONL artifact is not UTF-8: {target}") from error
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {target} at line {line_number}: {error.msg}"
                ) from error
        return records

    def write_checkpoint(self, payload: dict) -> Path:
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload must be a dict")
        checkpoint = copy.deepcopy(payload)
        checkpoint["schema_version"] = CURRENT_SCHEMA_VERSION
        path = self._resolve_relative("checkpoint.json")
        atomic_write_json(path, checkpoint)
        return path

    def load_checkpoint(self, *, expected_input_hash: str) -> dict:
        path = self._resolve_relative("checkpoint.json")
        try:
            checkpoint = json.loads(read_regular_bytes(path).decode("utf-8"))
        except ValueError as error:
            if "artifact file does not exist" in str(error):
                raise ValueError(f"checkpoint not found: {path}") from error
            raise
        except UnicodeDecodeError as error:
            raise ValueError(f"checkpoint is not UTF-8: {path}") from error
        if not isinstance(checkpoint, dict):
            raise ValueError(f"checkpoint must contain a JSON object: {path}")
        if type(checkpoint.get("schema_version")) is not int or checkpoint.get(
            "schema_version"
        ) != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint schema_version must be {CURRENT_SCHEMA_VERSION}: {path}"
            )
        if checkpoint.get("input_hash") != expected_input_hash:
            raise ValueError(
                "checkpoint does not match the frozen input; "
                f"expected {expected_input_hash!r}, got {checkpoint.get('input_hash')!r}"
            )
        return checkpoint
