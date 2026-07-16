#!/usr/bin/env python3
"""Versioned, traversal-safe storage for optimizer run artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Any, Optional, Union


CURRENT_SCHEMA_VERSION = 2
_CANDIDATE_ID = re.compile(r"[A-Za-z0-9._-]+")
_PathLike = Union[str, os.PathLike]


def sha256_file(path: _PathLike) -> str:
    """Return the SHA-256 digest of a regular file."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise ValueError(f"input file does not exist or is not a file: {source}")

    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry update, with a portable open fallback."""
    flags = os.O_RDONLY
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags | directory_flag)
    except OSError:
        if not directory_flag:
            raise
        fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _reject_symlink_components(path: _PathLike) -> Path:
    """Return an absolute path only when no existing component is a symlink."""
    target = Path(path).expanduser().absolute()
    current = Path(target.anchor)
    for component in target.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            role = "target" if current == target else "parent"
            raise ValueError(f"JSONL {role} path contains a symlink: {current}")
    return target


def atomic_write_json(path: _PathLike, payload: Any) -> None:
    """Atomically replace *path* with a formatted UTF-8 JSON document."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


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

    target = _reject_symlink_components(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(target.parent)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for line in encoded:
                stream.write(line)
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


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
        "sha256": sha256_file(target),
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
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        with os.fdopen(fd, "a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        return target

    def read_jsonl(self, relative_path: _PathLike) -> list:
        target = self._resolve_relative(relative_path)
        if not target.exists():
            return []
        if not target.is_file():
            raise ValueError(f"JSONL artifact is not a file: {target}")

        records = []
        with target.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
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
        if not path.is_file():
            raise ValueError(f"checkpoint not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            checkpoint = json.load(stream)
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
