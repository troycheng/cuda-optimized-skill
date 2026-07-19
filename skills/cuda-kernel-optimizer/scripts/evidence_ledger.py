#!/usr/bin/env python3
"""Create and verify an append-only, hash-chained optimization event ledger."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCHEMA = "cuda-optimizer/run-event-v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EVENT_FILE = re.compile(r"([0-9]{20})\.json\Z")
_PENDING_FILE = re.compile(r"\.pending-[0-9]+-[0-9a-f]{16}\Z")
_ZERO_SHA = "0" * 64
_RESERVED_EVENT_TYPES = {"observation_sealed"}


def _load_artifact_store():
    path = Path(__file__).with_name("artifact_store.py")
    spec = importlib.util.spec_from_file_location("cuda_optimizer_ledger_artifacts", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_ARTIFACT_STORE = _load_artifact_store()


class ValidationError(ValueError):
    pass


def _pairs_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_number(token: str):
    raise ValidationError(f"JSON number must be finite: {token}")


def _json_copy(value: Any, field: str = "payload") -> Any:
    if value is None or type(value) in {bool, str, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValidationError(f"{field} numbers must be finite")
        return value
    if type(value) is list:
        return [_json_copy(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if type(value) is dict:
        copied = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ValidationError(f"{field} keys must be non-empty strings")
            copied[key] = _json_copy(item, f"{field}.{key}")
        return copied
    raise ValidationError(f"{field} must contain finite JSON values")


def _sha(value: Any, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{field} must be lowercase SHA-256")
    return value


def _identifier(value: Any, field: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValidationError(f"{field} must be a safe identifier")
    return value


def _canonical_digest(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def seal_record(value: Mapping[str, Any]) -> dict:
    """Return a detached record with a digest over every other field."""
    if not isinstance(value, Mapping):
        raise ValidationError("record must be an object")
    record = _json_copy(dict(value), "record")
    record.pop("record_sha256", None)
    record["record_sha256"] = _canonical_digest(record)
    return record


def _validate_record(value: Any, *, expected_sequence: int, previous_sha256: str) -> dict:
    if type(value) is not dict:
        raise ValidationError(f"ledger sequence {expected_sequence} must contain an object")
    fields = {
        "schema_version",
        "sequence",
        "event_type",
        "contract_sha256",
        "previous_sha256",
        "payload",
        "record_sha256",
    }
    unknown = sorted(set(value) - fields)
    missing = sorted(fields - set(value))
    if unknown or missing:
        raise ValidationError(
            f"ledger sequence {expected_sequence} has an invalid closed record"
        )
    if value["schema_version"] != SCHEMA:
        raise ValidationError(f"ledger sequence {expected_sequence} schema changed")
    if type(value["sequence"]) is not int or value["sequence"] != expected_sequence:
        raise ValidationError(f"ledger sequence gap or mismatch at {expected_sequence}")
    _identifier(value["event_type"], "event_type")
    _sha(value["contract_sha256"], "contract_sha256")
    if _sha(value["previous_sha256"], "previous_sha256") != previous_sha256:
        raise ValidationError(f"ledger chain changed at sequence {expected_sequence}")
    if type(value["payload"]) is not dict:
        raise ValidationError("event payload must be an object")
    record = _json_copy(value, "record")
    recorded_sha = _sha(record.pop("record_sha256"), "record_sha256")
    if _canonical_digest(record) != recorded_sha:
        raise ValidationError(f"ledger record hash changed at sequence {expected_sequence}")
    value_copy = _json_copy(value, "record")
    return value_copy


def _open_lock(path: str | os.PathLike) -> tuple[int, int, Path]:
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(path)))) / ".lock"
    for attempt in range(2):
        try:
            directory_fd, leaf, stable_target = _ARTIFACT_STORE._open_parent_directory(
                target, create=True
            )
            break
        except FileExistsError:
            if attempt == 0:
                continue
            raise ValidationError(f"ledger path could not be created safely: {path}")
        except (OSError, ValueError) as error:
            raise ValidationError(f"ledger path contains a symlink or is unsafe: {path}") from error
    try:
        for attempt in range(3):
            try:
                lock_fd = os.open(
                    leaf,
                    os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=directory_fd,
                )
                break
            except OSError as error:
                if error.errno == errno.ENOENT and attempt < 2:
                    continue
                raise
    except OSError as error:
        os.close(directory_fd)
        raise ValidationError(f"ledger lock is a symlink or unsafe: {stable_target}") from error
    if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
        os.close(lock_fd)
        os.close(directory_fd)
        raise ValidationError("ledger lock must be a regular file")
    return directory_fd, lock_fd, stable_target.parent


def _event_names(directory_fd: int, *, clean_pending: bool) -> list[str]:
    names = []
    for name in os.listdir(directory_fd):
        if name == ".lock":
            continue
        if _PENDING_FILE.fullmatch(name) is not None:
            fd = None
            try:
                fd = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise ValidationError(f"ledger pending entry is unsafe: {name}")
            except OSError as error:
                raise ValidationError(f"ledger pending entry is unsafe: {name}") from error
            finally:
                if fd is not None:
                    os.close(fd)
            if clean_pending:
                try:
                    os.unlink(name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
            continue
        match = _EVENT_FILE.fullmatch(name)
        if match is None:
            raise ValidationError(f"ledger contains unexpected entry: {name}")
        names.append(name)
    names.sort()
    for expected, name in enumerate(names, start=1):
        if int(_EVENT_FILE.fullmatch(name).group(1)) != expected:
            raise ValidationError(f"ledger sequence gap before {name}")
    return names


def _read_event(directory_fd: int, name: str) -> dict:
    fd = None
    try:
        fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValidationError(f"ledger event is not a regular file: {name}")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        try:
            value = json.loads(
                b"".join(chunks).decode("utf-8"),
                object_pairs_hook=_pairs_without_duplicates,
                parse_constant=_invalid_number,
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValidationError(f"invalid ledger event {name}: {error}") from error
        return value
    except OSError as error:
        raise ValidationError(f"ledger event is a symlink or unsafe: {name}") from error
    finally:
        if fd is not None:
            os.close(fd)


def _verify_locked(
    directory_fd: int,
    expected_contract_sha256: str | None,
    *,
    clean_pending: bool = False,
) -> list[dict]:
    names = _event_names(directory_fd, clean_pending=clean_pending)
    records = []
    previous = _ZERO_SHA
    contract_sha = expected_contract_sha256
    if contract_sha is not None:
        _sha(contract_sha, "expected_contract_sha256")
    for sequence, name in enumerate(names, start=1):
        record = _validate_record(
            _read_event(directory_fd, name),
            expected_sequence=sequence,
            previous_sha256=previous,
        )
        if contract_sha is None:
            contract_sha = record["contract_sha256"]
        elif record["contract_sha256"] != contract_sha:
            raise ValidationError(f"ledger contract identity changed at sequence {sequence}")
        previous = record["record_sha256"]
        records.append(record)
    return records


def verify_ledger(
    path: str | os.PathLike, *, expected_contract_sha256: str | None = None
) -> list[dict]:
    """Verify sequence, contract identity, chain links, and record bytes."""
    directory_fd, lock_fd, _root = _open_lock(path)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        return _verify_locked(directory_fd, expected_contract_sha256)
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
            os.close(directory_fd)


def _append_event(
    path: str | os.PathLike,
    *,
    event_type: str,
    contract_sha256: str,
    payload: Mapping[str, Any],
    expected_previous_sha256: str | None = None,
) -> dict:
    """Append one create-once event after verifying the complete existing chain."""
    _identifier(event_type, "event_type")
    _sha(contract_sha256, "contract_sha256")
    if expected_previous_sha256 is not None:
        _sha(expected_previous_sha256, "expected_previous_sha256")
    if not isinstance(payload, Mapping):
        raise ValidationError("payload must be an object")
    clean_payload = _json_copy(dict(payload))
    directory_fd, lock_fd, _root = _open_lock(path)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        records = _verify_locked(directory_fd, contract_sha256, clean_pending=True)
        sequence = len(records) + 1
        previous = records[-1]["record_sha256"] if records else _ZERO_SHA
        if expected_previous_sha256 is not None and previous != expected_previous_sha256:
            raise ValidationError(
                "stale ledger snapshot: previous record changed before append"
            )
        record = seal_record(
            {
                "schema_version": SCHEMA,
                "sequence": sequence,
                "event_type": event_type,
                "contract_sha256": contract_sha256,
                "previous_sha256": previous,
                "payload": clean_payload,
            }
        )
        name = f"{sequence:020d}.json"
        pending = f".pending-{os.getpid()}-{secrets.token_hex(8)}"
        encoded = (
            json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode("utf-8")
        fd = None
        try:
            fd = os.open(
                pending,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o644,
                dir_fd=directory_fd,
            )
            offset = 0
            while offset < len(encoded):
                offset += os.write(fd, encoded[offset:])
            os.fsync(fd)
            os.close(fd)
            fd = None
            os.link(
                pending,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            os.fsync(directory_fd)
        finally:
            if fd is not None:
                os.close(fd)
            try:
                os.unlink(pending, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except FileNotFoundError:
                pass
        return record
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
            os.close(directory_fd)


def append_event(
    path: str | os.PathLike,
    *,
    event_type: str,
    contract_sha256: str,
    payload: Mapping[str, Any],
    expected_previous_sha256: str | None = None,
) -> dict:
    """Append a non-reserved event through the public ledger API."""
    if event_type in _RESERVED_EVENT_TYPES:
        raise ValidationError(
            f"event_type {event_type} is reserved for its deterministic adapter"
        )
    return _append_event(
        path,
        event_type=event_type,
        contract_sha256=contract_sha256,
        payload=payload,
        expected_previous_sha256=expected_previous_sha256,
    )


def _append_reserved_event(
    path: str | os.PathLike,
    *,
    event_type: str,
    contract_sha256: str,
    payload: Mapping[str, Any],
    expected_previous_sha256: str | None = None,
) -> dict:
    """Internal adapter hook; reserved records still use the normal ledger checks."""
    if event_type not in _RESERVED_EVENT_TYPES:
        raise ValidationError(f"event_type is not reserved: {event_type}")
    return _append_event(
        path,
        event_type=event_type,
        contract_sha256=contract_sha256,
        payload=payload,
        expected_previous_sha256=expected_previous_sha256,
    )
