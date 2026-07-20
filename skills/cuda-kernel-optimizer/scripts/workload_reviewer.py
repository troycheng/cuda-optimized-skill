#!/usr/bin/env python3
"""Advisory JSON protocol for a user-supplied local reviewer CLI."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


REQUEST_SCHEMA = "cuda-workload-optimizer/review-request-v1"
RESPONSE_SCHEMA = "cuda-workload-optimizer/review-v1"
ARTIFACT_SCHEMA = "cuda-workload-optimizer/review-artifact-v1"
AGGREGATE_SCHEMA = "cuda-workload-optimizer/review-aggregate-v1"
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_PROVIDER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_SECRET_NAME = re.compile(
    r"(^|[_-])(api[_-]?key|authorization|cookie|credential|password|secret|token)($|[_-])",
    re.IGNORECASE,
)
_SECRET_LOG = re.compile(
    r'''(?i)(["']?\b[A-Z0-9_]{0,128}(?:API[_-]?KEY|AUTH|COOKIE|CREDENTIAL|PASSWORD|SECRET|TOKEN)[A-Z0-9_]{0,128}\b["']?\s*[:=]\s*)(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\r\n,;}]+)'''
)
_SAFE_ENV = {
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONPATH",
    "TMPDIR",
}


class ReviewerError(ValueError):
    """Raised when reviewer input or output violates the advisory protocol."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json_copy(value: Any, field: str) -> Any:
    if value is None or type(value) in {bool, str, int}:
        return copy.deepcopy(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ReviewerError(f"{field} numbers must be finite")
        return value
    if type(value) is list:
        return [_json_copy(item, f"{field}[]") for item in value]
    if type(value) is dict:
        result = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ReviewerError(f"{field} keys must be non-empty strings")
            if _SECRET_NAME.search(key):
                raise ReviewerError(f"{field} must not contain credentials: {key}")
            result[key] = _json_copy(item, f"{field}.{key}")
        return result
    raise ReviewerError(f"{field} must contain JSON-compatible values")


def _object(value: Any, field: str) -> dict:
    if type(value) is not dict:
        raise ReviewerError(f"{field} must be an object")
    return value


def _closed(value: Mapping[str, Any], fields: set[str], field: str) -> None:
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ReviewerError(f"{field} contains unknown fields: {', '.join(unknown)}")


def _required(value: Mapping[str, Any], fields: set[str], field: str) -> None:
    missing = sorted(fields - set(value))
    if missing:
        raise ReviewerError(f"{field} is missing required fields: {', '.join(missing)}")


def _string(value: Any, field: str, maximum: int = 4096) -> str:
    if type(value) is not str or not value.strip():
        raise ReviewerError(f"{field} must be a non-empty string")
    if len(value) > maximum:
        raise ReviewerError(f"{field} exceeds {maximum} characters")
    return value


def build_review_request(
    *,
    diagnosis: Mapping[str, Any],
    change_set: Mapping[str, Any],
    redacted_diff: str,
    experiment: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
) -> dict:
    """Build a detached request whose digest covers every advisory input."""
    hashes = _object(artifact_hashes, "artifact_hashes")
    for name, digest in hashes.items():
        _string(name, "artifact_hashes key", maximum=512)
        if type(digest) is not str or _SHA256.fullmatch(digest) is None:
            raise ReviewerError(f"artifact_hashes.{name} must be a SHA-256 digest")
    diff = redacted_diff
    if type(diff) is not str:
        raise ReviewerError("redacted_diff must be a string")
    if len(diff.encode("utf-8")) > 256 * 1024:
        raise ReviewerError("redacted_diff exceeds 262144 bytes")
    base = {
        "schema_version": REQUEST_SCHEMA,
        "diagnosis": _json_copy(_object(diagnosis, "diagnosis"), "diagnosis"),
        "change_set": _json_copy(_object(change_set, "change_set"), "change_set"),
        "redacted_diff": diff,
        "experiment": _json_copy(_object(experiment, "experiment"), "experiment"),
        "artifact_hashes": copy.deepcopy(hashes),
    }
    return {**base, "request_digest": _digest(base)}


def request_digest(request: Mapping[str, Any]) -> str:
    """Recompute the digest of a review request without trusting its digest field."""
    value = copy.deepcopy(_object(request, "request"))
    value.pop("request_digest", None)
    return _digest(value)


def validate_review_response(
    value: Mapping[str, Any], request: Mapping[str, Any]
) -> dict:
    """Validate a response that can advise but cannot execute or promote."""
    response = _object(value, "review_response")
    fields = {
        "schema_version",
        "request_digest",
        "verdict",
        "concerns",
        "suggested_experiments",
    }
    _closed(response, fields, "review_response")
    _required(response, fields, "review_response")
    if response["schema_version"] != RESPONSE_SCHEMA:
        raise ReviewerError(f"review_response.schema_version must be {RESPONSE_SCHEMA}")
    expected_digest = request_digest(request)
    if request.get("request_digest") != expected_digest:
        raise ReviewerError("review request digest is invalid")
    if response["request_digest"] != expected_digest:
        raise ReviewerError("review response digest does not match request digest")
    if response["verdict"] not in {"support", "challenge", "insufficient"}:
        raise ReviewerError("review verdict must be support, challenge, or insufficient")

    concerns = response["concerns"]
    if type(concerns) is not list or len(concerns) > 32:
        raise ReviewerError("review concerns must be an array with at most 32 entries")
    for index, item in enumerate(concerns):
        concern = _object(item, f"review_response.concerns[{index}]")
        concern_fields = {"severity", "category", "message"}
        _closed(concern, concern_fields, f"review_response.concerns[{index}]")
        _required(concern, concern_fields, f"review_response.concerns[{index}]")
        if concern["severity"] not in {"low", "medium", "high"}:
            raise ReviewerError(f"review_response.concerns[{index}].severity is invalid")
        _string(concern["category"], f"review_response.concerns[{index}].category", 128)
        _string(concern["message"], f"review_response.concerns[{index}].message")

    suggestions = response["suggested_experiments"]
    if type(suggestions) is not list or len(suggestions) > 32:
        raise ReviewerError(
            "review suggested_experiments must be an array with at most 32 entries"
        )
    for index, item in enumerate(suggestions):
        _string(item, f"review_response.suggested_experiments[{index}]", 2048)
    return copy.deepcopy(response)


def _duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReviewerError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_response(payload: bytes) -> dict:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_pairs,
            parse_constant=lambda token: (_raise_number(token)),
        )
    except ReviewerError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ReviewerError(f"reviewer stdout must be strict JSON: {error}") from error
    return _object(value, "review_response")


def _raise_number(token: str):
    raise ReviewerError(f"reviewer JSON number must be finite: {token}")


class _BoundedCapture:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.data = bytearray()
        self.truncated = False

    def append(self, chunk: bytes) -> None:
        available = max(0, self.limit - len(self.data))
        self.data.extend(chunk[:available])
        if len(chunk) > available:
            self.truncated = True


def _drain(stream, capture: _BoundedCapture) -> None:
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            capture.append(chunk)
    finally:
        stream.close()


def _write_stdin(stream, payload: bytes, errors: list[str]) -> None:
    try:
        stream.write(payload)
        stream.flush()
    except (BrokenPipeError, OSError) as error:
        errors.append(f"reviewer stdin failed: {error}")
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_group(process) -> None:
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


def _environment() -> tuple[dict, tuple[str, ...]]:
    inherited = dict(os.environ)
    values = tuple(
        value for name, value in inherited.items() if _SECRET_NAME.search(name) and value
    )
    environment = {
        name: value
        for name, value in inherited.items()
        if name in _SAFE_ENV and not _SECRET_NAME.search(name)
    }
    return environment, values


def _redact(payload: bytes, secrets: Sequence[str]) -> str:
    value = payload.decode("utf-8", errors="replace")
    value = _SECRET_LOG.sub(lambda match: f"{match.group(1)}[REDACTED]", value)
    for secret in sorted(set(secrets), key=len, reverse=True):
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return value


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


def _artifact(
    request: Mapping[str, Any],
    *,
    status: str,
    response: dict | None,
    execution: Mapping[str, Any],
) -> dict:
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "status": status,
        "request_digest": request_digest(request),
        "response": copy.deepcopy(response),
        "execution": copy.deepcopy(execution),
    }


def run_reviewer(
    config: Mapping[str, Any],
    request: Mapping[str, Any],
    run_dir: str | os.PathLike[str],
    *,
    output_limit_bytes: int = 256 * 1024,
) -> dict:
    """Run a local CLI in advisory mode and always persist a review artifact."""
    configuration = _object(config, "reviewer config")
    _closed(configuration, {"argv", "timeout_seconds"}, "reviewer config")
    _required(configuration, {"argv", "timeout_seconds"}, "reviewer config")
    argv = configuration["argv"]
    if type(argv) is not list or not argv or any(
        type(item) is not str or not item for item in argv
    ):
        raise ReviewerError("reviewer argv must be a non-empty string array")
    timeout = configuration["timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ReviewerError("reviewer timeout_seconds must be numeric")
    if not math.isfinite(float(timeout)) or not 0.05 <= timeout <= 3600:
        raise ReviewerError("reviewer timeout_seconds must be between 0.05 and 3600")
    if isinstance(output_limit_bytes, bool) or not isinstance(output_limit_bytes, int):
        raise ReviewerError("output_limit_bytes must be an integer")
    if not 128 <= output_limit_bytes <= 1024 * 1024:
        raise ReviewerError("output_limit_bytes must be between 128 and 1048576")
    expected = request_digest(request)
    if request.get("request_digest") != expected:
        raise ReviewerError("review request digest is invalid")

    run_root = Path(run_dir).expanduser().resolve(strict=False)
    run_root.mkdir(parents=True, exist_ok=True)
    stdin_payload = _canonical_bytes(request) + b"\n"
    stdout = _BoundedCapture(output_limit_bytes)
    stderr = _BoundedCapture(output_limit_bytes)
    environment, secrets = _environment()
    started = time.monotonic()
    exit_code = None
    timed_out = False
    failure = None
    response = None
    deadline = started + float(timeout)

    with tempfile.TemporaryDirectory(prefix="reviewer-cwd-", dir=run_root) as cwd:
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=environment,
                stdin=subprocess.PIPE,
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
            writer_errors: list[str] = []
            writer = threading.Thread(
                target=_write_stdin,
                args=(process.stdin, stdin_payload, writer_errors),
                daemon=True,
            )
            writer.start()
            try:
                exit_code = process.wait(
                    timeout=max(0.001, deadline - time.monotonic())
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                failure = f"reviewer exceeded {timeout} seconds"
                _stop_group(process)
                exit_code = process.returncode
            else:
                if _process_group_exists(process.pid):
                    _stop_group(process)
            writer.join(timeout=1)
            for reader in readers:
                reader.join(timeout=1)
            if failure is None and writer_errors:
                failure = writer_errors[0]
        except (FileNotFoundError, OSError) as error:
            failure = f"reviewer unavailable: {error}"

    if failure is None and exit_code != 0:
        failure = f"reviewer exited with status {exit_code}"
    if failure is None and stdout.truncated:
        failure = f"reviewer stdout exceeds {output_limit_bytes} bytes"
    if failure is None:
        try:
            response = validate_review_response(_parse_response(bytes(stdout.data)), request)
        except ReviewerError as error:
            failure = str(error)

    execution = {
        "argv_sha256": _digest(argv),
        "stdin_sha256": hashlib.sha256(stdin_payload).hexdigest(),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_seconds": time.monotonic() - started,
        "stderr": _redact(bytes(stderr.data), secrets),
        "stderr_truncated": stderr.truncated,
        "failure": failure,
    }
    artifact = _artifact(
        request,
        status="completed" if failure is None else "unavailable",
        response=response,
        execution=execution,
    )
    _atomic_json(run_root / "review.json", artifact)
    return artifact


def run_reviewers(
    configs: Sequence[Mapping[str, Any]],
    request: Mapping[str, Any],
    run_dir: str | os.PathLike[str],
    *,
    total_timeout_seconds: float = 180.0,
) -> dict:
    """Run named advisory reviewers concurrently under one total wait bound."""
    if not isinstance(configs, Sequence) or isinstance(
        configs, (str, bytes, bytearray)
    ) or not configs:
        raise ReviewerError("reviewers must be a non-empty sequence")
    if len(configs) > 8:
        raise ReviewerError("reviewers must contain at most 8 providers")
    if (
        isinstance(total_timeout_seconds, bool)
        or not isinstance(total_timeout_seconds, (int, float))
        or not math.isfinite(float(total_timeout_seconds))
        or not 1 <= float(total_timeout_seconds) <= 180
    ):
        raise ReviewerError("total_timeout_seconds must be between 1 and 180")
    expected = request_digest(request)
    if request.get("request_digest") != expected:
        raise ReviewerError("review request digest is invalid")

    normalized = []
    providers = set()
    cleanup_reserve = min(4.0, float(total_timeout_seconds) * 0.25)
    provider_deadline = max(
        0.05, float(total_timeout_seconds) - cleanup_reserve
    )
    for index, raw in enumerate(configs):
        config = _object(raw, f"reviewers[{index}]")
        _closed(config, {"provider", "argv", "timeout_seconds"}, f"reviewers[{index}]")
        _required(config, {"provider", "argv", "timeout_seconds"}, f"reviewers[{index}]")
        provider = _string(config["provider"], f"reviewers[{index}].provider", 64)
        if _PROVIDER.fullmatch(provider) is None:
            raise ReviewerError(f"reviewers[{index}].provider is invalid")
        if provider in providers:
            raise ReviewerError("reviewer providers must be unique")
        providers.add(provider)
        timeout = config["timeout_seconds"]
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or not 1 <= float(timeout) <= 3600
        ):
            raise ReviewerError(
                f"reviewers[{index}].timeout_seconds must be between 1 and 3600"
            )
        argv = config["argv"]
        if type(argv) is not list or not argv or any(
            type(item) is not str or not item for item in argv
        ):
            raise ReviewerError(f"reviewers[{index}].argv must be a non-empty string array")
        normalized.append(
            {
                "provider": provider,
                "argv": list(argv),
                "timeout_seconds": min(
                    float(timeout), provider_deadline
                ),
            }
        )

    run_root = Path(run_dir).expanduser().resolve(strict=False)
    run_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def execute(config: Mapping[str, Any]) -> dict:
        provider = config["provider"]
        try:
            artifact = run_reviewer(
                {
                    "argv": config["argv"],
                    "timeout_seconds": config["timeout_seconds"],
                },
                request,
                run_root / "reviewers" / provider,
            )
            execution = artifact.get("execution", {})
            return {
                "provider": provider,
                "status": artifact["status"],
                "response": copy.deepcopy(artifact.get("response")),
                "failure": execution.get("failure"),
                "duration_seconds": float(execution.get("duration_seconds", 0.0)),
            }
        except (OSError, ReviewerError, RuntimeError) as error:
            return {
                "provider": provider,
                "status": "unavailable",
                "response": None,
                "failure": str(error),
                "duration_seconds": max(0.0, time.monotonic() - started),
            }

    with ThreadPoolExecutor(max_workers=len(normalized)) as executor:
        futures = [executor.submit(execute, config) for config in normalized]
        reviews = [future.result() for future in futures]

    elapsed = max(0.0, time.monotonic() - started)
    requested = [item["provider"] for item in normalized]
    completed = [
        item["provider"] for item in reviews if item["status"] == "completed"
    ]
    failed = [
        item["provider"] for item in reviews if item["status"] != "completed"
    ]
    aggregate = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "completed" if completed else "unavailable",
        "request_digest": expected,
        "providers_requested": requested,
        "providers_completed": completed,
        "failed_providers": failed,
        "total_timeout_seconds": float(total_timeout_seconds),
        "total_wait_seconds": float(elapsed),
        "reviews": reviews,
    }
    _atomic_json(run_root / "review.json", aggregate)
    return aggregate


def write_skipped_review(request: Mapping[str, Any], run_dir: str | os.PathLike[str]) -> dict:
    """Record that no reviewer was configured without changing the decision path."""
    artifact = _artifact(
        request,
        status="skipped",
        response=None,
        execution={"failure": None, "reason": "reviewer not configured"},
    )
    _atomic_json(Path(run_dir).expanduser().resolve(strict=False) / "review.json", artifact)
    return artifact
