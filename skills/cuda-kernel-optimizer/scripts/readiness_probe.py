#!/usr/bin/env python3
"""Run one bounded readiness capability probe and seal its evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping


PROBE_SCHEMA = "cuda-workload-optimizer/readiness-probe-v1"
EXECUTION_SCHEMA = "cuda-workload-optimizer/readiness-execution-v1"
COMPLETION_SCHEMA = "cuda-workload-optimizer/readiness-completion-v1"
PROBE_STATUSES = {"ready", "degraded", "unavailable", "failed"}
MAX_PROBE_BYTES = 1024 * 1024
MAX_LOG_BYTES = 64 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|token|password|passwd|secret)\s*([=:])\s*([^\s]+)"
)


def _load_sibling(name: str, module_name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_ARTIFACT_STORE = _load_sibling(
    "artifact_store.py", "cuda_readiness_probe_artifact_store"
)
_CONTRACT = _load_sibling(
    "readiness_contract.py", "cuda_readiness_probe_contract"
)


class ValidationError(ValueError):
    """Raised when probe input or evidence is unsafe or inconsistent."""


def _strict_json_copy(value: Any, field: str = "probe") -> Any:
    if value is None or type(value) in {bool, str, int}:
        return copy.deepcopy(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValidationError(f"{field} numbers must be finite")
        return value
    if type(value) is list:
        return [
            _strict_json_copy(item, f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValidationError(f"{field} keys must be strings")
            result[key] = _strict_json_copy(item, f"{field}.{key}")
        return result
    raise ValidationError(f"{field} must contain only strict JSON values")


def _closed(value: Any, fields: set[str], name: str) -> dict:
    if type(value) is not dict:
        raise ValidationError(f"{name} must be an object")
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ValidationError(
            f"{name} contains unknown fields: {', '.join(unknown)}"
        )
    missing = sorted(fields - set(value))
    if missing:
        raise ValidationError(
            f"{name} is missing required fields: {', '.join(missing)}"
        )
    return value


def validate_probe(value: Mapping[str, Any], expected_requirement_id: str) -> dict:
    """Validate and detach one readiness-probe-v1 payload."""
    expected = _CONTRACT._identifier(
        expected_requirement_id, "expected_requirement_id"
    )
    probe = _strict_json_copy(value)
    _closed(
        probe,
        {"schema_version", "requirement_id", "status", "observations", "artifacts"},
        "readiness probe",
    )
    if probe["schema_version"] != PROBE_SCHEMA:
        raise ValidationError(f"schema_version must be {PROBE_SCHEMA}")
    requirement_id = _CONTRACT._identifier(
        probe["requirement_id"], "requirement_id"
    )
    if requirement_id != expected:
        raise ValidationError(
            f"requirement_id must match expected requirement {expected}"
        )
    if type(probe["status"]) is not str or probe["status"] not in PROBE_STATUSES:
        raise ValidationError(
            f"status must be one of: {', '.join(sorted(PROBE_STATUSES))}"
        )
    if type(probe["observations"]) is not dict:
        raise ValidationError("observations must be an object")
    artifacts = probe["artifacts"]
    if type(artifacts) is not list:
        raise ValidationError("artifacts must be an array")
    seen_paths = set()
    for index, item in enumerate(artifacts):
        field = f"artifacts[{index}]"
        artifact = _closed(item, {"path", "sha256"}, field)
        path = artifact["path"]
        if type(path) is not str or not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValidationError(f"{field}.path must be a contained relative path")
        if path in seen_paths:
            raise ValidationError(f"duplicate artifact path: {path}")
        seen_paths.add(path)
        digest = artifact["sha256"]
        if type(digest) is not str or _SHA256.fullmatch(digest) is None:
            raise ValidationError(f"{field}.sha256 must be SHA-256")
    return probe


def _pairs_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_number(token: str):
    raise ValidationError(f"JSON number must be finite: {token}")


def _decode_probe(raw: bytes, requirement_id: str) -> dict:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_invalid_number,
        )
    except ValidationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"invalid probe output JSON: {error}") from error
    return validate_probe(value, requirement_id)


def _redact(text: str) -> str:
    redacted = _SENSITIVE_ASSIGNMENT.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", text
    )
    for name, value in os.environ.items():
        lowered = name.lower()
        if (
            len(value) >= 8
            and any(word in lowered for word in ("token", "password", "secret", "api_key", "apikey"))
        ):
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted


def _safe_environment(output_path: Path) -> dict[str, str]:
    allowed = {
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "SYSTEMROOT",
    }
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    environment.setdefault("PATH", os.defpath)
    environment.setdefault("LANG", "C.UTF-8")
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["CUDA_OPTIMIZER_READINESS_OUTPUT"] = str(output_path)
    return environment


def _group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_group(process: subprocess.Popen) -> None:
    process_group = process.pid
    if _group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline and _group_exists(process_group):
            process.poll()
            time.sleep(0.01)
        if _group_exists(process_group):
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_bounded(
    argv: list[str],
    *,
    timeout_seconds: float,
    cwd: Path,
    environment: Mapping[str, str],
    log_limit: int = MAX_LOG_BYTES,
) -> dict:
    half_limit = max(1, log_limit // 2)
    captured = {
        "stdout": {"head": bytearray(), "tail": bytearray(), "total": 0},
        "stderr": {"head": bytearray(), "tail": bytearray(), "total": 0},
    }

    def drain(stream, name: str) -> None:
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                state = captured[name]
                state["total"] += len(chunk)
                head_available = half_limit - len(state["head"])
                if head_available > 0:
                    state["head"].extend(chunk[:head_available])
                    chunk = chunk[head_available:]
                if chunk:
                    state["tail"].extend(chunk)
                    if len(state["tail"]) > half_limit:
                        del state["tail"][:-half_limit]
        finally:
            stream.close()

    def render(name: str) -> tuple[str, bool]:
        state = captured[name]
        total = state["total"]
        if total <= log_limit:
            raw = bytes(state["head"] + state["tail"])
            return _redact(raw.decode("utf-8", errors="replace")), False
        removed = max(0, total - log_limit)
        marker = f"\n...[truncated {removed} bytes]...\n".encode("ascii")
        for _ in range(2):
            payload_budget = max(0, log_limit - len(marker))
            head_budget = payload_budget // 2
            tail_budget = payload_budget - head_budget
            removed = max(0, total - head_budget - tail_budget)
            marker = f"\n...[truncated {removed} bytes]...\n".encode("ascii")
        payload_budget = max(0, log_limit - len(marker))
        head_budget = payload_budget // 2
        tail_budget = payload_budget - head_budget
        raw = (
            bytes(state["head"][:head_budget])
            + marker
            + bytes(state["tail"][-tail_budget:])
        )
        return _redact(raw.decode("utf-8", errors="replace")), True

    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=dict(environment),
        start_new_session=True,
    )
    readers = [
        threading.Thread(target=drain, args=(process.stdout, "stdout"), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, "stderr"), daemon=True),
    ]
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    try:
        while time.monotonic() < deadline:
            if (
                process.poll() is not None
                and not any(reader.is_alive() for reader in readers)
                and not _group_exists(process.pid)
            ):
                break
            time.sleep(0.01)
        else:
            timed_out = True
    finally:
        _stop_group(process)
        for reader in readers:
            reader.join(timeout=1)
        for stream in (process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()
    stdout, stdout_truncated = render("stdout")
    stderr, stderr_truncated = render("stderr")
    return {
        "returncode": process.returncode,
        "timed_out": timed_out,
        "logs_truncated": stdout_truncated or stderr_truncated,
        "stdout": stdout,
        "stderr": stderr,
    }


def _resolve_executable(requested: str, environment: Mapping[str, str]) -> str | None:
    if os.path.isabs(requested):
        return requested if os.path.isfile(requested) else None
    if os.sep in requested:
        return None
    return shutil.which(requested, path=environment.get("PATH"))


def _sha256_file(path: str | os.PathLike) -> str:
    return _ARTIFACT_STORE.sha256_file(path)


def _executable_identity(resolved: str | None) -> tuple[str | None, str | None]:
    if resolved is None:
        return None, None
    real = os.path.realpath(resolved)
    try:
        return real, _sha256_file(real)
    except (OSError, ValueError):
        return real, None


def _input_file_identities(argv: list[str], cwd: Path) -> list[dict]:
    """Bind regular-file argv inputs without interpreting command semantics."""
    identities = []
    for index, value in enumerate(argv[1:], start=1):
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = cwd / candidate
        try:
            metadata = candidate.lstat()
        except (FileNotFoundError, NotADirectoryError, OSError):
            continue
        if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)):
            continue
        realpath = os.path.realpath(candidate)
        try:
            digest = _sha256_file(realpath)
        except (OSError, ValueError):
            digest = None
        identities.append(
            {
                "argv_index": index,
                "path": str(candidate),
                "realpath": realpath,
                "sha256": digest,
                "symlink_target": (
                    os.readlink(candidate) if stat.S_ISLNK(metadata.st_mode) else None
                ),
            }
        )
    return identities


def _tool_version(
    resolved: str | None,
    *,
    cwd: Path,
    environment: Mapping[str, str],
    deadline_epoch: float,
) -> str | None:
    if resolved is None:
        return None
    remaining = deadline_epoch - time.time()
    if remaining <= 0.05:
        return None
    version_environment = dict(environment)
    version_environment.pop("CUDA_OPTIMIZER_READINESS_OUTPUT", None)
    try:
        result = _run_bounded(
            [resolved, "--version"],
            timeout_seconds=min(2.0, remaining),
            cwd=cwd,
            environment=version_environment,
            log_limit=4096,
        )
    except OSError:
        return None
    text = (result["stdout"] + "\n" + result["stderr"]).strip()
    return text.splitlines()[0][:512] if text else None


def _strict_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _remove_leaf(directory_fd: int, leaf: str) -> None:
    try:
        metadata = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(metadata.st_mode):
        raise ValidationError(f"probe artifact leaf is an unsafe directory: {leaf}")
    os.unlink(leaf, dir_fd=directory_fd)


def _directory_identity_matches(directory_fd: int, directory: Path) -> bool:
    marker = directory / ".identity-check"
    current_fd = None
    try:
        current_fd, _leaf, _target = _ARTIFACT_STORE._open_parent_directory(
            marker, create=False
        )
        current = os.fstat(current_fd)
        original = os.fstat(directory_fd)
        return (current.st_dev, current.st_ino) == (original.st_dev, original.st_ino)
    except (OSError, ValueError):
        return False
    finally:
        if current_fd is not None:
            os.close(current_fd)


def _read_attempt(directory_fd: int, leaf: str, target: Path) -> bytes:
    try:
        metadata = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError as error:
        raise ValidationError("probe_output_missing") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValidationError("probe_output_unsafe")
    if metadata.st_nlink != 1:
        raise ValidationError("probe_output_unsafe")
    if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
        raise ValidationError("probe_output_unsafe")
    if metadata.st_size > MAX_PROBE_BYTES:
        raise ValidationError("probe_output_too_large")
    return _ARTIFACT_STORE._read_regular_leaf(
        directory_fd, leaf, target, missing_ok=False
    )


def _synthetic_probe(requirement_id: str, status: str, reason: str) -> dict:
    return {
        "schema_version": PROBE_SCHEMA,
        "requirement_id": requirement_id,
        "status": status,
        "observations": {"reason": reason},
        "artifacts": [],
    }


def run_requirement(
    requirement: Mapping[str, Any],
    *,
    run_dir: Path,
    project_root: Path,
    environment_identity_digest: str,
    deadline_epoch: float,
) -> dict:
    """Run one capability probe and publish marker-last bounded evidence."""
    if type(requirement) is not dict:
        raise ValidationError("requirement must be an object")
    requirement_id = _CONTRACT._identifier(requirement.get("id"), "requirement.id")
    if type(environment_identity_digest) is not str or _SHA256.fullmatch(
        environment_identity_digest
    ) is None:
        raise ValidationError("environment_identity_digest must be SHA-256")
    if isinstance(deadline_epoch, bool) or not isinstance(deadline_epoch, (int, float)) or not math.isfinite(float(deadline_epoch)):
        raise ValidationError("deadline_epoch must be finite")
    project = _CONTRACT._safe_root(project_root, "project_root")
    probe_spec = _CONTRACT._validate_probe(
        requirement.get("probe"), "requirement.probe"
    )
    argv = list(probe_spec["argv"])
    probe_timeout = float(probe_spec["timeout_seconds"])

    probes_dir = Path(run_dir) / "readiness" / "probes"
    attempt_leaf = f".{requirement_id}.attempt-{secrets.token_hex(8)}.json"
    probe_leaf = f"{requirement_id}.json"
    execution_leaf = f"{requirement_id}.execution.json"
    marker_leaf = f"{requirement_id}.complete.json"
    attempt_path = probes_dir / attempt_leaf
    directory_fd, _leaf, _target = _ARTIFACT_STORE._open_parent_directory(
        attempt_path, create=True
    )
    os.fchmod(directory_fd, 0o700)
    try:
        try:
            existing = os.stat(
                marker_leaf, dir_fd=directory_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if not stat.S_ISREG(existing.st_mode):
                raise ValidationError("completion marker is unsafe")
            raise FileExistsError(
                f"readiness completion marker already exists: {marker_leaf}"
            )
        for stale in (probe_leaf, execution_leaf, attempt_leaf):
            _remove_leaf(directory_fd, stale)

        environment = _safe_environment(attempt_path)
        resolved = _resolve_executable(argv[0], environment)
        real_executable, executable_sha256 = _executable_identity(resolved)
        input_identities = _input_file_identities(argv, project)
        tool_version = _tool_version(
            resolved,
            cwd=project,
            environment=environment,
            deadline_epoch=float(deadline_epoch),
        )
        _remove_leaf(directory_fd, attempt_leaf)
        started_at = time.time()
        remaining = float(deadline_epoch) - started_at
        run_result = {
            "returncode": None,
            "timed_out": False,
            "logs_truncated": False,
            "stdout": "",
            "stderr": "",
        }
        if remaining <= 0:
            probe = _synthetic_probe(
                requirement_id, "unavailable", "readiness_deadline_exhausted"
            )
        elif resolved is None:
            probe = _synthetic_probe(
                requirement_id, "unavailable", "probe_command_unavailable"
            )
        else:
            try:
                run_result = _run_bounded(
                    [resolved, *argv[1:]],
                    timeout_seconds=min(probe_timeout, remaining),
                    cwd=project,
                    environment=environment,
                )
            except OSError as error:
                run_result["stderr"] = _redact(str(error))
                probe = _synthetic_probe(
                    requirement_id, "unavailable", "probe_command_unavailable"
                )
            else:
                if run_result["timed_out"]:
                    probe = _synthetic_probe(
                        requirement_id, "unavailable", "probe_timeout"
                    )
                elif run_result["returncode"] != 0:
                    probe = _synthetic_probe(
                        requirement_id,
                        "failed",
                        f"probe_returncode_{run_result['returncode']}",
                    )
                else:
                    try:
                        raw_probe = _read_attempt(
                            directory_fd, attempt_leaf, attempt_path
                        )
                        probe = _decode_probe(raw_probe, requirement_id)
                    except ValidationError as error:
                        reason = str(error)
                        if reason not in {
                            "probe_output_missing",
                            "probe_output_unsafe",
                            "probe_output_too_large",
                        }:
                            reason = f"invalid_probe_output: {reason}"
                        probe = _synthetic_probe(requirement_id, "failed", reason)

        finished_at = time.time()
        resolved_after = _resolve_executable(argv[0], environment)
        real_executable_after, executable_sha256_after = _executable_identity(
            resolved_after
        )
        input_identities_after = _input_file_identities(argv, project)
        executable_identity_stable = (
            resolved_after == resolved
            and real_executable_after == real_executable
            and executable_sha256_after == executable_sha256
        )
        input_identities_stable = input_identities_after == input_identities
        if not executable_identity_stable:
            probe = _synthetic_probe(
                requirement_id, "failed", "executable_identity_changed"
            )
        elif not input_identities_stable:
            probe = _synthetic_probe(
                requirement_id, "failed", "probe_input_identity_changed"
            )
        _remove_leaf(directory_fd, attempt_leaf)
        if not _directory_identity_matches(directory_fd, probes_dir):
            raise ValidationError("readiness probe directory identity was replaced")

        execution = {
            "schema_version": EXECUTION_SCHEMA,
            "requirement_id": requirement_id,
            "argv_sha256": hashlib.sha256(
                json.dumps(argv, separators=(",", ":"), ensure_ascii=False).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "requested_executable": argv[0],
            "resolved_executable": resolved,
            "executable_realpath": real_executable,
            "executable_sha256": executable_sha256,
            "executable_identity_after": {
                "resolved_executable": resolved_after,
                "executable_realpath": real_executable_after,
                "executable_sha256": executable_sha256_after,
            },
            "probe_input_identities": input_identities,
            "probe_input_identities_after": input_identities_after,
            "identity_stable": (
                executable_identity_stable and input_identities_stable
            ),
            "tool_version": tool_version,
            "returncode": run_result["returncode"],
            "timed_out": run_result["timed_out"],
            "duration_seconds": max(0.0, finished_at - started_at),
            "started_at": started_at,
            "finished_at": finished_at,
            "stdout": run_result["stdout"],
            "stderr": run_result["stderr"],
            "logs_truncated": run_result["logs_truncated"],
            "environment_identity_digest": environment_identity_digest,
            "uid": os.getuid() if hasattr(os, "getuid") else None,
            "container_identity": os.environ.get("CUDA_OPTIMIZER_CONTAINER_ID"),
            "visible_devices": {
                "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "nvidia": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            },
            "gpu_identity": os.environ.get("CUDA_OPTIMIZER_GPU_IDENTITY"),
            "permission_state": os.environ.get(
                "CUDA_OPTIMIZER_COUNTER_PERMISSION"
            ),
        }
        probe_bytes = _strict_bytes(probe)
        execution_bytes = _strict_bytes(execution)
        marker = {
            "schema_version": COMPLETION_SCHEMA,
            "requirement_id": requirement_id,
            "probe_sha256": hashlib.sha256(probe_bytes).hexdigest(),
            "execution_sha256": hashlib.sha256(execution_bytes).hexdigest(),
            "published_at": time.time(),
        }
        for leaf, payload in (
            (probe_leaf, probe_bytes),
            (execution_leaf, execution_bytes),
            (marker_leaf, _strict_bytes(marker)),
        ):
            _ARTIFACT_STORE._atomic_write_leaf(
                directory_fd, leaf, probes_dir / leaf, payload
            )
        os.fsync(directory_fd)
        if not _directory_identity_matches(directory_fd, probes_dir):
            _remove_leaf(directory_fd, marker_leaf)
            raise ValidationError("readiness probe directory identity was replaced")
        return probe
    except BaseException:
        try:
            _remove_leaf(directory_fd, marker_leaf)
        except (OSError, ValueError):
            pass
        raise
    finally:
        try:
            _remove_leaf(directory_fd, attempt_leaf)
        except (OSError, ValueError):
            pass
        os.close(directory_fd)
