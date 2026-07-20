#!/usr/bin/env python3
"""Execute the one authorized Phase 0 remediation: isolated hashed pip."""

from __future__ import annotations

import importlib.util
import os
import stat
import time
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "cuda-workload-optimizer/readiness-install-v1"


def _load_sibling(name: str, module_name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_STORE = _load_sibling("artifact_store.py", "cuda_readiness_install_store")
_CONTRACT = _load_sibling(
    "readiness_contract.py", "cuda_readiness_install_contract"
)
_PROBE = _load_sibling("readiness_probe.py", "cuda_readiness_install_probe")


def _python_identity(path: str) -> tuple[str | None, str | None, str | None]:
    candidate = Path(path)
    metadata = candidate.lstat()
    symlink_target = (
        os.readlink(candidate) if stat.S_ISLNK(metadata.st_mode) else None
    )
    realpath, digest = _PROBE._executable_identity(str(candidate))
    return realpath, digest, symlink_target


def _result(
    *,
    remediation: Mapping[str, Any],
    status: str,
    reason: str | None,
    started_at: float,
    finished_at: float,
    run_result: Mapping[str, Any] | None = None,
) -> dict:
    execution = run_result or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "authorization_id": remediation.get("authorization_id"),
        "status": status,
        "reason": reason,
        "requirements_sha256": remediation.get("requirements_sha256"),
        "python": remediation.get("python"),
        "returncode": execution.get("returncode"),
        "timed_out": bool(execution.get("timed_out", False)),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": max(0.0, finished_at - started_at),
        "stdout": execution.get("stdout", ""),
        "stderr": execution.get("stderr", ""),
        "logs_truncated": bool(execution.get("logs_truncated", False)),
    }


def install_isolated_pip(
    remediation: Mapping[str, Any],
    *,
    project_root: Path,
    environment_root: Path,
    run_dir: Path,
    deadline_epoch: float,
) -> dict:
    """Install one hash-locked requirements file into an authorized environment."""
    project = _CONTRACT._safe_root(project_root, "project_root")
    environment_root = _CONTRACT._safe_root(
        environment_root, "environment_root"
    )
    try:
        validated = _CONTRACT._validate_remediation(
            remediation,
            field="remediation",
            control_scope="isolated_environment",
            project_root=project,
            environment_root=environment_root,
        )
    except (ValueError, OSError) as error:
        now = time.time()
        return _result(
            remediation=remediation,
            status="failed",
            reason=f"invalid_remediation: {error}",
            started_at=now,
            finished_at=now,
        )
    if validated["mode"] != "isolated_pip":
        now = time.time()
        return _result(
            remediation=validated,
            status="failed",
            reason="unsupported_remediation_mode",
            started_at=now,
            finished_at=now,
        )

    requirements = Path(validated["requirements_file"])
    expected_digest = validated["requirements_sha256"]
    try:
        actual_digest = _STORE.sha256_file(requirements)
    except (OSError, ValueError):
        actual_digest = None
    started_at = time.time()
    if actual_digest != expected_digest:
        result = _result(
            remediation=validated,
            status="failed",
            reason="requirements_digest_mismatch",
            started_at=started_at,
            finished_at=time.time(),
        )
        _STORE.atomic_write_json(
            Path(run_dir)
            / "readiness"
            / "installs"
            / f"{validated['authorization_id']}.json",
            result,
        )
        return result

    remaining = float(deadline_epoch) - started_at
    if remaining <= 0:
        result = _result(
            remediation=validated,
            status="failed",
            reason="readiness_deadline_exhausted",
            started_at=started_at,
            finished_at=time.time(),
        )
    else:
        command = [
            validated["python"],
            "-I",
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "-r",
            validated["requirements_file"],
        ]
        environment = _PROBE._safe_environment(
            Path(run_dir) / "readiness" / "installs" / ".unused-probe-output"
        )
        environment.pop("CUDA_OPTIMIZER_READINESS_OUTPUT", None)
        environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        environment["PIP_NO_INPUT"] = "1"
        python_before = _python_identity(validated["python"])
        try:
            execution = _PROBE._run_bounded(
                command,
                timeout_seconds=min(
                    float(validated["timeout_seconds"]), remaining
                ),
                cwd=project,
                environment=environment,
            )
        except OSError as error:
            execution = {
                "returncode": None,
                "timed_out": False,
                "logs_truncated": False,
                "stdout": "",
                "stderr": _PROBE._redact(str(error)),
            }
            reason = "pip_command_unavailable"
            status = "failed"
        else:
            python_after = _python_identity(validated["python"])
            requirements_after = _STORE.sha256_file(requirements)
            if python_after != python_before:
                status, reason = "failed", "python_identity_changed"
            elif requirements_after != expected_digest:
                status, reason = "failed", "requirements_digest_changed"
            elif execution["timed_out"]:
                status, reason = "failed", "pip_timeout"
            elif execution["returncode"] != 0:
                status, reason = "failed", "pip_failed"
            else:
                status, reason = "succeeded", None
        result = _result(
            remediation=validated,
            status=status,
            reason=reason,
            started_at=started_at,
            finished_at=time.time(),
            run_result=execution,
        )

    _STORE.atomic_write_json(
        Path(run_dir)
        / "readiness"
        / "installs"
        / f"{validated['authorization_id']}.json",
        result,
    )
    return result
