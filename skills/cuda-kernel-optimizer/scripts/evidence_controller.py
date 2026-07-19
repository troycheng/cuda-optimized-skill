#!/usr/bin/env python3
"""Controller-owned execution and sealing for V3 gate adapters."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import resource
import signal
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


class ValidationError(ValueError):
    pass


def _sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"cuda_controller_{name}", path)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise ValidationError(f"cannot load {name}.py")
    spec.loader.exec_module(module)
    return module


_ARTIFACT_STORE = _sibling("artifact_store")
_GATE_EVIDENCE = _sibling("gate_evidence")
_DIAGNOSTIC_EVIDENCE = _sibling("diagnostic_evidence")
_SUMMARY = _sibling("evidence_summary")
_LEDGER = _sibling("evidence_ledger")

_GATE_KINDS = {
    "correctness_reference",
    "dispatch_identity",
    "target_compile_probe",
    "candidate_correctness",
    "paired_measurement",
    "workload_replay",
}
_DIAGNOSTIC_KINDS = {"nsys_timeline", "pytorch_profile"}
_KINDS = _GATE_KINDS | _DIAGNOSTIC_KINDS
_PRODUCERS = {
    "correctness_reference": "correctness-reference-adapter",
    "dispatch_identity": "dispatch-identity-adapter",
    "target_compile_probe": "compiler-evidence-adapter",
    "candidate_correctness": "candidate-correctness-adapter",
    "paired_measurement": "paired-measurement-adapter",
    "workload_replay": "workload-replay-adapter",
    "nsys_timeline": "nsys-timeline-adapter",
    "pytorch_profile": "pytorch-profile-adapter",
}
_IDENTIFIER = __import__("re").compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = __import__("re").compile(r"[0-9a-f]{64}\Z")
_ADAPTER_FIELDS = {
    "id",
    "version",
    "path",
    "entrypoint_sha256",
    "runtime_path",
    "runtime_sha256",
    "implementation_sha256",
    "timeout_seconds",
    "max_output_bytes",
}
_MAX_REQUEST_BYTES = 64 * 1024
_RUNTIME_MODE = "python-isolated-v1"


def _identifier(value: Any, label: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a safe identifier")
    return value


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be lowercase SHA-256")
    return value


def _closed(value: Any, fields: set[str], label: str) -> dict:
    if type(value) is not dict:
        raise ValidationError(f"{label} must be an object")
    missing = fields - set(value)
    extra = set(value) - fields
    if missing or extra:
        raise ValidationError(
            f"{label} must be closed; missing={sorted(missing)} extra={sorted(extra)}"
        )
    return value


def _json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("adapter request must contain finite JSON values") from exc


def adapter_implementation_sha256(
    *,
    producer_id: str,
    producer_version: str,
    entrypoint_sha256: str,
    runtime_sha256: str,
) -> str:
    """Bind the isolated runtime and self-contained adapter entrypoint."""
    material = {
        "schema_version": "cuda-optimizer/adapter-implementation-v1",
        "producer_id": _identifier(producer_id, "producer_id"),
        "producer_version": producer_version,
        "runtime_mode": _RUNTIME_MODE,
        "runtime_sha256": _sha(runtime_sha256, "runtime_sha256"),
        "entrypoint_sha256": _sha(entrypoint_sha256, "entrypoint_sha256"),
    }
    return hashlib.sha256(_json_bytes(material)).hexdigest()


def _adapter_spec(kind: str, value: Any) -> dict:
    spec = _closed(value, _ADAPTER_FIELDS, f"adapter {kind}")
    if spec["id"] != _PRODUCERS[kind] or spec["version"] != "1.0.0":
        raise ValidationError(f"untrusted producer for {kind}")
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(spec["path"]))))
    entrypoint_sha = _sha(spec["entrypoint_sha256"], "entrypoint_sha256")
    runtime_path = Path(
        os.path.abspath(os.path.expanduser(os.fspath(spec["runtime_path"])))
    )
    runtime_sha = _sha(spec["runtime_sha256"], "runtime_sha256")
    digest = _sha(spec["implementation_sha256"], "implementation_sha256")
    expected_digest = adapter_implementation_sha256(
        producer_id=spec["id"],
        producer_version=spec["version"],
        entrypoint_sha256=entrypoint_sha,
        runtime_sha256=runtime_sha,
    )
    if digest != expected_digest:
        raise ValidationError("adapter implementation digest does not match its materials")
    timeout = spec["timeout_seconds"]
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
        raise ValidationError("timeout_seconds must be positive and finite")
    output_limit = spec["max_output_bytes"]
    if type(output_limit) is not int or output_limit < 1:
        raise ValidationError("max_output_bytes must be a positive integer")
    return {
        "id": spec["id"],
        "version": spec["version"],
        "path": path,
        "entrypoint_sha256": entrypoint_sha,
        "runtime_path": runtime_path,
        "runtime_sha256": runtime_sha,
        "implementation_sha256": digest,
        "timeout_seconds": float(timeout),
        "max_output_bytes": output_limit,
    }


class EvidenceController:
    """Hold the seal key and mediate every authoritative adapter execution.

    Production callers must run this object inside the dedicated Controller
    process. Planner and adapter processes receive requests, never this object
    or its key.
    """

    def __init__(
        self,
        *,
        run_id: str,
        ledger_id: str,
        contract_sha256: str,
        environment_sha256: str,
        ledger_path: str | os.PathLike,
        artifact_root: str | os.PathLike,
        controller_seal_key: bytes,
        adapters: Mapping[str, Mapping[str, Any]],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._run_id = _identifier(run_id, "run_id")
        self._ledger_id = _identifier(ledger_id, "ledger_id")
        self._contract_sha256 = _sha(contract_sha256, "contract_sha256")
        self._environment_sha256 = _sha(
            environment_sha256, "environment_sha256"
        )
        if not isinstance(controller_seal_key, bytes) or len(controller_seal_key) < 32:
            raise ValidationError("controller_seal_key must contain at least 32 bytes")
        self._seal_key = bytes(controller_seal_key)
        self._ledger_path = Path(
            os.path.abspath(os.path.expanduser(os.fspath(ledger_path)))
        )
        self._artifact_root = Path(
            os.path.abspath(os.path.expanduser(os.fspath(artifact_root)))
        )
        if self._artifact_root.is_symlink() or not self._artifact_root.is_dir():
            raise ValidationError("artifact_root is missing, a symlink, or unsafe")
        if not callable(clock):
            raise ValidationError("clock must be callable")
        self._clock = clock
        if not isinstance(adapters, Mapping) or not adapters:
            raise ValidationError("adapters must be a non-empty mapping")
        unknown = set(adapters) - _KINDS
        if unknown:
            raise ValidationError(f"unsupported adapter kinds: {sorted(unknown)}")
        self._adapters = {
            kind: _adapter_spec(kind, dict(value)) for kind, value in adapters.items()
        }

    def _capture_implementation(self, spec: Mapping[str, Any]) -> tuple[bytes, bytes]:
        try:
            entrypoint = _ARTIFACT_STORE.read_regular_bytes(spec["path"])
            runtime = _ARTIFACT_STORE.read_regular_bytes(spec["runtime_path"])
        except ValueError as exc:
            raise ValidationError("adapter implementation is missing or unsafe") from exc
        if hashlib.sha256(entrypoint).hexdigest() != spec["entrypoint_sha256"]:
            raise ValidationError("adapter entrypoint identity changed")
        if hashlib.sha256(runtime).hexdigest() != spec["runtime_sha256"]:
            raise ValidationError("adapter runtime identity changed")
        metadata = os.stat(spec["runtime_path"], follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or not metadata.st_mode & stat.S_IXUSR:
            raise ValidationError("adapter runtime must be an executable regular file")
        return entrypoint, runtime

    def _find_existing_observation(
        self,
        *,
        observation_id: str,
    ) -> dict | None:
        if not os.path.lexists(self._ledger_path):
            return None
        records = _LEDGER.verify_ledger(
            self._ledger_path,
            expected_contract_sha256=self._contract_sha256,
        )
        matches = [
            record
            for record in records
            if record["event_type"] == "observation_sealed"
            and record["payload"].get("observation_id") == observation_id
        ]
        if len(matches) > 1:
            raise ValidationError(f"duplicate observation_id in ledger: {observation_id}")
        if not matches:
            return None
        record = matches[0]
        _SUMMARY._validate_payload(
            record["payload"],
            artifact_root=self._artifact_root,
            contract_sha256=self._contract_sha256,
            environment_sha256=self._environment_sha256,
            run_id=self._run_id,
            ledger_id=self._ledger_id,
            controller_seal_key=self._seal_key,
            as_of=0.0,
            max_age_seconds=1.0,
        )
        return record

    def _reuse_existing(
        self,
        record: Mapping[str, Any],
        *,
        adapter_request_sha256: str,
        implementation_sha256: str,
    ) -> dict:
        payload = record["payload"]
        if (
            payload.get("run_id") != self._run_id
            or payload.get("ledger_id") != self._ledger_id
            or payload.get("adapter_request_sha256") != adapter_request_sha256
            or payload.get("adapter_implementation_sha256")
            != implementation_sha256
        ):
            raise ValidationError(
                f"observation_id conflicts with another request: {payload.get('observation_id')}"
            )
        artifact_path = self._artifact_root / payload["artifact"]["path"]
        return {"record": dict(record), "artifact_path": str(artifact_path)}

    def run_and_seal(
        self,
        *,
        kind: str,
        observation_id: str,
        request: Mapping[str, Any],
        expected_previous_sha256: str | None = None,
    ) -> dict:
        if kind not in self._adapters:
            raise ValidationError(f"adapter kind is not allowlisted: {kind}")
        observation = _identifier(observation_id, "observation_id")
        if not isinstance(request, Mapping):
            raise ValidationError("adapter request must be an object")
        spec = self._adapters[kind]
        trusted_request = {
            "schema_version": "cuda-optimizer/evidence-adapter-request-v1",
            "run_id": self._run_id,
            "ledger_id": self._ledger_id,
            "contract_sha256": self._contract_sha256,
            "environment_sha256": self._environment_sha256,
            "kind": kind,
            "input": dict(request),
        }
        request_bytes = _json_bytes(trusted_request)
        if len(request_bytes) > _MAX_REQUEST_BYTES:
            raise ValidationError("adapter request exceeds controller byte limit")
        request_sha = hashlib.sha256(request_bytes).hexdigest()
        existing = self._find_existing_observation(observation_id=observation)
        if existing is not None:
            return self._reuse_existing(
                existing,
                adapter_request_sha256=request_sha,
                implementation_sha256=spec["implementation_sha256"],
            )
        entrypoint, _runtime = self._capture_implementation(spec)
        environment = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONNOUSERSITE": "1",
        }
        with tempfile.TemporaryDirectory(
            prefix=".adapter-", dir=self._artifact_root
        ) as staging_value:
            staging = Path(staging_value)
            snapshot = staging / "adapter.py"
            _ARTIFACT_STORE.create_regular_bytes(snapshot, entrypoint)
            os.chmod(snapshot, 0o400, follow_symlinks=False)
            with tempfile.TemporaryFile() as output:
                def limit_adapter_output() -> None:
                    limit = spec["max_output_bytes"] + 1
                    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))

                try:
                    process = subprocess.Popen(
                        [str(spec["runtime_path"]), "-I", "-S", str(snapshot)],
                        stdin=subprocess.PIPE,
                        stdout=output,
                        stderr=subprocess.DEVNULL,
                        cwd=staging,
                        env=environment,
                        preexec_fn=limit_adapter_output,
                        start_new_session=True,
                    )
                    try:
                        process.communicate(
                            input=request_bytes,
                            timeout=spec["timeout_seconds"],
                        )
                    except subprocess.TimeoutExpired as exc:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                        raise ValidationError("allowlisted adapter execution timed out") from exc
                    finally:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                except OSError as exc:
                    raise ValidationError("allowlisted adapter execution failed") from exc
                output.seek(0)
                adapter_output = output.read(spec["max_output_bytes"] + 1)
            if hashlib.sha256(_ARTIFACT_STORE.read_regular_bytes(snapshot)).hexdigest() != spec[
                "entrypoint_sha256"
            ]:
                raise ValidationError("captured adapter entrypoint changed during execution")
            returncode = process.returncode
        if returncode != 0:
            raise ValidationError("allowlisted adapter returned a non-zero status")
        if len(adapter_output) > spec["max_output_bytes"]:
            raise ValidationError("adapter measurement exceeds max_output_bytes")
        # The source path may be mutable, but it cannot affect the captured
        # executable. Rechecking detects drift and prevents sealing stale code.
        self._capture_implementation(spec)
        recorded_at = self._clock()
        evidence_module = (
            _GATE_EVIDENCE if kind in _GATE_KINDS else _DIAGNOSTIC_EVIDENCE
        )
        derive = (
            evidence_module.derive_gate_evidence
            if kind in _GATE_KINDS
            else evidence_module.derive_diagnostic_evidence
        )
        evidence = derive(
            adapter_output,
            kind=kind,
            producer_id=spec["id"],
            producer_version=spec["version"],
            implementation_sha256=spec["implementation_sha256"],
            adapter_request_sha256=request_sha,
            contract_sha256=self._contract_sha256,
            environment_sha256=self._environment_sha256,
            recorded_at=recorded_at,
        )
        artifact_sha = hashlib.sha256(evidence).hexdigest()
        artifact_name = f"sealed-{observation}-{artifact_sha[:16]}.json"
        artifact_path = self._artifact_root / artifact_name
        try:
            _ARTIFACT_STORE.create_regular_bytes(artifact_path, evidence)
        except FileExistsError:
            try:
                published = _ARTIFACT_STORE.read_regular_bytes(artifact_path)
            except ValueError as exc:
                raise ValidationError(
                    "existing observation artifact is unsafe or unreadable"
                ) from exc
            if published != evidence:
                raise ValidationError(
                    "observation artifact name conflicts with different bytes"
                )
        artifact = {
            "path": artifact_name,
            "sha256": artifact_sha,
            "size_bytes": len(evidence),
        }
        try:
            record = _SUMMARY._append_controller_gate_observation(
                self._ledger_path,
                artifact_root=self._artifact_root,
                contract_sha256=self._contract_sha256,
                environment_sha256=self._environment_sha256,
                run_id=self._run_id,
                ledger_id=self._ledger_id,
                observation_id=observation,
                artifact=artifact,
                adapter_implementation_sha256=spec["implementation_sha256"],
                adapter_request_sha256=request_sha,
                as_of=recorded_at,
                max_age_seconds=1.0,
                controller_seal_key=self._seal_key,
                expected_previous_sha256=expected_previous_sha256,
            )
        except BaseException:
            try:
                committed = self._find_existing_observation(
                    observation_id=observation,
                )
            except BaseException:
                # The append outcome is ambiguous. Keep the artifact so a
                # possibly committed ledger reference never becomes dangling.
                raise
            if committed is None:
                raise
            try:
                reused = self._reuse_existing(
                    committed,
                    adapter_request_sha256=request_sha,
                    implementation_sha256=spec["implementation_sha256"],
                )
            except BaseException:
                raise
            return reused
        return {"record": record, "artifact_path": str(artifact_path)}
