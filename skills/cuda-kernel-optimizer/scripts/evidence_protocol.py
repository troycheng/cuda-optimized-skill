#!/usr/bin/env python3
"""Pure, fail-closed validators for V2.5 formal evidence artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from experiment_design import validate_frozen_design  # noqa: E402


_PHASES = ("correctness", "sanitizer", "diagnostic", "timing")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SERVING_METRICS = {
    "qps",
    "avg_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "server_input_ms",
    "server_infer_ms",
    "server_output_ms",
}
_TERMINAL_STATES = {
    "valid",
    "invalid_contaminated",
    "invalid_identity",
    "partial",
    "superseded",
}
_BASE_ARTIFACT_KINDS = {
    "runner",
    "guard",
    "analysis",
    "schedule",
    "source",
    "diagnostic_binary",
    "binary",
    "experiment_design",
    "guard_policy",
    "guard_samples",
    "phase_markers",
    "execution_path",
    "raw_rows",
    "performance_verdict",
}
_SERVING_ARTIFACT_KINDS = {
    "serving_experiment",
    "artifact_identities",
    "plugin",
    "engine",
    "backend",
    "server",
    "image",
}


def _closed(value, *, field: str, keys: set[str]) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    actual = set(value)
    missing = sorted(keys - actual)
    unknown = sorted(actual - keys)
    if missing or unknown:
        raise ValueError(f"{field} has missing={missing} unknown={unknown}")
    return dict(value)


def _string(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _finite(value, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be finite")
    parsed = float(value)
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return parsed


def _literal_int(value, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _string_list(value, field: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ValueError(f"{field} must be an array of strings")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{field} must be an array of non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} must contain unique values")
    return list(value)


def _sha256(value, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _validate_guardrail_set(value, field: str) -> None:
    guardrails = _closed(value, field=field, keys={"relative", "absolute"})
    relative = guardrails["relative"]
    absolute = guardrails["absolute"]
    if not isinstance(relative, list) or not relative:
        raise ValueError(f"{field}.relative must be non-empty")
    if not isinstance(absolute, list) or not absolute:
        raise ValueError(f"{field}.absolute must be non-empty")
    for index, item in enumerate(relative):
        row = _closed(
            item,
            field=f"{field}.relative[{index}]",
            keys={"metric", "comparison", "direction", "limit_pct"},
        )
        _string(row["metric"], f"{field}.relative[{index}].metric")
        if row["comparison"] not in {"min_improvement", "max_regression"}:
            raise ValueError(f"{field}.relative[{index}].comparison is invalid")
        if row["direction"] not in {"lower", "higher"}:
            raise ValueError(f"{field}.relative[{index}].direction is invalid")
        _finite(row["limit_pct"], f"{field}.relative[{index}].limit_pct", minimum=0)
    for index, item in enumerate(absolute):
        row = _closed(
            item,
            field=f"{field}.absolute[{index}]",
            keys={"metric", "operator", "limit"},
        )
        _string(row["metric"], f"{field}.absolute[{index}].metric")
        if row["operator"] not in {"<=", ">="}:
            raise ValueError(f"{field}.absolute[{index}].operator is invalid")
        _finite(row["limit"], f"{field}.absolute[{index}].limit")


def _identity(value, field: str) -> dict:
    identity = _closed(value, field=field, keys={"uuid", "pci_bus_id"})
    _string(identity["uuid"], f"{field}.uuid")
    _string(identity["pci_bus_id"], f"{field}.pci_bus_id")
    return identity


def _validate_guard_policy(value) -> dict:
    keys = {
        "schema_version",
        "formal",
        "sample_interval_ms",
        "max_sample_gap_ms",
        "joint_clean_window_ms",
        "gpus",
        "cpu",
        "allowlist",
        "limits",
        "phase_requirements",
        "not_applicable_reasons",
    }
    policy = _closed(value, field="guard_policy", keys=keys)
    if policy["schema_version"] != "cuda-evidence/guard-policy-v1":
        raise ValueError("guard_policy.schema_version is unsupported")
    if policy["formal"] is not True:
        raise ValueError("guard_policy.formal must be true")
    interval = _finite(policy["sample_interval_ms"], "sample_interval_ms", minimum=1)
    gap = _finite(policy["max_sample_gap_ms"], "max_sample_gap_ms", minimum=1)
    window = _finite(policy["joint_clean_window_ms"], "joint_clean_window_ms", minimum=interval)
    if gap < interval:
        raise ValueError("max_sample_gap_ms cannot be below sample_interval_ms")

    gpus = _closed(policy["gpus"], field="gpus", keys={"target", "peers", "siblings"})
    identities = [("target", _identity(gpus["target"], "gpus.target"))]
    for group, role in (("peers", "peer"), ("siblings", "sibling")):
        rows = gpus[group]
        if not isinstance(rows, list):
            raise ValueError(f"gpus.{group} must be an array")
        identities.extend(
            (role, _identity(item, f"gpus.{group}[{index}]"))
            for index, item in enumerate(rows)
        )
    uuid_values = [item["uuid"] for _, item in identities]
    pci_values = [item["pci_bus_id"] for _, item in identities]
    if len(set(uuid_values)) != len(uuid_values) or len(set(pci_values)) != len(pci_values):
        raise ValueError("guard GPU identities must be unique")

    cpu = _closed(policy["cpu"], field="cpu", keys={"cpus", "numa_nodes"})
    for field in ("cpus", "numa_nodes"):
        values = cpu[field]
        if not isinstance(values, list) or not values:
            raise ValueError(f"cpu.{field} must be a non-empty array")
        normalized = [_literal_int(item, f"cpu.{field}") for item in values]
        if normalized != sorted(set(normalized)):
            raise ValueError(f"cpu.{field} must be sorted and unique")

    allowlist = _closed(
        policy["allowlist"], field="allowlist", keys={"pids", "containers"}
    )
    pids = allowlist["pids"]
    if not isinstance(pids, list):
        raise ValueError("allowlist.pids must be an array")
    normalized_pids = [_literal_int(pid, "allowlist.pids", minimum=1) for pid in pids]
    if len(set(normalized_pids)) != len(normalized_pids):
        raise ValueError("allowlist.pids must be unique")
    _string_list(allowlist["containers"], "allowlist.containers")

    limit_keys = {
        "min_sm_clock_mhz",
        "max_temperature_c",
        "max_power_w",
        "forbidden_throttle_reasons",
        "max_swap_used_bytes",
        "max_memory_pressure_pct",
        "max_foreign_cpu_pct",
        "max_foreign_gpu_pct",
    }
    limits = _closed(policy["limits"], field="limits", keys=limit_keys)
    for field in limit_keys - {"forbidden_throttle_reasons"}:
        _finite(limits[field], f"limits.{field}", minimum=0)
    _string_list(
        limits["forbidden_throttle_reasons"],
        "limits.forbidden_throttle_reasons",
    )

    requirements = _closed(
        policy["phase_requirements"],
        field="phase_requirements",
        keys=set(_PHASES),
    )
    for phase in _PHASES:
        if requirements[phase] not in {"required", "not_applicable"}:
            raise ValueError(f"phase_requirements.{phase} is invalid")
    if requirements["timing"] != "required":
        raise ValueError("formal timing phase must be required")
    reasons = policy["not_applicable_reasons"]
    if not isinstance(reasons, Mapping):
        raise ValueError("not_applicable_reasons must be an object")
    expected_reason_keys = {
        phase for phase in _PHASES if requirements[phase] == "not_applicable"
    }
    if set(reasons) != expected_reason_keys:
        raise ValueError("not_applicable_reasons must match not_applicable phases")
    for phase in expected_reason_keys:
        _string(reasons[phase], f"not_applicable_reasons.{phase}")

    normalized = copy.deepcopy(policy)
    normalized["_validated_identities"] = copy.deepcopy(identities)
    normalized["_sample_interval"] = interval
    normalized["_max_gap"] = gap
    normalized["_clean_window"] = window
    return normalized


def _sample_reasons(sample, index: int, policy: dict) -> tuple[float | None, list[str]]:
    prefix = f"sample[{index}]"
    reasons: list[str] = []
    keys = {"monotonic_ms", "gpus", "cpu", "memory", "contamination_markers"}
    try:
        row = _closed(sample, field=prefix, keys=keys)
    except ValueError as error:
        return None, [f"{prefix}.unknown:{error}"]
    try:
        timestamp = _finite(row["monotonic_ms"], f"{prefix}.monotonic_ms", minimum=0)
    except ValueError:
        timestamp = None
        reasons.append(f"{prefix}.monotonic_ms_unknown")

    markers = row["contamination_markers"]
    try:
        marker_values = _string_list(markers, f"{prefix}.contamination_markers")
    except ValueError:
        marker_values = []
        reasons.append(f"{prefix}.contamination_marker_unknown")
    if marker_values:
        reasons.append(f"{prefix}.contamination_marker")

    expected_identities = policy["_validated_identities"]
    gpu_rows = row["gpus"]
    gpu_keys = {
        "role",
        "uuid",
        "pci_bus_id",
        "sm_clock_mhz",
        "temperature_c",
        "power_w",
        "throttle_reasons",
        "foreign_gpu_pct",
        "processes",
    }
    observed: dict[str, dict] = {}
    if not isinstance(gpu_rows, list):
        reasons.append(f"{prefix}.gpu_identity_unknown")
        gpu_rows = []
    for gpu_index, gpu in enumerate(gpu_rows):
        field = f"{prefix}.gpus[{gpu_index}]"
        try:
            item = _closed(gpu, field=field, keys=gpu_keys)
            uuid = _string(item["uuid"], f"{field}.uuid")
            if uuid in observed:
                reasons.append(f"{field}.identity_duplicate")
            observed[uuid] = item
        except ValueError:
            reasons.append(f"{field}.identity_unknown")
    if len(observed) != len(expected_identities):
        reasons.append(f"{prefix}.gpu_identity_count")

    limits = policy["limits"]
    allowed_pids = set(policy["allowlist"]["pids"])
    allowed_containers = set(policy["allowlist"]["containers"])
    forbidden_throttles = set(limits["forbidden_throttle_reasons"])
    for role, identity in expected_identities:
        item = observed.get(identity["uuid"])
        if item is None:
            reasons.append(f"{prefix}.gpu_identity_missing:{identity['uuid']}")
            continue
        if item.get("role") != role or item.get("pci_bus_id") != identity["pci_bus_id"]:
            reasons.append(f"{prefix}.gpu_identity_mismatch:{identity['uuid']}")
        numeric_checks = (
            ("sm_clock_mhz", "clock", lambda value: value < limits["min_sm_clock_mhz"]),
            ("temperature_c", "temperature", lambda value: value > limits["max_temperature_c"]),
            ("power_w", "power", lambda value: value > limits["max_power_w"]),
            ("foreign_gpu_pct", "foreign_gpu", lambda value: value > limits["max_foreign_gpu_pct"]),
        )
        for field, reason, violates in numeric_checks:
            try:
                parsed = _finite(item[field], f"{prefix}.{field}", minimum=0)
            except (KeyError, ValueError):
                reasons.append(f"{prefix}.{field}_unknown")
                continue
            if violates(parsed):
                reasons.append(f"{prefix}.{reason}_limit")
        try:
            throttles = set(_string_list(item["throttle_reasons"], f"{prefix}.throttle_reasons"))
        except (KeyError, ValueError):
            throttles = set()
            reasons.append(f"{prefix}.throttle_unknown")
        if throttles & forbidden_throttles:
            reasons.append(f"{prefix}.throttle_forbidden")
        processes = item.get("processes")
        if not isinstance(processes, list):
            reasons.append(f"{prefix}.foreign_process_unknown")
            processes = []
        for process_index, process in enumerate(processes):
            try:
                actor = _closed(
                    process,
                    field=f"{prefix}.processes[{process_index}]",
                    keys={"pid", "container_id"},
                )
                pid = _literal_int(actor["pid"], "process.pid", minimum=1)
                container = _string(actor["container_id"], "process.container_id")
            except ValueError:
                reasons.append(f"{prefix}.foreign_process_unknown")
                continue
            if pid not in allowed_pids and container not in allowed_containers:
                reasons.append(f"{prefix}.foreign_process:{pid}")

    try:
        cpu = _closed(
            row["cpu"],
            field=f"{prefix}.cpu",
            keys={"cpus", "numa_nodes", "foreign_cpu_pct"},
        )
        if cpu["cpus"] != policy["cpu"]["cpus"]:
            reasons.append(f"{prefix}.cpu_affinity_drift")
        if cpu["numa_nodes"] != policy["cpu"]["numa_nodes"]:
            reasons.append(f"{prefix}.numa_drift")
        foreign_cpu = _finite(cpu["foreign_cpu_pct"], "foreign_cpu_pct", minimum=0)
        if foreign_cpu > limits["max_foreign_cpu_pct"]:
            reasons.append(f"{prefix}.foreign_cpu_limit")
    except (KeyError, ValueError):
        reasons.append(f"{prefix}.cpu_unknown")

    try:
        memory = _closed(
            row["memory"],
            field=f"{prefix}.memory",
            keys={"swap_used_bytes", "pressure_pct"},
        )
        swap = _finite(memory["swap_used_bytes"], "swap_used_bytes", minimum=0)
        pressure = _finite(memory["pressure_pct"], "pressure_pct", minimum=0)
        if swap > limits["max_swap_used_bytes"]:
            reasons.append(f"{prefix}.swap_limit")
        if pressure > limits["max_memory_pressure_pct"]:
            reasons.append(f"{prefix}.memory_pressure_limit")
    except KeyError as error:
        reasons.append(f"{prefix}.memory.{error.args[0]}_missing")
    except ValueError as error:
        text = str(error)
        field = "pressure_pct" if "pressure_pct" in text else "memory"
        reasons.append(f"{prefix}.memory.{field}_unknown")

    return timestamp, reasons


def audit_shared_host_guard(policy, samples, phase_markers) -> dict:
    """Audit continuous, phase-bounded normalized telemetry.

    The function deliberately does not collect host data. Site adapters own
    sampling; this validator owns the formal fail-closed decision.
    """
    normalized = _validate_guard_policy(policy)
    if isinstance(samples, (str, bytes, bytearray, Mapping)) or not isinstance(samples, Sequence):
        raise ValueError("guard samples must be an array")
    if isinstance(phase_markers, (str, bytes, bytearray, Mapping)) or not isinstance(phase_markers, Sequence):
        raise ValueError("phase markers must be an array")

    reasons: list[str] = []
    timestamps: list[float] = []
    for index, sample in enumerate(samples):
        timestamp, sample_failures = _sample_reasons(sample, index, normalized)
        reasons.extend(sample_failures)
        if timestamp is not None:
            timestamps.append(timestamp)
    if not timestamps:
        reasons.append("samples_missing")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        reasons.append("sample_order_invalid")

    marker_map: dict[str, dict] = {}
    marker_keys = {"phase", "watcher_ready_ms", "start_ms", "end_ms"}
    for index, marker in enumerate(phase_markers):
        if not isinstance(marker, Mapping):
            reasons.append(f"phase_marker[{index}].unknown")
            continue
        phase = marker.get("phase")
        if phase not in _PHASES or phase in marker_map:
            reasons.append(f"phase_marker[{index}].phase_invalid")
            continue
        missing = marker_keys - set(marker)
        unknown = set(marker) - marker_keys
        if missing or unknown:
            for field in sorted(missing):
                reasons.append(f"phase.{phase}.{field.replace('_ms', '')}_missing")
            if unknown:
                reasons.append(f"phase.{phase}.unknown_fields")
            continue
        marker_map[phase] = dict(marker)

    phase_results = {}
    max_gap = normalized["_max_gap"]
    clean_window = normalized["_clean_window"]
    for phase in _PHASES:
        requirement = normalized["phase_requirements"][phase]
        if requirement == "not_applicable":
            phase_results[phase] = {
                "status": "not_applicable",
                "reason": normalized["not_applicable_reasons"][phase],
            }
            continue
        phase_reasons = []
        marker = marker_map.get(phase)
        if marker is None:
            if not any(item.startswith(f"phase.{phase}.") for item in reasons):
                phase_reasons.append(f"phase.{phase}.marker_missing")
        else:
            try:
                ready = _finite(marker["watcher_ready_ms"], "watcher_ready_ms", minimum=0)
                start = _finite(marker["start_ms"], "start_ms", minimum=0)
                end = _finite(marker["end_ms"], "end_ms", minimum=0)
                if not ready <= start <= end:
                    phase_reasons.append(f"phase.{phase}.watcher_ready_order")
                clean_start = ready - clean_window
                before = [item for item in timestamps if item <= clean_start]
                after = [item for item in timestamps if item >= end]
                if clean_start < 0 or not before:
                    phase_reasons.append(f"phase.{phase}.joint_clean_window")
                if not after:
                    phase_reasons.append(f"phase.{phase}.coverage_end")
                if before and after:
                    lower = before[-1]
                    upper = after[0]
                    covered = [item for item in timestamps if lower <= item <= upper]
                    if clean_start - lower > max_gap or upper - end > max_gap:
                        phase_reasons.append(f"phase.{phase}.boundary_gap")
                    if any(
                        right - left > max_gap
                        for left, right in zip(covered, covered[1:])
                    ):
                        phase_reasons.append(f"phase.{phase}.sample_gap")
            except (KeyError, ValueError):
                phase_reasons.append(f"phase.{phase}.watcher_ready_unknown")
        reasons.extend(phase_reasons)
        phase_results[phase] = {
            "status": "PASS" if not phase_reasons else "FAIL",
            "reasons": phase_reasons,
        }

    unique_reasons = list(dict.fromkeys(reasons))
    status = "PASS" if not unique_reasons else "FAIL"
    return {
        "schema_version": "cuda-evidence/guard-audit-v1",
        "status": status,
        "evidence_integrity": status,
        "formal": True,
        "sample_count": len(samples),
        "phase_results": phase_results,
        "reasons": unique_reasons,
    }


def _binary_identity(value, field: str) -> dict:
    row = _closed(
        value,
        field=field,
        keys={"sha256", "source_sha256", "build_config_sha256", "diagnostic_features"},
    )
    for key in ("sha256", "source_sha256", "build_config_sha256"):
        _sha256(row[key], f"{field}.{key}")
    _string_list(row["diagnostic_features"], f"{field}.diagnostic_features")
    return row


def validate_execution_path(value) -> dict:
    """Require real-case dispatch hits and a rebuilt diagnostic-free timed binary."""
    keys = {
        "schema_version",
        "expected_cases",
        "case_hits",
        "proof_kind",
        "trace_sha256",
        "diagnostic_binary",
        "timed_binary",
        "rebuilt_after_diagnostics",
        "timed_binary_bound",
    }
    proof = _closed(value, field="execution_path", keys=keys)
    if proof["schema_version"] != "cuda-evidence/execution-path-v1":
        raise ValueError("execution_path.schema_version is unsupported")
    expected = _string_list(proof["expected_cases"], "expected_cases", allow_empty=False)
    if proof["proof_kind"] not in {"dispatch_counter", "trace", "topology"}:
        raise ValueError("execution_path.proof_kind is invalid")
    _sha256(proof["trace_sha256"], "execution_path.trace_sha256")

    hits = proof["case_hits"]
    if not isinstance(hits, list):
        raise ValueError("case_hits must be an array")
    hit_map = {}
    for index, item in enumerate(hits):
        row = _closed(
            item, field=f"case_hits[{index}]", keys={"case_id", "hit_count"}
        )
        case_id = _string(row["case_id"], f"case_hits[{index}].case_id")
        if case_id in hit_map:
            raise ValueError("case_hits case_id values must be unique")
        hit_map[case_id] = _literal_int(
            row["hit_count"], f"case_hits[{index}].hit_count", minimum=1
        )
    if set(hit_map) != set(expected):
        raise ValueError("case_hits must cover exactly every expected case")

    diagnostic = _binary_identity(proof["diagnostic_binary"], "diagnostic_binary")
    timed = _binary_identity(proof["timed_binary"], "timed_binary")
    if not diagnostic["diagnostic_features"]:
        raise ValueError("diagnostic_binary must declare its diagnostic features")
    if timed["diagnostic_features"]:
        raise ValueError("timed_binary must contain no diagnostic features")
    if timed["sha256"] == diagnostic["sha256"]:
        raise ValueError("diagnostic binary cannot be used for formal timing")
    if timed["source_sha256"] != diagnostic["source_sha256"]:
        raise ValueError("timed and diagnostic binaries must bind the same source")
    if timed["build_config_sha256"] != diagnostic["build_config_sha256"]:
        raise ValueError("timed and diagnostic binaries must bind the same build config")
    if proof["rebuilt_after_diagnostics"] is not True:
        raise ValueError("timed binary must be rebuilt after diagnostic removal")
    if proof["timed_binary_bound"] is not True:
        raise ValueError("timed binary must be bound to formal measurements")
    return {
        "schema_version": "cuda-evidence/execution-path-audit-v1",
        "status": "PASS",
        "proof_kind": proof["proof_kind"],
        "expected_cases": copy.deepcopy(expected),
        "case_hits": copy.deepcopy(hit_map),
        "timed_binary_sha256": timed["sha256"],
    }


def validate_serving_experiment(value) -> dict:
    """Validate the frozen HTTP/gRPC c1/c2/c4/c8/c12 serving matrix."""
    plan = _closed(
        value,
        field="serving_experiment",
        keys={
            "schema_version",
            "protocols",
            "fresh_process_per_role",
            "request_corpus_sha256",
            "strata",
        },
    )
    if plan["schema_version"] != "cuda-evidence/serving-experiment-v1":
        raise ValueError("serving_experiment.schema_version is unsupported")
    protocols = _string_list(plan["protocols"], "protocols", allow_empty=False)
    if not set(protocols) <= {"http", "grpc"}:
        raise ValueError("protocols may contain only http and grpc")
    if plan["fresh_process_per_role"] is not True:
        raise ValueError("serving formal roles require fresh processes")
    _sha256(plan["request_corpus_sha256"], "request_corpus_sha256")
    strata = plan["strata"]
    if not isinstance(strata, list):
        raise ValueError("strata must be an array")
    expected = {(protocol, c) for protocol in protocols for c in (1, 2, 4, 8, 12)}
    observed = set()
    stratum_keys = {
        "id",
        "protocol",
        "concurrency",
        "warmup_requests",
        "measured_requests",
        "metrics",
        "must_pass",
    }
    for index, item in enumerate(strata):
        row = _closed(item, field=f"strata[{index}]", keys=stratum_keys)
        protocol = row["protocol"]
        concurrency = row["concurrency"]
        if (protocol, concurrency) not in expected or (protocol, concurrency) in observed:
            raise ValueError("strata must contain each requested protocol/concurrency once")
        observed.add((protocol, concurrency))
        if row["id"] != f"{protocol}-c{concurrency}":
            raise ValueError("stratum id must bind protocol and concurrency")
        _literal_int(row["warmup_requests"], "warmup_requests", minimum=1)
        _literal_int(row["measured_requests"], "measured_requests", minimum=1)
        metrics = _string_list(row["metrics"], f"strata[{index}].metrics", allow_empty=False)
        if set(metrics) != _SERVING_METRICS:
            raise ValueError("each stratum must declare the standard serving metrics")
        _validate_guardrail_set(row["must_pass"], f"strata[{index}].must_pass")
    if observed != expected:
        raise ValueError("serving experiment is missing required strata")
    return {
        "schema_version": "cuda-evidence/serving-experiment-audit-v1",
        "status": "PASS",
        "protocols": copy.deepcopy(protocols),
        "strata_count": len(strata),
        "request_corpus_sha256": plan["request_corpus_sha256"],
    }


def validate_artifact_identities(value) -> dict:
    """Validate a complete TensorRT/Triton serving-stack identity boundary."""
    identities = _closed(
        value,
        field="artifact_identities",
        keys={"schema_version", "source", "binary", "plugin", "engine", "backend", "server", "image"},
    )
    if identities["schema_version"] != "cuda-evidence/artifact-identities-v1":
        raise ValueError("artifact_identities.schema_version is unsupported")
    source = _closed(identities["source"], field="source", keys={"sha256"})
    source_sha = _sha256(source["sha256"], "source.sha256")
    binary = _closed(
        identities["binary"],
        field="binary",
        keys={"sha256", "source_sha256", "build_config_sha256"},
    )
    binary_sha = _sha256(binary["sha256"], "binary.sha256")
    if _sha256(binary["source_sha256"], "binary.source_sha256") != source_sha:
        raise ValueError("binary must bind the declared source")
    _sha256(binary["build_config_sha256"], "binary.build_config_sha256")

    plugin = _closed(
        identities["plugin"],
        field="plugin",
        keys={"sha256", "source_sha256", "compiler_version", "abi"},
    )
    plugin_sha = _sha256(plugin["sha256"], "plugin.sha256")
    _sha256(plugin["source_sha256"], "plugin.source_sha256")
    _string(plugin["compiler_version"], "plugin.compiler_version")
    _string(plugin["abi"], "plugin.abi")

    engine = _closed(
        identities["engine"],
        field="engine",
        keys={
            "sha256",
            "plugin_sha256",
            "builder_version",
            "runtime_version",
            "tactic_digest",
            "timing_cache_digest",
        },
    )
    _sha256(engine["sha256"], "engine.sha256")
    if _sha256(engine["plugin_sha256"], "engine.plugin_sha256") != plugin_sha:
        raise ValueError("engine must bind the declared plugin")
    for field in ("tactic_digest", "timing_cache_digest"):
        _sha256(engine[field], f"engine.{field}")
    for field in ("builder_version", "runtime_version"):
        _string(engine[field], f"engine.{field}")

    for kind in ("backend", "server"):
        row = _closed(
            identities[kind], field=kind, keys={"sha256", "version", "abi"}
        )
        _sha256(row["sha256"], f"{kind}.sha256")
        _string(row["version"], f"{kind}.version")
        _string(row["abi"], f"{kind}.abi")
    image = _closed(identities["image"], field="image", keys={"digest", "tag"})
    digest = image["digest"]
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ValueError("image.digest must be immutable sha256, not a tag")
    _sha256(digest.removeprefix("sha256:"), "image.digest")
    _string(image["tag"], "image.tag")

    return {
        "schema_version": "cuda-evidence/artifact-identities-audit-v1",
        "status": "PASS",
        "source_sha256": source_sha,
        "binary_sha256": binary_sha,
        "plugin_sha256": plugin_sha,
        "engine_sha256": engine["sha256"],
        "image_digest": digest,
    }


def validate_profiler_bundle(value) -> dict:
    """Validate an Nsys/NCU explanation bundle with non-promotional authority."""
    bundle = _closed(
        value,
        field="profiler_bundle",
        keys={
            "schema_version",
            "authority",
            "target_binary_sha256",
            "reports",
            "observations",
            "limitations",
        },
    )
    if bundle["schema_version"] != "cuda-evidence/profiler-bundle-v1":
        raise ValueError("profiler_bundle.schema_version is unsupported")
    if bundle["authority"] != "non_promotional":
        raise ValueError("profiler bundles are always non_promotional")
    target = _sha256(bundle["target_binary_sha256"], "target_binary_sha256")
    reports = bundle["reports"]
    if not isinstance(reports, list):
        raise ValueError("reports must be an array")
    tools = set()
    report_keys = {
        "tool",
        "version",
        "status",
        "report_sha256",
        "command",
        "target_binary_sha256",
    }
    for index, item in enumerate(reports):
        row = _closed(item, field=f"reports[{index}]", keys=report_keys)
        tool = row["tool"]
        if tool not in {"nsys", "ncu"} or tool in tools:
            raise ValueError("reports must contain unique nsys and ncu records")
        tools.add(tool)
        _string(row["version"], f"reports[{index}].version")
        if row["status"] not in {"available", "unavailable"}:
            raise ValueError("profiler report status is invalid")
        if row["status"] == "available":
            _sha256(row["report_sha256"], f"reports[{index}].report_sha256")
        elif row["report_sha256"] is not None:
            raise ValueError("unavailable profiler report hash must be null")
        command = row["command"]
        if not isinstance(command, list) or not command:
            raise ValueError("profiler command must be a non-empty argv array")
        _string_list(command, f"reports[{index}].command", allow_empty=False)
        if _sha256(row["target_binary_sha256"], "report target") != target:
            raise ValueError("profiler report must bind the timed binary")
    if tools != {"nsys", "ncu"}:
        raise ValueError("profiler bundle must cover both Nsys and NCU")
    _string_list(bundle["observations"], "observations")
    _string_list(bundle["limitations"], "limitations", allow_empty=False)
    return {
        "schema_version": "cuda-evidence/profiler-bundle-audit-v1",
        "status": "PASS",
        "authority": "non_promotional",
        "target_binary_sha256": target,
        "tools": sorted(tools),
    }


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_value_strict(path: Path | str):
    file_path = Path(path)
    try:
        raw = file_path.read_text(encoding="utf-8")
        payload = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON file {file_path}: {error}") from error
    return payload


def load_json_strict(path: Path | str) -> dict:
    file_path = Path(path)
    payload = _load_json_value_strict(file_path)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {file_path}")
    return payload


def _load_jsonl_strict(path: Path | str) -> list[dict]:
    file_path = Path(path)
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"invalid JSONL file {file_path}: {error}") from error
    rows = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid JSONL line {line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"JSONL line {line_number} must be an object")
        rows.append(row)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_regular(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("artifact path must be a non-empty relative path")
    relative_path = Path(relative)
    if any(part in {"", ".", ".."} for part in relative_path.parts):
        raise ValueError("artifact path contains an unsafe component")
    root = root.resolve(strict=True)
    current = root
    for part in relative_path.parts:
        current = current / part
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise ValueError(f"artifact is missing: {relative}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"artifact path must not contain symlinks: {relative}")
    if not stat.S_ISREG(os.lstat(current).st_mode):
        raise ValueError(f"artifact must be a regular file: {relative}")
    try:
        current.relative_to(root)
    except ValueError as error:
        raise ValueError(f"artifact escapes attempt root: {relative}") from error
    return current


def _write_json_create_once(path: Path | str, payload: dict, *, immutable: bool = True) -> None:
    output = Path(path)
    parent = output.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError(f"output parent must be a real directory: {parent}")
    data = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = None
    try:
        descriptor = os.open(output, flags, 0o600)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if immutable:
            os.chmod(output, 0o400)
        directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError:
        raise
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        try:
            output.unlink()
        except OSError:
            pass
        raise


def _artifact_index(attempt: dict, root: Path) -> tuple[dict[str, dict], list[dict]]:
    artifacts = attempt["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("attempt.artifacts must be a non-empty array")
    by_kind = {}
    ids = set()
    paths = set()
    sealed = []
    for index, item in enumerate(artifacts):
        row = _closed(
            item, field=f"artifacts[{index}]", keys={"id", "kind", "path"}
        )
        artifact_id = _string(row["id"], f"artifacts[{index}].id")
        kind = _string(row["kind"], f"artifacts[{index}].kind")
        relative = _string(row["path"], f"artifacts[{index}].path")
        if artifact_id in ids or kind in by_kind or relative in paths:
            raise ValueError("artifact ids, kinds, and paths must be unique")
        ids.add(artifact_id)
        paths.add(relative)
        file_path = _safe_regular(root, relative)
        record = {
            "id": artifact_id,
            "kind": kind,
            "path": relative,
            "sha256": _sha256_file(file_path),
            "size_bytes": file_path.stat().st_size,
        }
        by_kind[kind] = {**row, "file_path": file_path, "record": record}
        sealed.append(record)
    return by_kind, sealed


def _validate_raw_rows(rows: list[dict], design: dict) -> dict:
    expected = {row["pair_id"]: row["order"] for row in design["schedule"]}
    observed = {}
    for index, item in enumerate(rows):
        row = _closed(
            item,
            field=f"raw_rows[{index}]",
            keys={"pair_id", "order", "valid", "attempts"},
        )
        pair_id = _string(row["pair_id"], f"raw_rows[{index}].pair_id")
        if pair_id in observed or pair_id not in expected:
            raise ValueError("raw rows must contain each scheduled pair exactly once")
        if row["order"] != expected[pair_id]:
            raise ValueError("raw row order differs from frozen schedule")
        if type(row["valid"]) is not bool:
            raise ValueError("raw row valid must be a bool")
        attempts = _closed(
            row["attempts"],
            field=f"raw_rows[{index}].attempts",
            keys={"baseline", "candidate"},
        )
        if attempts["baseline"] != 1 or attempts["candidate"] != 1:
            raise ValueError("formal timed work forbids single-role retries")
        observed[pair_id] = row["order"]
    if observed != expected:
        raise ValueError("no-exclusion requires every frozen schedule row")
    return {"status": "PASS", "rows": len(rows)}


def _validate_performance_verdict(value) -> dict:
    verdict = _closed(
        value,
        field="performance_verdict",
        keys={
            "schema_version",
            "status",
            "promotional_eligible",
            "analysis_sha256",
            "experiment_design_sha256",
            "raw_rows_sha256",
        },
    )
    if verdict["schema_version"] != "cuda-evidence/performance-verdict-v1":
        raise ValueError("performance verdict schema is unsupported")
    if verdict["status"] not in {
        "confirmed_win",
        "confirmed_loss",
        "inconclusive",
        "failed",
    }:
        raise ValueError("performance verdict status is invalid")
    if type(verdict["promotional_eligible"]) is not bool:
        raise ValueError("performance promotional_eligible must be a bool")
    if verdict["promotional_eligible"] and verdict["status"] != "confirmed_win":
        raise ValueError("only a confirmed win can be promotional_eligible")
    for field in (
        "analysis_sha256",
        "experiment_design_sha256",
        "raw_rows_sha256",
    ):
        _sha256(verdict[field], f"performance_verdict.{field}")
    return copy.deepcopy(verdict)


def _semantic_gate_results(by_kind: dict[str, dict]) -> tuple[dict, dict]:
    gates = {}
    errors = {}

    def run(name, callback):
        try:
            result = callback()
            gates[name] = result if isinstance(result, dict) else {"status": "PASS"}
        except Exception as error:
            gates[name] = {"status": "FAIL", "reason": str(error)[:512]}
            errors[name] = str(error)

    design_holder = {}

    def experiment():
        design = validate_frozen_design(load_json_strict(by_kind["experiment_design"]["file_path"]))
        design_holder["value"] = design
        return {"status": "PASS", "pairs": len(design["schedule"])}

    run("experiment_design", experiment)

    if "value" in design_holder and "schedule" in by_kind:
        def schedule_binding():
            schedule = _load_json_value_strict(by_kind["schedule"]["file_path"])
            if schedule != design_holder["value"]["schedule"]:
                raise ValueError("schedule artifact differs from frozen experiment design")
            return {"status": "PASS", "pairs": len(schedule)}

        run("schedule_binding", schedule_binding)
    run(
        "guard",
        lambda: audit_shared_host_guard(
            load_json_strict(by_kind["guard_policy"]["file_path"]),
            _load_jsonl_strict(by_kind["guard_samples"]["file_path"]),
            _load_json_value_strict(by_kind["phase_markers"]["file_path"]),
        ),
    )
    def execution_path():
        proof = load_json_strict(by_kind["execution_path"]["file_path"])
        result = validate_execution_path(proof)
        bindings = (
            ("source", proof["timed_binary"]["source_sha256"]),
            ("diagnostic_binary", proof["diagnostic_binary"]["sha256"]),
            ("binary", proof["timed_binary"]["sha256"]),
        )
        for kind, declared_sha in bindings:
            if by_kind[kind]["record"]["sha256"] != declared_sha:
                raise ValueError(f"execution-path {kind} digest differs from sealed artifact")
        return result

    run("execution_path", execution_path)
    if "serving_experiment" in by_kind:
        run(
            "serving_experiment",
            lambda: validate_serving_experiment(
                load_json_strict(by_kind["serving_experiment"]["file_path"])
            ),
        )
    if "artifact_identities" in by_kind:
        def artifact_identities():
            identities = load_json_strict(by_kind["artifact_identities"]["file_path"])
            result = validate_artifact_identities(identities)
            for kind in ("source", "binary", "plugin", "engine", "backend", "server"):
                if by_kind[kind]["record"]["sha256"] != identities[kind]["sha256"]:
                    raise ValueError(f"{kind} identity differs from sealed artifact")
            try:
                image_digest = by_kind["image"]["file_path"].read_text(
                    encoding="utf-8"
                ).strip()
            except (OSError, UnicodeError) as error:
                raise ValueError("image digest artifact is unreadable") from error
            if image_digest != identities["image"]["digest"]:
                raise ValueError("image identity differs from sealed digest artifact")
            return result

        run(
            "artifact_identities",
            artifact_identities,
        )
    if "profiler_bundle" in by_kind:
        def profiler_bundle():
            bundle = load_json_strict(by_kind["profiler_bundle"]["file_path"])
            result = validate_profiler_bundle(bundle)
            if bundle["target_binary_sha256"] != by_kind["binary"]["record"]["sha256"]:
                raise ValueError("profiler target differs from sealed timed binary")
            for report in bundle["reports"]:
                artifact_kind = f"profiler_{report['tool']}"
                if report["status"] == "available":
                    if artifact_kind not in by_kind:
                        raise ValueError(f"available {report['tool']} report is not sealed")
                    if by_kind[artifact_kind]["record"]["sha256"] != report["report_sha256"]:
                        raise ValueError(
                            f"{report['tool']} report identity differs from sealed artifact"
                        )
                elif artifact_kind in by_kind:
                    raise ValueError(
                        f"unavailable {report['tool']} report must not have a sealed artifact"
                    )
            return result

        run(
            "profiler_bundle",
            profiler_bundle,
        )
    if "value" in design_holder:
        run(
            "raw_rows",
            lambda: _validate_raw_rows(
                _load_jsonl_strict(by_kind["raw_rows"]["file_path"]),
                design_holder["value"],
            ),
        )
    def performance_verdict():
        verdict = _validate_performance_verdict(
            load_json_strict(by_kind["performance_verdict"]["file_path"])
        )
        bindings = {
            "analysis_sha256": "analysis",
            "experiment_design_sha256": "experiment_design",
            "raw_rows_sha256": "raw_rows",
        }
        for field, kind in bindings.items():
            if verdict[field] != by_kind[kind]["record"]["sha256"]:
                raise ValueError(f"performance verdict {field} differs from sealed artifact")
        return {
            "status": "PASS",
            "verdict": verdict["status"],
        }

    run("performance_verdict", performance_verdict)
    return gates, errors


def recompute_sealed_semantics(seal: Mapping, by_kind: dict[str, dict]) -> dict:
    """Recompute V2.5 semantic gates and promotion eligibility without writes."""
    gates, errors = _semantic_gate_results(by_kind)
    verdict_status = "unknown"
    promotional_eligible = False
    performance = by_kind.get("performance_verdict")
    if performance is not None:
        try:
            verdict = _validate_performance_verdict(
                load_json_strict(performance["file_path"])
            )
            verdict_status = verdict["status"]
            promotional_eligible = verdict["promotional_eligible"]
        except Exception:
            pass
    required_gates = {
        "experiment_design",
        "schedule_binding",
        "guard",
        "execution_path",
        "raw_rows",
        "performance_verdict",
    }
    if seal.get("claim_layer") == "serving_endpoint":
        required_gates |= {"serving_experiment", "artifact_identities"}
    gates_pass = all(
        isinstance(gates.get(name), Mapping)
        and gates[name].get("status") == "PASS"
        for name in required_gates
    )
    return {
        "gate_results": gates,
        "gate_errors": errors,
        "performance_verdict": verdict_status,
        "promotional_eligible": promotional_eligible,
        "promotion_without_integrity": (
            seal.get("attempt_state") == "valid"
            and gates_pass
            and verdict_status == "confirmed_win"
            and promotional_eligible
        ),
    }


def seal_attempt(attempt_manifest_path: Path | str, output_path: Path | str) -> dict:
    """Create one immutable content seal for a terminal V2.5 attempt."""
    manifest_path = Path(attempt_manifest_path)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("attempt manifest must be a regular non-symlink file")
    root = manifest_path.parent.resolve(strict=True)
    seal_output = Path(output_path)
    if seal_output.parent.is_symlink() or seal_output.parent.resolve(strict=True) != root:
        raise ValueError("seal output must be in the attempt evidence directory")
    manifest = _closed(
        load_json_strict(manifest_path),
        field="attempt",
        keys={"schema_version", "attempt_id", "state", "claim_layer", "artifacts"},
    )
    if manifest["schema_version"] != "cuda-evidence/attempt-v1":
        raise ValueError("attempt schema is unsupported")
    attempt_id = _string(manifest["attempt_id"], "attempt_id")
    state = manifest["state"]
    if state not in _TERMINAL_STATES:
        raise ValueError("attempt state must be terminal")
    if manifest["claim_layer"] not in {"isolated_operator", "matched_runtime", "serving_endpoint"}:
        raise ValueError("attempt claim_layer is invalid")
    by_kind, sealed_artifacts = _artifact_index(manifest, root)
    required = set(_BASE_ARTIFACT_KINDS)
    if manifest["claim_layer"] == "serving_endpoint":
        required |= _SERVING_ARTIFACT_KINDS
    missing = sorted(required - set(by_kind))
    if state == "valid" and missing:
        raise ValueError(f"valid attempt is missing required evidence kinds: {missing}")
    gates, gate_errors = _semantic_gate_results(by_kind) if not missing else ({}, {"missing": missing})
    if state == "valid":
        failed = [name for name, result in gates.items() if result.get("status") != "PASS"]
        if gate_errors or failed:
            raise ValueError(f"valid attempt has failing evidence gates: {sorted(set(failed) | set(gate_errors))}")

    manifest_relative = manifest_path.name
    seal = {
        "schema_version": "cuda-evidence/seal-v1",
        "attempt_id": attempt_id,
        "attempt_state": state,
        "claim_layer": manifest["claim_layer"],
        "attempt_manifest": {
            "path": manifest_relative,
            "sha256": _sha256_file(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        },
        "artifacts": sealed_artifacts,
        "gate_results": gates,
        "gate_errors": gate_errors,
        "evidence_integrity": "not_audited",
    }
    _write_json_create_once(output_path, seal)
    return copy.deepcopy(seal)


def _compute_seal_audit(seal_file: Path) -> tuple[dict, dict]:
    """Return the current byte-integrity audit without writing an artifact."""
    root = seal_file.parent.resolve(strict=True)
    seal = load_json_strict(seal_file)
    reasons = []
    seal_keys = {
        "schema_version",
        "attempt_id",
        "attempt_state",
        "claim_layer",
        "attempt_manifest",
        "artifacts",
        "gate_results",
        "gate_errors",
        "evidence_integrity",
    }
    if set(seal) != seal_keys:
        reasons.append("seal_schema_invalid")
    if (
        seal.get("schema_version") != "cuda-evidence/seal-v1"
        or seal.get("attempt_state") not in _TERMINAL_STATES
        or seal.get("claim_layer")
        not in {"isolated_operator", "matched_runtime", "serving_endpoint"}
        or seal.get("evidence_integrity") != "not_audited"
        or not isinstance(seal.get("gate_results"), Mapping)
        or not isinstance(seal.get("gate_errors"), Mapping)
    ):
        reasons.append("seal_contract_invalid")
    manifest_ref = seal.get("attempt_manifest")
    if not isinstance(manifest_ref, Mapping):
        reasons.append("attempt_manifest_ref_invalid")
    else:
        if set(manifest_ref) != {"path", "sha256", "size_bytes"}:
            reasons.append("attempt_manifest_ref_invalid")
        try:
            _sha256(manifest_ref["sha256"], "attempt_manifest.sha256")
            _literal_int(
                manifest_ref["size_bytes"], "attempt_manifest.size_bytes"
            )
            path = _safe_regular(root, manifest_ref["path"])
            if _sha256_file(path) != manifest_ref["sha256"] or path.stat().st_size != manifest_ref["size_bytes"]:
                reasons.append("attempt_manifest_digest_mismatch")
            manifest = _closed(
                load_json_strict(path),
                field="attempt",
                keys={
                    "schema_version",
                    "attempt_id",
                    "state",
                    "claim_layer",
                    "artifacts",
                },
            )
            manifest_artifacts = manifest["artifacts"]
            sealed_artifacts = seal.get("artifacts")
            expected_refs = []
            if isinstance(manifest_artifacts, list):
                for item in manifest_artifacts:
                    if not isinstance(item, Mapping):
                        raise ValueError("attempt artifact ref is not an object")
                    expected_refs.append(
                        {key: item.get(key) for key in ("id", "kind", "path")}
                    )
            actual_refs = []
            if isinstance(sealed_artifacts, list):
                for item in sealed_artifacts:
                    if not isinstance(item, Mapping):
                        raise ValueError("sealed artifact ref is not an object")
                    actual_refs.append(
                        {key: item.get(key) for key in ("id", "kind", "path")}
                    )
            if (
                manifest.get("schema_version") != "cuda-evidence/attempt-v1"
                or manifest.get("attempt_id") != seal.get("attempt_id")
                or manifest.get("state") != seal.get("attempt_state")
                or manifest.get("claim_layer") != seal.get("claim_layer")
                or expected_refs != actual_refs
            ):
                reasons.append("attempt_manifest_binding_mismatch")
        except Exception as error:
            reasons.append(f"attempt_manifest_unavailable:{error}")
    artifacts = seal.get("artifacts")
    if not isinstance(artifacts, list):
        reasons.append("artifacts_invalid")
        artifacts = []
    for record in artifacts:
        kind = record.get("kind", "unknown") if isinstance(record, Mapping) else "unknown"
        if not isinstance(record, Mapping) or set(record) != {
            "id",
            "kind",
            "path",
            "sha256",
            "size_bytes",
        }:
            reasons.append(f"artifact_record_invalid:{kind}")
            continue
        try:
            _string(record["id"], "artifact.id")
            _string(record["kind"], "artifact.kind")
            _sha256(record["sha256"], "artifact.sha256")
            _literal_int(record["size_bytes"], "artifact.size_bytes")
            path = _safe_regular(root, record["path"])
            if _sha256_file(path) != record["sha256"] or path.stat().st_size != record["size_bytes"]:
                reasons.append(f"artifact_digest_mismatch:{kind}")
        except Exception as error:
            reasons.append(f"artifact_unavailable:{kind}:{error}")
    integrity = "PASS" if not reasons else "FAIL"
    audit = {
        "schema_version": "cuda-evidence/audit-v1",
        "attempt_id": seal.get("attempt_id"),
        "attempt_state": seal.get("attempt_state"),
        "seal_sha256": _sha256_file(seal_file),
        "evidence_integrity": integrity,
        "artifact_count": len(artifacts),
        "reasons": reasons,
    }
    return seal, audit


def audit_seal(seal_path: Path | str, output_path: Path | str) -> dict:
    """Rehash a seal without interpreting its performance verdict."""
    seal_file = Path(seal_path)
    if seal_file.is_symlink() or not seal_file.is_file():
        raise ValueError("seal must be a regular non-symlink file")
    _, audit = _compute_seal_audit(seal_file)
    _write_json_create_once(output_path, audit)
    return copy.deepcopy(audit)


def decide_attempt(
    seal_path: Path | str,
    audit_path: Path | str,
    decision_path: Path | str,
    manifest_path: Path | str,
) -> dict:
    """Bind integrity and performance as separate inputs to one decision closure."""
    decision_file = Path(decision_path)
    closure_file = Path(manifest_path)
    if decision_file.exists() or decision_file.is_symlink():
        raise FileExistsError(decision_file)
    if closure_file.exists() or closure_file.is_symlink():
        raise FileExistsError(closure_file)
    seal_file = Path(seal_path)
    audit_file = Path(audit_path)
    if seal_file.is_symlink() or not seal_file.is_file():
        raise ValueError("seal must be a regular non-symlink file")
    if audit_file.is_symlink() or not audit_file.is_file():
        raise ValueError("audit must be a regular non-symlink file")
    evidence_root = seal_file.parent.resolve(strict=True)
    for label, path in (
        ("audit", audit_file),
        ("decision", decision_file),
        ("manifest", closure_file),
    ):
        if path.parent.is_symlink() or path.parent.resolve(strict=True) != evidence_root:
            raise ValueError(f"{label} must be in the seal evidence directory")
    seal, current_audit = _compute_seal_audit(seal_file)
    provided_audit = load_json_strict(audit_file)
    seal_sha = _sha256_file(seal_file)
    integrity = current_audit["evidence_integrity"]
    reasons = list(current_audit["reasons"])
    if provided_audit != current_audit:
        integrity = "FAIL"
        reasons.append("provided_audit_mismatch")

    by_kind = {}
    try:
        for index, record in enumerate(seal.get("artifacts", [])):
            if not isinstance(record, Mapping):
                raise ValueError(f"seal.artifacts[{index}] must be an object")
            kind = _string(record.get("kind"), f"seal.artifacts[{index}].kind")
            if kind in by_kind:
                raise ValueError(f"duplicate sealed artifact kind: {kind}")
            by_kind[kind] = {
                "file_path": _safe_regular(seal_file.parent, record.get("path")),
                "record": record,
            }
        recomputed_gates, recomputed_errors = _semantic_gate_results(by_kind)
    except Exception as error:
        recomputed_gates = {}
        recomputed_errors = {"seal_artifacts": str(error)}
    if recomputed_gates != seal.get("gate_results") or recomputed_errors != seal.get("gate_errors"):
        integrity = "FAIL"
        reasons.append("sealed_gate_results_mismatch")
    performance_record = next(
        (
            record
            for record in seal.get("artifacts", [])
            if isinstance(record, Mapping) and record.get("kind") == "performance_verdict"
        ),
        None,
    )
    verdict_status = "unknown"
    promotional_eligible = False
    if performance_record is None:
        reasons.append("performance_verdict_ref_missing")
    else:
        try:
            performance_path = _safe_regular(seal_file.parent, performance_record["path"])
            verdict = _validate_performance_verdict(load_json_strict(performance_path))
            verdict_status = verdict["status"]
            promotional_eligible = verdict["promotional_eligible"]
        except Exception as error:
            reasons.append(f"performance_verdict_invalid:{error}")
    required_gates = {
        "experiment_design",
        "schedule_binding",
        "guard",
        "execution_path",
        "raw_rows",
        "performance_verdict",
    }
    if seal.get("claim_layer") == "serving_endpoint":
        required_gates |= {"serving_experiment", "artifact_identities"}
    gate_results = recomputed_gates
    gates_pass = all(
        isinstance(gate_results.get(name), Mapping)
        and gate_results[name].get("status") == "PASS"
        for name in required_gates
    )
    promote = (
        seal.get("attempt_state") == "valid"
        and integrity == "PASS"
        and gates_pass
        and verdict_status == "confirmed_win"
        and promotional_eligible
    )
    if not promote and not reasons:
        reasons.append("promotion_gates_not_satisfied")
    decision = {
        "schema_version": "cuda-evidence/decision-v1",
        "attempt_id": seal.get("attempt_id"),
        "attempt_state": seal.get("attempt_state"),
        "performance_verdict": verdict_status,
        "evidence_integrity": integrity if integrity in {"PASS", "FAIL"} else "FAIL",
        "decision": "promote" if promote else "retain",
        "reasons": reasons,
        "evidence_refs": {
            "seal": {"path": seal_file.name, "sha256": seal_sha},
            "audit": {"path": audit_file.name, "sha256": _sha256_file(audit_file)},
            "performance_verdict": (
                None
                if performance_record is None
                else {
                    "path": performance_record["path"],
                    "sha256": performance_record["sha256"],
                }
            ),
        },
    }
    _write_json_create_once(decision_file, decision)
    closure = {
        "schema_version": "cuda-evidence/manifest-v1",
        "attempt_id": seal.get("attempt_id"),
        "evidence_refs": {
            "seal": {"path": seal_file.name, "sha256": seal_sha},
            "audit": {"path": audit_file.name, "sha256": _sha256_file(audit_file)},
            "decision": {"path": decision_file.name, "sha256": _sha256_file(decision_file)},
        },
    }
    _write_json_create_once(closure_file, closure)
    return copy.deepcopy(decision)


def audit_imported_run(import_root: Path | str, output_dir: Path | str) -> dict:
    """Audit an imported run without writing beneath or mutating its source tree."""
    source = Path(import_root).resolve(strict=True)
    if not source.is_dir():
        raise ValueError("import root must be a directory")
    output = Path(output_dir)
    prospective = output.resolve(strict=False)
    if prospective == source or source in prospective.parents:
        raise ValueError("import audit output must be outside the imported tree")
    output.mkdir(parents=True, exist_ok=False)
    seal_file = source / "seal.json"
    if seal_file.is_file() and not seal_file.is_symlink():
        audited = audit_seal(seal_file, output / "sealed-audit.json")
        result = {
            "schema_version": "cuda-evidence/import-audit-v1",
            "status": "sealed_import",
            "promotional": False,
            "evidence_integrity": audited["evidence_integrity"],
            "audit_ref": {
                "path": "sealed-audit.json",
                "sha256": _sha256_file(output / "sealed-audit.json"),
            },
        }
    else:
        manifest_file = source / "manifest.json"
        schema_version = None
        if manifest_file.is_file() and not manifest_file.is_symlink():
            schema_version = load_json_strict(manifest_file).get("schema_version")
        result = {
            "schema_version": "cuda-evidence/import-audit-v1",
            "status": "legacy_unsealed",
            "promotional": False,
            "evidence_integrity": "UNKNOWN",
            "source_schema_version": schema_version,
            "migration_required": True,
        }
    _write_json_create_once(output / "import-audit.json", result)
    return copy.deepcopy(result)
