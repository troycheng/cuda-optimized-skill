#!/usr/bin/env python3
"""CPU/static installation check for the CUDA optimizer skill."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from types import ModuleType


_SCHEMAS = (
    "guard_policy.schema.json",
    "experiment_design.schema.json",
    "attempt.schema.json",
    "execution_path.schema.json",
    "serving_experiment.schema.json",
    "artifact_identities.schema.json",
    "profiler_bundle.schema.json",
    "performance_verdict.schema.json",
    "evidence_manifest.schema.json",
)
_SCRIPTS = (
    "direction_guard.py",
    "evidence.py",
    "evidence_protocol.py",
    "experiment_design.py",
    "iteration_guard.py",
    "nonstationarity_guard.py",
    "workload_evaluate.py",
)
_V2_6_SCHEMAS = (
    "iteration_binding.schema.json",
    "iteration_lineage.schema.json",
    "measurement_path_registry.schema.json",
    "performance_iteration.schema.json",
)
_V2_7_SCHEMAS = (
    "direction_portfolio.schema.json",
    "direction_evidence.schema.json",
    "direction_lineage.schema.json",
    "direction_decision.schema.json",
)
_V2_8_SCHEMAS = (
    "nonstationarity_anchor.schema.json",
    "nonstationarity_design.schema.json",
    "nonstationarity_series.schema.json",
    "nonstationarity_verdict.schema.json",
)
_V3_SCRIPTS = (
    "workload_contract.py",
    "evidence_ledger.py",
    "run_control.py",
    "capability_query.py",
    "evidence_summary.py",
    "gate_evidence.py",
    "diagnostic_evidence.py",
    "evidence_controller.py",
    "planner_admission.py",
    "planner_boundary.py",
    "stability_calibration.py",
)
_V3_SCHEMAS = (
    "workload_contract.schema.json",
    "candidate_proposal.schema.json",
    "run_event.schema.json",
    "run_control.schema.json",
    "capability.schema.json",
    "observation_summary.schema.json",
    "gate_evidence.schema.json",
    "gate_measurement.schema.json",
    "diagnostic_evidence.schema.json",
    "diagnostic_measurement.schema.json",
    "planner_admission.schema.json",
    "stability_calibration.schema.json",
    "stability_audit.schema.json",
)


def _read_safe_file(root: Path, relative: Path | str) -> bytes:
    """Read a package file without following any child symlink."""
    root = Path(os.path.abspath(root))
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe package path: {relative}")
    common_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    directory_flags = common_flags | getattr(os, "O_DIRECTORY", 0)
    descriptors = []
    try:
        parent = os.open(root, directory_flags)
        descriptors.append(parent)
        for part in relative.parts[:-1]:
            parent = os.open(part, directory_flags, dir_fd=parent)
            descriptors.append(parent)
        descriptor = os.open(relative.parts[-1], common_flags, dir_fd=parent)
        descriptors.append(descriptor)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"unsafe package file: {relative}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as exc:
        raise ValueError(
            f"package path contains a symlink or unsafe component: {relative}"
        ) from exc
    finally:
        for opened in reversed(descriptors):
            os.close(opened)


def _validate_capability_registry(root: Path) -> None:
    script = root / "scripts" / "capability_query.py"
    module = ModuleType("installed_capability_query")
    module.__file__ = str(script)
    script_bytes = _read_safe_file(root, Path("scripts") / "capability_query.py")
    exec(compile(script_bytes, str(script), "exec"), module.__dict__)
    capability_root = root / "references" / "capabilities"
    module.validate_registry(
        registry_path=capability_root / "registry.json",
        sources_path=capability_root / "sources.json",
        capability_root=capability_root,
        trusted_root=root,
    )


def _validate_gate_schema_contract(root: Path) -> None:
    script = root / "scripts" / "gate_evidence.py"
    module = ModuleType("installed_gate_evidence")
    module.__file__ = str(script)
    source = _read_safe_file(root, Path("scripts") / "gate_evidence.py")
    exec(compile(source, str(script), "exec"), module.__dict__)
    measurement = json.loads(
        _read_safe_file(root, Path("templates") / "gate_measurement.schema.json")
    )
    variants = measurement.get("oneOf")
    if not isinstance(variants, list):
        raise ValueError("gate measurement schema must define closed kind variants")
    by_kind = {item.get("title"): item for item in variants}
    if set(by_kind) != set(module._SUBJECT_FIELDS):
        raise ValueError("gate measurement schema kind set differs from runtime")
    if measurement.get("properties", {}).get("checks", {}).get("uniqueItems") is not True:
        raise ValueError("gate measurement schema must reject duplicate checks")
    for kind, variant in by_kind.items():
        properties = variant.get("properties", {})
        subject = properties.get("subject", {})
        result = properties.get("result", {})
        if "$ref" in subject:
            subject_required = {"candidate_id", "candidate_sha256"}
        else:
            subject_required = set(subject.get("required", []))
        if subject_required != set(module._SUBJECT_FIELDS[kind]):
            raise ValueError(f"gate measurement subject schema differs for {kind}")
        if set(result.get("required", [])) != set(module._RESULT_FIELDS[kind]):
            raise ValueError(f"gate measurement result schema differs for {kind}")
    evidence = json.loads(
        _read_safe_file(root, Path("templates") / "gate_evidence.schema.json")
    )
    producer = evidence.get("properties", {}).get("producer", {})
    if "implementation_sha256" not in set(producer.get("required", [])):
        raise ValueError("gate evidence schema must bind adapter implementation")


def _validate_diagnostic_schema_contract(root: Path) -> None:
    script = root / "scripts" / "diagnostic_evidence.py"
    module = ModuleType("installed_diagnostic_evidence")
    module.__file__ = str(script)
    source = _read_safe_file(root, Path("scripts") / "diagnostic_evidence.py")
    exec(compile(source, str(script), "exec"), module.__dict__)
    measurement = json.loads(
        _read_safe_file(root, Path("templates") / "diagnostic_measurement.schema.json")
    )
    if measurement.get("properties", {}).get("schema_version", {}).get("const") != module.MEASUREMENT_SCHEMA:
        raise ValueError("diagnostic measurement schema version differs from runtime")
    signal_enum = set(
        measurement.get("properties", {}).get("signals", {}).get("items", {}).get("enum", [])
    )
    runtime_signals = set().union(*module._SIGNALS.values())
    if signal_enum != runtime_signals:
        raise ValueError("diagnostic measurement signal vocabulary differs from runtime")
    if measurement.get("properties", {}).get("checks", {}).get("uniqueItems") is not True:
        raise ValueError("diagnostic measurement schema must reject duplicate checks")
    evidence = json.loads(
        _read_safe_file(root, Path("templates") / "diagnostic_evidence.schema.json")
    )
    kind_enum = set(evidence.get("properties", {}).get("kind", {}).get("enum", []))
    if kind_enum != set(module._PRODUCERS):
        raise ValueError("diagnostic evidence kind set differs from runtime")
    if evidence.get("properties", {}).get("schema_version", {}).get("const") != module.EVIDENCE_SCHEMA:
        raise ValueError("diagnostic evidence schema version differs from runtime")
    producer = evidence.get("properties", {}).get("producer", {})
    if "implementation_sha256" not in set(producer.get("required", [])):
        raise ValueError("diagnostic evidence schema must bind adapter implementation")
    top_producers = set(
        producer.get("properties", {}).get("id", {}).get("enum", [])
    )
    if top_producers != set(module._PRODUCERS.values()):
        raise ValueError("diagnostic evidence producer vocabulary differs from runtime")
    top_signals = set(
        evidence.get("properties", {}).get("signals", {}).get("items", {}).get("enum", [])
    )
    if top_signals != runtime_signals:
        raise ValueError("diagnostic evidence signal vocabulary differs from runtime")
    contracts = {}
    for rule in evidence.get("allOf", []):
        kind = rule.get("if", {}).get("properties", {}).get("kind", {}).get("const")
        properties = rule.get("then", {}).get("properties", {})
        producer_id = (
            properties.get("producer", {}).get("properties", {}).get("id", {}).get("const")
        )
        signals = set(properties.get("signals", {}).get("items", {}).get("enum", []))
        if kind is not None:
            contracts[kind] = {"producer": producer_id, "signals": signals}
    expected_contracts = {
        kind: {"producer": module._PRODUCERS[kind], "signals": set(module._SIGNALS[kind])}
        for kind in module._PRODUCERS
    }
    if contracts != expected_contracts:
        raise ValueError("diagnostic per-kind schema contract differs from runtime")


def _validate_planner_admission_schema_contract(root: Path) -> None:
    script = root / "scripts" / "planner_admission.py"
    module = ModuleType("installed_planner_admission")
    module.__file__ = str(script)
    source = _read_safe_file(root, Path("scripts") / "planner_admission.py")
    exec(compile(source, str(script), "exec"), module.__dict__)
    schema = json.loads(
        _read_safe_file(root, Path("templates") / "planner_admission.schema.json")
    )
    if set(schema.get("required", [])) != set(module._FIELDS):
        raise ValueError("planner admission schema fields differ from runtime")
    if schema.get("properties", {}).get("schema_version", {}).get("const") != module.SCHEMA:
        raise ValueError("planner admission schema version differs from runtime")
    if "controller_attestation" not in schema.get("properties", {}):
        raise ValueError("planner admission schema must bind Controller attestation")


def _validate_stability_schema_contract(root: Path) -> None:
    script = root / "scripts" / "stability_calibration.py"
    module = ModuleType("installed_stability_calibration")
    module.__file__ = str(script)
    source = _read_safe_file(root, Path("scripts") / "stability_calibration.py")
    exec(compile(source, str(script), "exec"), module.__dict__)
    for filename, fields, schema_version in (
        ("stability_calibration.schema.json", module._CALIBRATION_FIELDS, module.CALIBRATION_SCHEMA),
        ("stability_audit.schema.json", module._AUDIT_FIELDS, module.AUDIT_SCHEMA),
    ):
        schema = json.loads(_read_safe_file(root, Path("templates") / filename))
        if set(schema.get("required", [])) != set(fields):
            raise ValueError(f"{filename} fields differ from runtime")
        if schema.get("properties", {}).get("schema_version", {}).get("const") != schema_version:
            raise ValueError(f"{filename} version differs from runtime")
        if "controller_attestation" not in schema.get("properties", {}):
            raise ValueError(f"{filename} must bind Controller attestation")

    calibration = json.loads(
        _read_safe_file(root, Path("templates") / "stability_calibration.schema.json")
    )
    properties = calibration.get("properties", {})
    calibration_reasons = set(
        properties.get("reasons", {}).get("items", {}).get("enum", [])
    )
    if calibration_reasons != set(module._CALIBRATION_REASONS):
        raise ValueError("stability calibration reason vocabulary differs from runtime")
    baseline_variants = properties.get("baseline_median", {}).get("oneOf", [])
    baseline_number = next(
        (item for item in baseline_variants if item.get("type") == "number"), {}
    )
    if baseline_number.get("exclusiveMinimum") != 0:
        raise ValueError("stability calibration baseline minimum differs from runtime")
    if properties.get("mde_method", {}).get("const") != module.MDE_METHOD:
        raise ValueError("stability calibration MDE method differs from runtime")
    if properties.get("bootstrap_samples", {}).get("minimum") != 1000:
        raise ValueError("stability calibration bootstrap minimum differs from runtime")
    if properties.get("min_valid_pairs", {}).get("minimum") != 4:
        raise ValueError("stability calibration pair minimum differs from runtime")

    audit = json.loads(
        _read_safe_file(root, Path("templates") / "stability_audit.schema.json")
    )
    audit_reasons = set(
        audit.get("properties", {}).get("reasons", {}).get("items", {}).get("enum", [])
    )
    if audit_reasons != set(module._AUDIT_REASONS):
        raise ValueError("stability audit reason vocabulary differs from runtime")


def check_installation(skill_dir: Path | str) -> dict:
    root = Path(skill_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"missing or unsafe skill directory: {root}")
    checks = []
    skill_file = root / "SKILL.md"
    if skill_file.is_symlink() or not skill_file.is_file():
        raise ValueError("missing SKILL.md")
    checks.append("skill_metadata")

    for name in _SCRIPTS:
        path = root / "scripts" / name
        source = _read_safe_file(root, Path("scripts") / name).decode("utf-8")
        compile(source, str(path), "exec")
    checks.append("python_scripts")

    for name in _SCHEMAS:
        payload = json.loads(_read_safe_file(root, Path("templates") / name))
        if "v2.5" not in payload.get("$id", ""):
            raise ValueError(f"schema does not declare V2.5 identity: {name}")
        if payload.get("additionalProperties") is not False:
            raise ValueError(f"schema root must be closed: {name}")
    checks.append("v2_5_schemas")

    for name in _V2_6_SCHEMAS:
        payload = json.loads(_read_safe_file(root, Path("templates") / name))
        if "v2.6" not in payload.get("$id", ""):
            raise ValueError(f"schema does not declare V2.6 identity: {name}")
        if payload.get("additionalProperties") is not False:
            raise ValueError(f"schema root must be closed: {name}")
    checks.append("v2_6_iteration_guard")

    for name in _V2_7_SCHEMAS:
        payload = json.loads(_read_safe_file(root, Path("templates") / name))
        if "v2.7" not in payload.get("$id", ""):
            raise ValueError(f"schema does not declare V2.7 identity: {name}")
        if payload.get("additionalProperties") is not False:
            raise ValueError(f"schema root must be closed: {name}")
    _read_safe_file(root, Path("references") / "direction_admission.md")
    checks.append("v2_7_direction_guard")

    for name in _V2_8_SCHEMAS:
        payload = json.loads(_read_safe_file(root, Path("templates") / name))
        if "v2.8" not in payload.get("$id", ""):
            raise ValueError(f"schema does not declare V2.8 identity: {name}")
        if payload.get("additionalProperties") is not False:
            raise ValueError(f"schema root must be closed: {name}")
    _read_safe_file(root, Path("references") / "nonstationary_serving_evidence.md")
    checks.append("v2_8_nonstationarity_guard")

    for name in _V3_SCRIPTS:
        path = root / "scripts" / name
        source = _read_safe_file(root, Path("scripts") / name).decode("utf-8")
        compile(source, str(path), "exec")
    for name in _V3_SCHEMAS:
        payload = json.loads(_read_safe_file(root, Path("templates") / name))
        if payload.get("additionalProperties") is not False:
            raise ValueError(f"V3 schema root must be closed: {name}")
    _validate_gate_schema_contract(root)
    _validate_diagnostic_schema_contract(root)
    _validate_planner_admission_schema_contract(root)
    _validate_stability_schema_contract(root)
    checks.append("v3_control_runtime")

    _validate_capability_registry(root)
    checks.append("v3_capability_registry")

    return {
        "schema_version": "cuda-evidence/self-check-v1",
        "status": "PASS",
        "checks": checks,
        "gpu_checks_run": False,
        "network_checks_run": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run CPU/static skill installation checks.")
    parser.add_argument(
        "--skill-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="installed cuda-kernel-optimizer skill directory",
    )
    args = parser.parse_args(argv)
    try:
        result = check_installation(args.skill_dir)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
