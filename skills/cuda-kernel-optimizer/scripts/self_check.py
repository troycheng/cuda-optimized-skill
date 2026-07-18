#!/usr/bin/env python3
"""CPU/static installation check for the CUDA optimizer skill."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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
    "direction_lineage.schema.json",
    "direction_decision.schema.json",
)


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
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing script: {name}")
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    checks.append("python_scripts")

    for name in _SCHEMAS:
        path = root / "templates" / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing schema: {name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "v2.5" not in payload.get("$id", ""):
            raise ValueError(f"schema does not declare V2.5 identity: {name}")
        if payload.get("additionalProperties") is not False:
            raise ValueError(f"schema root must be closed: {name}")
    checks.append("v2_5_schemas")

    for name in _V2_6_SCHEMAS:
        path = root / "templates" / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing schema: {name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "v2.6" not in payload.get("$id", ""):
            raise ValueError(f"schema does not declare V2.6 identity: {name}")
        if payload.get("additionalProperties") is not False:
            raise ValueError(f"schema root must be closed: {name}")
    checks.append("v2_6_iteration_guard")

    for name in _V2_7_SCHEMAS:
        path = root / "templates" / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing schema: {name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "v2.7" not in payload.get("$id", ""):
            raise ValueError(f"schema does not declare V2.7 identity: {name}")
        if payload.get("additionalProperties") is not False:
            raise ValueError(f"schema root must be closed: {name}")
    reference = root / "references" / "direction_admission.md"
    if reference.is_symlink() or not reference.is_file():
        raise ValueError("missing reference: direction_admission.md")
    checks.append("v2_7_direction_guard")

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
