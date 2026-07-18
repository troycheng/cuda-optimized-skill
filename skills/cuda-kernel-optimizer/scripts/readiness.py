#!/usr/bin/env python3
"""Assess the strongest optimization claim supported by the available inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


FIELDS = (
    "source_available",
    "correctness_reference",
    "compile_command",
    "stable_kernel_benchmark",
    "representative_workload",
    "serving_benchmark",
)

NEXT_ACTIONS = {
    "source_available": "Make the target source accessible to this task, or provide an existing profiler report.",
    "correctness_reference": "Add a reference implementation, validator, or trusted output set.",
    "compile_command": "Record a reproducible build or import command and its isolated environment.",
    "stable_kernel_benchmark": "Add warmup, repeated paired timing, representative shapes, and raw samples.",
    "representative_workload": "Provide the real workload or replay and approve its case distribution.",
    "serving_benchmark": "Add the serving load plan, KPI strata, warmup, constraints, and request corpus.",
}

CLAIM_REQUIREMENTS = {
    "kernel": FIELDS[:4],
    "workload": FIELDS[:5],
    "serving": FIELDS,
}


def assess(payload: Dict[str, Any]) -> Dict[str, Any]:
    requested_claim = str(payload.get("requested_claim", "kernel"))
    if requested_claim not in CLAIM_REQUIREMENTS:
        raise ValueError("requested_claim must be kernel, workload, or serving")
    available = {field: bool(payload.get(field, False)) for field in FIELDS}
    required = CLAIM_REQUIREMENTS[requested_claim]
    missing = [field for field in required if not available[field]]

    if not available["source_available"]:
        ceiling = "blocked"
    elif not available["correctness_reference"]:
        ceiling = "static_hypotheses"
    elif not available["compile_command"]:
        ceiling = "correctness_design"
    elif not available["stable_kernel_benchmark"]:
        ceiling = "compile_evidence"
    elif requested_claim == "kernel" or not available["representative_workload"]:
        ceiling = "kernel_performance"
    elif requested_claim == "workload" or not available["serving_benchmark"]:
        ceiling = "workload_performance"
    else:
        ceiling = "serving_performance"

    return {
        "schema_version": "cuda-optimizer/readiness-v1",
        "status": "ready" if not missing else "needs_foundation",
        "requested_claim": requested_claim,
        "claim_ceiling": ceiling,
        "can_start_mutation": ceiling
        in {"kernel_performance", "workload_performance", "serving_performance"},
        "available": available,
        "missing": missing,
        "next_actions": [NEXT_ACTIONS[field] for field in missing],
        "host_change_policy": "recommend_only",
        "notes": [
            "Static or compiler findings are hypotheses, not performance wins.",
            "Only the user can approve representative workload cases and objectives.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report the strongest claim supported by the current GPU optimization setup."
    )
    parser.add_argument(
        "--input",
        default="-",
        help="Readiness JSON input path, or - for stdin (default)",
    )
    parser.add_argument("--out", help="Optional output JSON path; stdout is always written")
    args = parser.parse_args()

    if args.input == "-":
        payload = json.load(sys.stdin)
    else:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    try:
        report = assess(payload)
    except ValueError as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
