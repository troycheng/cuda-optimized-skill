#!/usr/bin/env python3
"""CLI for V2.5 formal guard, seal, audit, and decision evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import evidence_protocol as protocol
from experiment_design import validate_frozen_design


def _publish(path: str, payload: dict) -> None:
    protocol._write_json_create_once(Path(path), payload)  # noqa: SLF001


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and close CUDA optimizer V2.5 formal evidence."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    design = commands.add_parser("validate-design", help="validate a frozen experiment design")
    design.add_argument("--design", required=True)
    design.add_argument("--out", required=True)

    guard = commands.add_parser("guard-audit", help="audit normalized continuous shared-host samples")
    guard.add_argument("--policy", required=True)
    guard.add_argument("--samples", required=True)
    guard.add_argument("--markers", required=True)
    guard.add_argument("--out", required=True)

    coverage = commands.add_parser("coverage-audit", help="validate execution-path coverage")
    coverage.add_argument("--proof", required=True)
    coverage.add_argument("--out", required=True)

    serving = commands.add_parser("validate-serving", help="validate a serving experiment")
    serving.add_argument("--experiment", required=True)
    serving.add_argument("--out", required=True)

    identities = commands.add_parser("validate-identities", help="validate serving artifact identities")
    identities.add_argument("--identities", required=True)
    identities.add_argument("--out", required=True)

    profiler = commands.add_parser("validate-profiler", help="validate a non-promotional Nsys/NCU bundle")
    profiler.add_argument("--bundle", required=True)
    profiler.add_argument("--out", required=True)

    seal = commands.add_parser("seal", help="seal one terminal attempt")
    seal.add_argument("--attempt", required=True)
    seal.add_argument("--out", required=True)

    audit = commands.add_parser("audit", help="rehash one immutable seal")
    audit.add_argument("--seal", required=True)
    audit.add_argument("--out", required=True)

    decide = commands.add_parser("decide", help="bind seal and audit to a performance decision")
    decide.add_argument("--seal", required=True)
    decide.add_argument("--audit", required=True)
    decide.add_argument("--out", required=True)
    decide.add_argument("--manifest", required=True)

    imported = commands.add_parser("audit-imported", help="audit a serving run read-only")
    imported.add_argument("--run-dir", required=True)
    imported.add_argument("--out-dir", required=True)
    return parser


def _run(args: argparse.Namespace) -> tuple[dict, int]:
    if args.command == "validate-design":
        result = {
            "schema_version": "cuda-evidence/experiment-design-audit-v1",
            "status": "PASS",
            "design": validate_frozen_design(protocol.load_json_strict(args.design)),
        }
        _publish(args.out, result)
        return result, 0
    if args.command == "guard-audit":
        result = protocol.audit_shared_host_guard(
            protocol.load_json_strict(args.policy),
            protocol._load_jsonl_strict(args.samples),  # noqa: SLF001
            protocol._load_json_value_strict(args.markers),  # noqa: SLF001
        )
        _publish(args.out, result)
        return result, 0 if result["status"] == "PASS" else 3
    if args.command == "coverage-audit":
        result = protocol.validate_execution_path(protocol.load_json_strict(args.proof))
        _publish(args.out, result)
        return result, 0
    if args.command == "validate-serving":
        result = protocol.validate_serving_experiment(
            protocol.load_json_strict(args.experiment)
        )
        _publish(args.out, result)
        return result, 0
    if args.command == "validate-identities":
        result = protocol.validate_artifact_identities(
            protocol.load_json_strict(args.identities)
        )
        _publish(args.out, result)
        return result, 0
    if args.command == "validate-profiler":
        result = protocol.validate_profiler_bundle(protocol.load_json_strict(args.bundle))
        _publish(args.out, result)
        return result, 0
    if args.command == "seal":
        return protocol.seal_attempt(args.attempt, args.out), 0
    if args.command == "audit":
        result = protocol.audit_seal(args.seal, args.out)
        return result, 0 if result["evidence_integrity"] == "PASS" else 3
    if args.command == "decide":
        return (
            protocol.decide_attempt(args.seal, args.audit, args.out, args.manifest),
            0,
        )
    if args.command == "audit-imported":
        return protocol.audit_imported_run(args.run_dir, args.out_dir), 0
    raise AssertionError("unreachable")


def main(argv=None) -> int:
    try:
        result, returncode = _run(_parser().parse_args(argv))
    except (FileExistsError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
