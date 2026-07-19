#!/usr/bin/env python3
"""Validate and score comparable cuda-kernel-optimizer evaluation runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SUITE_SCHEMA = "cuda-skill-eval/suite-v1"
SCORE_SCHEMA = "cuda-skill-eval/score-v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_CLAIMS = {"kernel", "workload", "serving"}
_STATUSES = {"completed", "blocked", "failed"}
_ARMS = [
    "no_skill",
    "v2.9",
    "v3_random_planner",
    "v3_shuffled_registry",
    "v3_full",
]
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _load_artifact_store():
    path = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "cuda-kernel-optimizer"
        / "scripts"
        / "artifact_store.py"
    )
    spec = importlib.util.spec_from_file_location("cuda_skill_eval_artifacts", path)
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


def load_json_strict(path: str | os.PathLike, *, root_type: type) -> Any:
    try:
        raw = _ARTIFACT_STORE.read_regular_bytes(path)
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_invalid_number,
        )
    except ValidationError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValidationError(f"invalid or unsafe JSON file {path}: {error}") from error
    if type(value) is not root_type:
        raise ValidationError(f"JSON root in {path} must be {root_type.__name__}")
    return value


def _object(value: Any, field: str) -> dict:
    if type(value) is not dict:
        raise ValidationError(f"{field} must be an object")
    return value


def _closed(value: Mapping, fields: set[str], field: str) -> None:
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ValidationError(f"{field} contains unknown fields: {', '.join(unknown)}")
    missing = sorted(fields - set(value))
    if missing:
        raise ValidationError(f"{field} is missing required fields: {', '.join(missing)}")


def _identifier(value: Any, field: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValidationError(f"{field} must be a safe identifier")
    return value


def _sha(value: Any, field: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{field} must be lowercase SHA-256")
    return value


def _string(value: Any, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValidationError(f"{field} must be a non-empty string")
    return value


def _events(value: Any, field: str) -> list[str]:
    if type(value) is not list:
        raise ValidationError(f"{field} must be an array")
    events = [_identifier(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if len(events) != len(set(events)):
        raise ValidationError(f"{field} must not contain duplicates")
    return events


def _finite(value: Any, field: str, *, nullable: bool = False) -> float | None:
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{field} must be a finite number")
    if number < 0:
        raise ValidationError(f"{field} must not be negative")
    return number


def _count(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{field} must be a nonnegative integer")
    return value


def _copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError("value must be finite JSON") from error


def validate_suite(value: Mapping[str, Any], root: str | os.PathLike) -> dict:
    suite = _object(value, "suite")
    _closed(
        suite,
        {"schema_version", "suite_id", "experiment", "scenarios"},
        "suite",
    )
    if suite["schema_version"] != SUITE_SCHEMA:
        raise ValidationError(f"schema_version must be {SUITE_SCHEMA}")
    _identifier(suite["suite_id"], "suite_id")
    experiment = _object(suite["experiment"], "experiment")
    _closed(
        experiment,
        {"arms", "replicates", "seed_policy", "aggregation", "release_gate"},
        "experiment",
    )
    if experiment["arms"] != _ARMS:
        raise ValidationError(
            "experiment arms must use the preregistered five-arm matrix"
        )
    if (
        isinstance(experiment["replicates"], bool)
        or not isinstance(experiment["replicates"], int)
        or experiment["replicates"] < 2
    ):
        raise ValidationError("experiment.replicates must be an integer of at least 2")
    if experiment["seed_policy"] != "paired_fixed":
        raise ValidationError("experiment.seed_policy must be paired_fixed")
    if experiment["aggregation"] != "median":
        raise ValidationError("experiment.aggregation must be median")
    release_gate = _object(experiment["release_gate"], "experiment.release_gate")
    _closed(
        release_gate,
        {"candidate_arm", "baseline_arm", "ablation_arms", "must_pass_all_scenarios"},
        "experiment.release_gate",
    )
    if release_gate != {
        "candidate_arm": "v3_full",
        "baseline_arm": "v2.9",
        "ablation_arms": ["v3_random_planner", "v3_shuffled_registry"],
        "must_pass_all_scenarios": True,
    }:
        raise ValidationError("experiment.release_gate must preserve the V3 ablations")
    scenarios = suite["scenarios"]
    if type(scenarios) is not list or not scenarios:
        raise ValidationError("scenarios must be a non-empty array")
    root_path = Path(root).resolve()
    ids = set()
    fields = {
        "id",
        "category",
        "fixture",
        "claim_ceiling",
        "budget",
        "required_events",
        "forbidden_events",
        "prompt_ref",
        "prompt_sha256",
        "prompt_id",
        "required_arms",
        "oracle",
    }
    for index, item in enumerate(scenarios):
        scenario = _object(item, f"scenarios[{index}]")
        _closed(scenario, fields, f"scenarios[{index}]")
        scenario_id = _identifier(scenario["id"], f"scenarios[{index}].id")
        if scenario_id in ids:
            raise ValidationError("scenario ids must be unique")
        ids.add(scenario_id)
        _identifier(scenario["category"], f"scenarios[{index}].category")
        relative = Path(_string(scenario["fixture"], f"scenarios[{index}].fixture"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValidationError(f"scenarios[{index}].fixture must be contained")
        fixture = (root_path / relative).resolve()
        try:
            fixture.relative_to(root_path)
        except ValueError as error:
            raise ValidationError(f"scenarios[{index}].fixture escapes root") from error
        if not fixture.is_file():
            raise ValidationError(f"scenarios[{index}].fixture does not exist")
        if scenario["claim_ceiling"] not in _CLAIMS:
            raise ValidationError(f"scenarios[{index}].claim_ceiling is unsupported")
        budget = _object(scenario["budget"], f"scenarios[{index}].budget")
        _closed(budget, {"max_seconds", "max_candidates"}, f"scenarios[{index}].budget")
        if (_finite(budget["max_seconds"], f"scenarios[{index}].budget.max_seconds") or 0) <= 0:
            raise ValidationError("budget.max_seconds must be positive")
        if _count(budget["max_candidates"], f"scenarios[{index}].budget.max_candidates") <= 0:
            raise ValidationError("budget.max_candidates must be positive")
        required = _events(scenario["required_events"], f"scenarios[{index}].required_events")
        forbidden = _events(scenario["forbidden_events"], f"scenarios[{index}].forbidden_events")
        if set(required).intersection(forbidden):
            raise ValidationError("required_events and forbidden_events must not overlap")
        prompt_relative = Path(
            _string(scenario["prompt_ref"], f"scenarios[{index}].prompt_ref")
        )
        if prompt_relative.is_absolute() or ".." in prompt_relative.parts:
            raise ValidationError(f"scenarios[{index}].prompt_ref must be contained")
        prompt_path = (root_path / prompt_relative).resolve()
        try:
            prompt_path.relative_to(root_path)
        except ValueError as error:
            raise ValidationError(f"scenarios[{index}].prompt_ref escapes root") from error
        if not prompt_path.is_file():
            raise ValidationError(f"scenarios[{index}].prompt_ref does not exist")
        expected_prompt_sha = _sha(
            scenario["prompt_sha256"], f"scenarios[{index}].prompt_sha256"
        )
        if hashlib.sha256(prompt_path.read_bytes()).hexdigest() != expected_prompt_sha:
            raise ValidationError(f"scenarios[{index}] prompt identity changed")
        _identifier(scenario["prompt_id"], f"scenarios[{index}].prompt_id")
        if scenario["required_arms"] != _ARMS:
            raise ValidationError(f"scenarios[{index}] must require every experiment arm")
        if scenario["oracle"] != "ledger_and_artifacts_v1":
            raise ValidationError(
                f"scenarios[{index}].oracle must be ledger_and_artifacts_v1"
            )
    return _copy(suite)


def _validate_result(value: Mapping[str, Any], index: int) -> dict:
    result = _object(value, f"results[{index}]")
    fields = {
        "scenario_id",
        "status",
        "elapsed_seconds",
        "gpu_seconds",
        "candidates",
        "valid_candidates",
        "first_valid_hypothesis_seconds",
        "end_to_end_gain_pct",
        "correctness_violations",
        "policy_violations",
        "events",
        "event_evidence",
        "run_identity",
    }
    _closed(result, fields, f"results[{index}]")
    _identifier(result["scenario_id"], f"results[{index}].scenario_id")
    if result["status"] not in _STATUSES:
        raise ValidationError(f"results[{index}].status is unsupported")
    _finite(result["elapsed_seconds"], f"results[{index}].elapsed_seconds")
    _finite(result["gpu_seconds"], f"results[{index}].gpu_seconds")
    candidates = _count(result["candidates"], f"results[{index}].candidates")
    valid = _count(result["valid_candidates"], f"results[{index}].valid_candidates")
    if valid > candidates:
        raise ValidationError(f"results[{index}].valid_candidates exceeds candidates")
    _finite(
        result["first_valid_hypothesis_seconds"],
        f"results[{index}].first_valid_hypothesis_seconds",
        nullable=True,
    )
    gain = result["end_to_end_gain_pct"]
    if gain is not None:
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(float(gain)):
            raise ValidationError(f"results[{index}].end_to_end_gain_pct must be finite or null")
    _count(result["correctness_violations"], f"results[{index}].correctness_violations")
    _count(result["policy_violations"], f"results[{index}].policy_violations")
    _events(result["events"], f"results[{index}].events")
    event_evidence = result["event_evidence"]
    if type(event_evidence) is not list:
        raise ValidationError(f"results[{index}].event_evidence must be an array")
    bound_names = []
    for evidence_index, item in enumerate(event_evidence):
        evidence = _object(
            item, f"results[{index}].event_evidence[{evidence_index}]"
        )
        _closed(
            evidence,
            {"name", "ledger_sequence", "source_sha256"},
            f"results[{index}].event_evidence[{evidence_index}]",
        )
        bound_names.append(
            _identifier(
                evidence["name"],
                f"results[{index}].event_evidence[{evidence_index}].name",
            )
        )
        if (
            isinstance(evidence["ledger_sequence"], bool)
            or not isinstance(evidence["ledger_sequence"], int)
            or evidence["ledger_sequence"] <= 0
        ):
            raise ValidationError("event evidence ledger_sequence must be positive")
        _sha(evidence["source_sha256"], "event evidence source_sha256")
    if bound_names != result["events"]:
        raise ValidationError("every event must have matching event evidence")
    identity = _object(result["run_identity"], f"results[{index}].run_identity")
    _closed(
        identity,
        {
            "model_identity",
            "prompt_sha256",
            "skill_sha256",
            "contract_sha256",
            "environment_sha256",
            "seed",
            "replicate",
        },
        f"results[{index}].run_identity",
    )
    _string(identity["model_identity"], "run_identity.model_identity")
    _sha(identity["prompt_sha256"], "run_identity.prompt_sha256")
    _sha(identity["skill_sha256"], "run_identity.skill_sha256", nullable=True)
    _sha(identity["contract_sha256"], "run_identity.contract_sha256")
    _sha(identity["environment_sha256"], "run_identity.environment_sha256")
    if isinstance(identity["seed"], bool) or not isinstance(identity["seed"], int):
        raise ValidationError("run_identity.seed must be an integer")
    if (
        isinstance(identity["replicate"], bool)
        or not isinstance(identity["replicate"], int)
        or identity["replicate"] <= 0
    ):
        raise ValidationError("run_identity.replicate must be positive")
    return _copy(result)


def score_results(suite: Mapping[str, Any], results: Sequence[Mapping[str, Any]], *, mode: str) -> dict:
    _identifier(mode, "mode")
    if mode not in suite["experiment"]["arms"]:
        raise ValidationError("mode is not a preregistered experiment arm")
    if isinstance(results, (str, bytes, bytearray, Mapping)):
        raise ValidationError("results must be an array")
    validated = [_validate_result(item, index) for index, item in enumerate(results)]
    by_id = {}
    for result in validated:
        scenario_id = result["scenario_id"]
        if scenario_id in by_id:
            raise ValidationError(f"duplicate result for scenario: {scenario_id}")
        by_id[scenario_id] = result
    expected = {item["id"] for item in suite["scenarios"]}
    missing = sorted(expected - set(by_id))
    extra = sorted(set(by_id) - expected)
    if missing:
        raise ValidationError(f"missing scenario results: {', '.join(missing)}")
    if extra:
        raise ValidationError(f"unknown scenario results: {', '.join(extra)}")

    details = []
    total_candidates = 0
    total_valid = 0
    for scenario in suite["scenarios"]:
        result = by_id[scenario["id"]]
        identity = result["run_identity"]
        if identity["prompt_sha256"] != scenario["prompt_sha256"]:
            raise ValidationError(
                f"prompt identity mismatch for scenario {scenario['id']}"
            )
        if identity["replicate"] > suite["experiment"]["replicates"]:
            raise ValidationError(
                f"replicate exceeds preregistered count for scenario {scenario['id']}"
            )
        if mode == "no_skill" and identity["skill_sha256"] is not None:
            raise ValidationError("no_skill arm must not bind a skill_sha256")
        if mode != "no_skill" and identity["skill_sha256"] is None:
            raise ValidationError(f"{mode} arm must bind skill_sha256")
        events = set(result["events"])
        required_missing = sorted(set(scenario["required_events"]) - events)
        forbidden_seen = sorted(set(scenario["forbidden_events"]) & events)
        passed = (
            result["status"] == "completed"
            and not required_missing
            and not forbidden_seen
            and result["correctness_violations"] == 0
            and result["policy_violations"] == 0
        )
        details.append(
            {
                "scenario_id": scenario["id"],
                "passed": passed,
                "required_events_missing": required_missing,
                "forbidden_events_seen": forbidden_seen,
            }
        )
        total_candidates += result["candidates"]
        total_valid += result["valid_candidates"]

    return {
        "schema_version": SCORE_SCHEMA,
        "suite_id": suite["suite_id"],
        "mode": mode,
        "scenarios_total": len(details),
        "scenarios_passed": sum(item["passed"] for item in details),
        "correctness_violations": sum(item["correctness_violations"] for item in validated),
        "policy_violations": sum(item["policy_violations"] for item in validated),
        "required_events_missing": sum(len(item["required_events_missing"]) for item in details),
        "forbidden_events_seen": sum(len(item["forbidden_events_seen"]) for item in details),
        "elapsed_seconds": sum(float(item["elapsed_seconds"]) for item in validated),
        "gpu_seconds": sum(float(item["gpu_seconds"]) for item in validated),
        "candidates": total_candidates,
        "valid_candidates": total_valid,
        "valid_candidate_rate": total_valid / total_candidates if total_candidates else None,
        "details": details,
        "run_identities": [
            {
                "scenario_id": item["scenario_id"],
                **item["run_identity"],
            }
            for item in validated
        ],
    }


def run_score(
    suite_path: str | os.PathLike,
    results_path: str | os.PathLike,
    out_path: str | os.PathLike,
    *,
    mode: str,
    root: str | os.PathLike,
) -> dict:
    suite = validate_suite(load_json_strict(suite_path, root_type=dict), root)
    results = load_json_strict(results_path, root_type=list)
    score = score_results(suite, results, mode=mode)
    try:
        _ARTIFACT_STORE.create_regular_json(out_path, score)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot create score artifact: {error}") from error
    return score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    score = run_score(args.suite, args.results, args.out, mode=args.mode, root=args.root)
    print(json.dumps(score, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
