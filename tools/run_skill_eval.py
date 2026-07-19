#!/usr/bin/env python3
"""Validate and score comparable cuda-kernel-optimizer evaluation runs."""

from __future__ import annotations

import argparse
import copy
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
    _closed(suite, {"schema_version", "suite_id", "scenarios"}, "suite")
    if suite["schema_version"] != SUITE_SCHEMA:
        raise ValidationError(f"schema_version must be {SUITE_SCHEMA}")
    _identifier(suite["suite_id"], "suite_id")
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
    return _copy(result)


def score_results(suite: Mapping[str, Any], results: Sequence[Mapping[str, Any]], *, mode: str) -> dict:
    _identifier(mode, "mode")
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

