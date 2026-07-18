#!/usr/bin/env python3
"""Classify one performance iteration without running or changing the target."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


RECORD_SCHEMA = "cuda-optimizer/performance-iteration-v1"
REGISTRY_SCHEMA = "cuda-optimizer/measurement-path-registry-v1"
DECISION_SCHEMA = "cuda-optimizer/iteration-decision-v1"
SHA256_LENGTH = 64
NON_CANDIDATE_CLASSES = {"measurement_blocked", "infrastructure_only"}
VERDICTS = {"confirmed_win", "confirmed_loss", "inconclusive"}


def _pairs_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def load_json_strict(path: Path | str) -> dict:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"unsafe or missing JSON file: {source}")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {token}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {source}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return payload


def load_history(path: Path | str) -> list[dict]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"unsafe or missing history file: {source}")
    rows = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(
                raw,
                object_pairs_hook=_pairs_no_duplicates,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON value: {token}")
                ),
            )
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid history JSON at line {line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"history line {line_number} must be an object")
        _validate_history_row(row, f"history[{line_number}]")
        rows.append(row)
    return rows


def _closed_object(
    value: object,
    *,
    required: set[str],
    field: str,
    optional: set[str] | None = None,
) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    optional = optional or set()
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{field} missing keys: {', '.join(missing)}")
    unknown = sorted(set(value) - required - optional)
    if unknown:
        raise ValueError(f"{field} has unknown keys: {', '.join(unknown)}")
    return value


def _string(value: object, field: str, *, min_length: int = 1) -> str:
    if not isinstance(value, str) or len(value.strip()) < min_length:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _sha256(value: object, field: str) -> str:
    text = _string(value, field)
    if len(text) != SHA256_LENGTH or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    try:
        numeric = float(value)
    except OverflowError:
        raise ValueError(f"{field} must be a finite number") from None
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be a finite number")
    return numeric


def _positive_number(value: object, field: str) -> float:
    numeric = _finite_number(value, field)
    if numeric <= 0:
        raise ValueError(f"{field} must be positive")
    return numeric


def _non_negative_number(value: object, field: str) -> float:
    numeric = _finite_number(value, field)
    if numeric < 0:
        raise ValueError(f"{field} must be non-negative")
    return numeric


def _non_negative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _safe_relative_path(value: object, field: str) -> str:
    text = _string(value, field)
    path = PurePosixPath(text)
    if (
        text.startswith("/")
        or "\\" in text
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field} must be a safe relative path")
    return text


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_registry(payload: object) -> tuple[dict, str]:
    registry = _closed_object(
        payload,
        required={"schema_version", "paths"},
        field="registry",
    )
    if registry["schema_version"] != REGISTRY_SCHEMA:
        raise ValueError(f"registry.schema_version must be {REGISTRY_SCHEMA}")
    if not isinstance(registry["paths"], list) or not registry["paths"]:
        raise ValueError("registry.paths must be a non-empty array")
    normalized_paths = []
    identities = set()
    for index, value in enumerate(registry["paths"]):
        field = f"registry.paths[{index}]"
        item = _closed_object(
            value,
            required={"id", "version", "definition_sha256", "status"},
            field=field,
        )
        normalized = {
            "id": _string(item["id"], f"{field}.id"),
            "version": _string(item["version"], f"{field}.version"),
            "definition_sha256": _sha256(
                item["definition_sha256"], f"{field}.definition_sha256"
            ),
            "status": item["status"],
        }
        if normalized["status"] != "validated":
            raise ValueError(f"{field}.status must be validated")
        identity = (normalized["id"], normalized["version"])
        if identity in identities:
            raise ValueError(f"duplicate measurement path: {identity[0]}@{identity[1]}")
        identities.add(identity)
        normalized_paths.append(normalized)
    normalized = {"schema_version": REGISTRY_SCHEMA, "paths": normalized_paths}
    return normalized, _canonical_digest(normalized)


def _validate_path_binding(value: object, field: str, *, include_registry: bool) -> dict:
    required = {"id", "version", "definition_sha256"}
    if include_registry:
        required.add("registry_sha256")
    binding = _closed_object(value, required=required, field=field)
    normalized = {
        "id": _string(binding["id"], f"{field}.id"),
        "version": _string(binding["version"], f"{field}.version"),
        "definition_sha256": _sha256(
            binding["definition_sha256"], f"{field}.definition_sha256"
        ),
    }
    if include_registry:
        normalized["registry_sha256"] = _sha256(
            binding["registry_sha256"], f"{field}.registry_sha256"
        )
    return normalized


def _validate_hypothesis(value: object) -> dict:
    hypothesis = _closed_object(
        value,
        required={
            "statement",
            "mechanism",
            "target_metric",
            "direction",
            "minimum_effect_pct",
            "mutation_scope",
        },
        field="record.hypothesis",
    )
    direction = hypothesis["direction"]
    if direction not in {"lower", "higher"}:
        raise ValueError("record.hypothesis.direction must be lower or higher")
    if not isinstance(hypothesis["mutation_scope"], list) or not hypothesis[
        "mutation_scope"
    ]:
        raise ValueError("record.hypothesis.mutation_scope must be a non-empty array")
    scope = [
        _safe_relative_path(path, f"record.hypothesis.mutation_scope[{index}]")
        for index, path in enumerate(hypothesis["mutation_scope"])
    ]
    if len(scope) != len(set(scope)):
        raise ValueError("record.hypothesis.mutation_scope contains duplicates")
    return {
        "statement": _string(
            hypothesis["statement"], "record.hypothesis.statement", min_length=12
        ),
        "mechanism": _string(hypothesis["mechanism"], "record.hypothesis.mechanism"),
        "target_metric": _string(
            hypothesis["target_metric"], "record.hypothesis.target_metric"
        ),
        "direction": direction,
        "minimum_effect_pct": _positive_number(
            hypothesis["minimum_effect_pct"],
            "record.hypothesis.minimum_effect_pct",
        ),
        "mutation_scope": scope,
    }


def _validate_budget(value: object) -> dict:
    budget = _closed_object(
        value,
        required={
            "round_seconds",
            "infrastructure_seconds",
            "infrastructure_repairs",
        },
        field="record.budget",
    )
    return {
        "round_seconds": _positive_number(
            budget["round_seconds"], "record.budget.round_seconds"
        ),
        "infrastructure_seconds": _non_negative_number(
            budget["infrastructure_seconds"],
            "record.budget.infrastructure_seconds",
        ),
        "infrastructure_repairs": _non_negative_integer(
            budget["infrastructure_repairs"],
            "record.budget.infrastructure_repairs",
        ),
    }


def _validate_baseline(value: object) -> dict:
    baseline = _closed_object(
        value,
        required={"snapshot_sha256", "environment_sha256"},
        field="record.baseline",
    )
    return {
        "snapshot_sha256": _sha256(
            baseline["snapshot_sha256"], "record.baseline.snapshot_sha256"
        ),
        "environment_sha256": _sha256(
            baseline["environment_sha256"], "record.baseline.environment_sha256"
        ),
    }


def _path_is_in_scope(path: str, scope: str) -> bool:
    return path == scope or path.startswith(scope.rstrip("/") + "/")


def _validate_candidate(value: object, hypothesis: dict, baseline: dict) -> dict | None:
    if value is None:
        return None
    candidate = _closed_object(
        value,
        required={
            "candidate_id",
            "baseline_snapshot_sha256",
            "candidate_snapshot_sha256",
            "environment_sha256",
            "mechanism",
            "changed_paths",
        },
        field="record.candidate",
    )
    normalized = {
        "candidate_id": _string(candidate["candidate_id"], "record.candidate.candidate_id"),
        "baseline_snapshot_sha256": _sha256(
            candidate["baseline_snapshot_sha256"],
            "record.candidate.baseline_snapshot_sha256",
        ),
        "candidate_snapshot_sha256": _sha256(
            candidate["candidate_snapshot_sha256"],
            "record.candidate.candidate_snapshot_sha256",
        ),
        "environment_sha256": _sha256(
            candidate["environment_sha256"], "record.candidate.environment_sha256"
        ),
        "mechanism": _string(candidate["mechanism"], "record.candidate.mechanism"),
    }
    if normalized["baseline_snapshot_sha256"] != baseline["snapshot_sha256"]:
        raise ValueError("record.candidate.baseline_snapshot_sha256 does not match baseline")
    if normalized["environment_sha256"] != baseline["environment_sha256"]:
        raise ValueError("record.candidate.environment_sha256 does not match baseline")
    if normalized["candidate_snapshot_sha256"] == baseline["snapshot_sha256"]:
        raise ValueError("record.candidate snapshot must differ from baseline")
    if normalized["mechanism"] != hypothesis["mechanism"]:
        raise ValueError("record.candidate.mechanism does not match hypothesis")
    changed = candidate["changed_paths"]
    if not isinstance(changed, list) or not changed:
        raise ValueError("record.candidate.changed_paths must be a non-empty array")
    normalized_paths = [
        _safe_relative_path(path, f"record.candidate.changed_paths[{index}]")
        for index, path in enumerate(changed)
    ]
    if len(normalized_paths) != len(set(normalized_paths)):
        raise ValueError("record.candidate.changed_paths contains duplicates")
    for path in normalized_paths:
        if not any(_path_is_in_scope(path, scope) for scope in hypothesis["mutation_scope"]):
            raise ValueError(f"record.candidate.changed_paths escapes mutation_scope: {path}")
    normalized["changed_paths"] = normalized_paths
    return normalized


def _validate_bound_identity(
    value: Mapping,
    field: str,
    *,
    baseline: dict,
    candidate: dict,
    path_binding: dict,
) -> dict:
    normalized = {
        "baseline_snapshot_sha256": _sha256(
            value["baseline_snapshot_sha256"], f"{field}.baseline_snapshot_sha256"
        ),
        "candidate_snapshot_sha256": _sha256(
            value["candidate_snapshot_sha256"], f"{field}.candidate_snapshot_sha256"
        ),
        "environment_sha256": _sha256(
            value["environment_sha256"], f"{field}.environment_sha256"
        ),
        "measurement_path": _validate_path_binding(
            value["measurement_path"], f"{field}.measurement_path", include_registry=False
        ),
    }
    expected = {
        "baseline_snapshot_sha256": baseline["snapshot_sha256"],
        "candidate_snapshot_sha256": candidate["candidate_snapshot_sha256"],
        "environment_sha256": baseline["environment_sha256"],
        "measurement_path": {
            key: path_binding[key] for key in ("id", "version", "definition_sha256")
        },
    }
    for key in ("baseline_snapshot_sha256", "candidate_snapshot_sha256", "environment_sha256"):
        if normalized[key] != expected[key]:
            raise ValueError(f"{field}.{key} does not match the frozen iteration")
    if normalized["measurement_path"] != expected["measurement_path"]:
        raise ValueError(f"{field}.measurement_path does not match the frozen iteration")
    return normalized


def _validate_correctness(
    value: object,
    *,
    baseline: dict,
    candidate: dict | None,
    path_binding: dict,
) -> dict | None:
    if value is None:
        return None
    if candidate is None:
        raise ValueError("record.correctness requires a candidate")
    correctness = _closed_object(
        value,
        required={
            "status",
            "baseline_snapshot_sha256",
            "candidate_snapshot_sha256",
            "environment_sha256",
            "measurement_path",
        },
        field="record.correctness",
    )
    if correctness["status"] not in {"passed", "failed"}:
        raise ValueError("record.correctness.status must be passed or failed")
    return {
        "status": correctness["status"],
        **_validate_bound_identity(
            correctness,
            "record.correctness",
            baseline=baseline,
            candidate=candidate,
            path_binding=path_binding,
        ),
    }


def _validate_performance(
    value: object,
    *,
    hypothesis: dict,
    baseline: dict,
    candidate: dict | None,
    path_binding: dict,
) -> dict | None:
    if value is None:
        return None
    if candidate is None:
        raise ValueError("record.performance requires a candidate")
    if not isinstance(value, dict):
        raise ValueError("record.performance must be an object or null")
    status = value.get("status")
    identity_keys = {
        "baseline_snapshot_sha256",
        "candidate_snapshot_sha256",
        "environment_sha256",
        "measurement_path",
    }
    common = {"status", "target_metric", "direction"} | identity_keys
    if status == "failed":
        performance = _closed_object(
            value,
            required=common | {"error"},
            field="record.performance",
        )
        normalized = {
            "status": "failed",
            "error": _string(performance["error"], "record.performance.error"),
            "target_metric": _string(
                performance["target_metric"], "record.performance.target_metric"
            ),
            "direction": performance["direction"],
        }
    elif status == "completed":
        performance = _closed_object(
            value,
            required=common
            | {
                "verdict",
                "minimum_effect_pct",
                "estimate_pct",
                "ci_low_pct",
                "ci_high_pct",
            },
            field="record.performance",
        )
        if performance["verdict"] not in VERDICTS:
            raise ValueError("record.performance.verdict is invalid")
        low = _finite_number(performance["ci_low_pct"], "record.performance.ci_low_pct")
        high = _finite_number(performance["ci_high_pct"], "record.performance.ci_high_pct")
        if low > high:
            raise ValueError("record.performance confidence interval is reversed")
        normalized = {
            "status": "completed",
            "verdict": performance["verdict"],
            "target_metric": _string(
                performance["target_metric"], "record.performance.target_metric"
            ),
            "direction": performance["direction"],
            "minimum_effect_pct": _positive_number(
                performance["minimum_effect_pct"],
                "record.performance.minimum_effect_pct",
            ),
            "estimate_pct": _finite_number(
                performance["estimate_pct"], "record.performance.estimate_pct"
            ),
            "ci_low_pct": low,
            "ci_high_pct": high,
        }
        if normalized["minimum_effect_pct"] != hypothesis["minimum_effect_pct"]:
            raise ValueError("record.performance.minimum_effect_pct does not match hypothesis")
        if (
            normalized["verdict"] == "confirmed_win"
            and normalized["ci_low_pct"] < normalized["minimum_effect_pct"]
        ):
            raise ValueError(
                "record.performance confirmed_win does not clear minimum_effect_pct"
            )
    else:
        raise ValueError("record.performance.status must be completed or failed")
    if normalized["target_metric"] != hypothesis["target_metric"]:
        raise ValueError("record.performance.target_metric does not match hypothesis")
    if normalized["direction"] != hypothesis["direction"]:
        raise ValueError("record.performance.direction does not match hypothesis")
    normalized.update(
        _validate_bound_identity(
            performance,
            "record.performance",
            baseline=baseline,
            candidate=candidate,
            path_binding=path_binding,
        )
    )
    return normalized


def _validate_history_row(row: object, field: str) -> None:
    if not isinstance(row, dict):
        raise ValueError(f"{field} must be an object")
    if row.get("schema_version") != DECISION_SCHEMA:
        raise ValueError(f"{field}.schema_version must be {DECISION_SCHEMA}")
    _string(row.get("lineage_id"), f"{field}.lineage_id")
    if row.get("work_class") not in {
        "candidate_evaluated",
        "measurement_blocked",
        "infrastructure_only",
    }:
        raise ValueError(f"{field}.work_class is invalid")


def _select_fallback(registry: dict, current: dict) -> dict | None:
    for path in registry["paths"]:
        if (path["id"], path["version"], path["definition_sha256"]) != (
            current["id"],
            current["version"],
            current["definition_sha256"],
        ):
            return dict(path)
    return None


def classify_iteration(
    record_payload: Mapping,
    registry_payload: Mapping,
    history: Sequence[Mapping] = (),
) -> dict:
    registry, registry_sha256 = _validate_registry(registry_payload)
    record = _closed_object(
        record_payload,
        required={
            "schema_version",
            "round_id",
            "lineage_id",
            "hypothesis",
            "budget",
            "measurement_path",
            "baseline",
            "candidate",
            "correctness",
            "performance",
        },
        field="record",
    )
    if record["schema_version"] != RECORD_SCHEMA:
        raise ValueError(f"record.schema_version must be {RECORD_SCHEMA}")
    round_id = _string(record["round_id"], "record.round_id")
    lineage_id = _string(record["lineage_id"], "record.lineage_id")
    hypothesis = _validate_hypothesis(record["hypothesis"])
    budget = _validate_budget(record["budget"])
    path_binding = _validate_path_binding(
        record["measurement_path"], "record.measurement_path", include_registry=True
    )
    if path_binding["registry_sha256"] != registry_sha256:
        raise ValueError("record.measurement_path.registry_sha256 does not match registry")
    matching_paths = [
        path
        for path in registry["paths"]
        if all(path[key] == path_binding[key] for key in ("id", "version", "definition_sha256"))
    ]
    if len(matching_paths) != 1:
        raise ValueError("record measurement path is not a validated registry entry")
    baseline = _validate_baseline(record["baseline"])
    candidate = _validate_candidate(record["candidate"], hypothesis, baseline)
    correctness = _validate_correctness(
        record["correctness"],
        baseline=baseline,
        candidate=candidate,
        path_binding=path_binding,
    )
    performance = _validate_performance(
        record["performance"],
        hypothesis=hypothesis,
        baseline=baseline,
        candidate=candidate,
        path_binding=path_binding,
    )
    for index, row in enumerate(history):
        _validate_history_row(row, f"history[{index}]")

    if candidate is None:
        if correctness is not None or performance is not None:
            raise ValueError("candidate evidence cannot exist without a candidate")
        if budget["infrastructure_seconds"] == 0 and budget["infrastructure_repairs"] == 0:
            raise ValueError("a round without a candidate must record infrastructure work")
        work_class = "infrastructure_only"
        performance_result = "not_measured"
        claims = []
    elif correctness is None:
        work_class = "measurement_blocked"
        performance_result = "not_measured"
        claims = []
    elif correctness["status"] == "failed":
        if performance is not None:
            raise ValueError("performance must be null after failed correctness")
        work_class = "candidate_evaluated"
        performance_result = "correctness_failed"
        claims = ["candidate_evaluated"]
    elif performance is None or performance["status"] == "failed":
        work_class = "measurement_blocked"
        performance_result = "not_measured"
        claims = []
    else:
        work_class = "candidate_evaluated"
        performance_result = performance["verdict"]
        claims = ["candidate_evaluated"]

    infrastructure_cap = min(1200, math.floor(budget["round_seconds"] * 0.15))
    reasons = []
    if budget["infrastructure_seconds"] > infrastructure_cap:
        reasons.append("infrastructure_budget_exceeded")
    if budget["infrastructure_repairs"] > 1:
        reasons.append("infrastructure_repair_limit_exceeded")
    lineage_history = [row for row in history if row.get("lineage_id") == lineage_id]
    if (
        work_class in NON_CANDIDATE_CLASSES
        and lineage_history
        and lineage_history[-1].get("work_class") in NON_CANDIDATE_CLASSES
    ):
        reasons.append("two_consecutive_non_candidate_rounds")

    fallback = _select_fallback(registry, path_binding)
    forced = bool(reasons)
    if forced or work_class == "measurement_blocked":
        next_action = "switch_measurement_path" if fallback else "stop_direction"
    elif work_class == "infrastructure_only":
        next_action = "return_to_candidate"
    elif performance_result == "confirmed_win":
        next_action = "proceed_to_existing_promotion_gate"
    else:
        next_action = "continue_candidate_search"

    normalized_record = {
        "schema_version": RECORD_SCHEMA,
        "round_id": round_id,
        "lineage_id": lineage_id,
        "hypothesis": hypothesis,
        "budget": budget,
        "measurement_path": path_binding,
        "baseline": baseline,
        "candidate": candidate,
        "correctness": correctness,
        "performance": performance,
    }
    return {
        "schema_version": DECISION_SCHEMA,
        "round_id": round_id,
        "lineage_id": lineage_id,
        "record_sha256": _canonical_digest(normalized_record),
        "registry_sha256": registry_sha256,
        "work_class": work_class,
        "performance_result": performance_result,
        "claims": claims,
        "budget": {
            "round_seconds": budget["round_seconds"],
            "infrastructure_seconds": budget["infrastructure_seconds"],
            "infrastructure_cap_seconds": infrastructure_cap,
            "infrastructure_repairs": budget["infrastructure_repairs"],
            "infrastructure_repair_cap": 1,
        },
        "measurement_path": {
            key: path_binding[key] for key in ("id", "version", "definition_sha256")
        },
        "fallback_measurement_path": fallback,
        "reasons": reasons,
        "next_action": next_action,
    }


def _publish_create_once(path: Path | str, payload: Mapping) -> None:
    target = Path(path)
    if target.is_symlink():
        raise ValueError(f"output path is a symlink: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(target, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            target.unlink()
        except OSError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify CUDA optimizer rounds without running target commands."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="validate and classify one iteration record")
    check.add_argument("--record", required=True)
    check.add_argument("--registry", required=True)
    check.add_argument("--history")
    check.add_argument("--out", required=True)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        history = load_history(args.history) if args.history else []
        result = classify_iteration(
            load_json_strict(args.record),
            load_json_strict(args.registry),
            history,
        )
        _publish_create_once(args.out, result)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
