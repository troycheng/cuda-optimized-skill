#!/usr/bin/env python3
"""Validate normalized probe evidence and classify workload bottlenecks."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROBE_SCHEMA = "cuda-workload-optimizer/probe-v1"
DIAGNOSIS_SCHEMA = "cuda-workload-optimizer/diagnosis-v1"
POLICY_SCHEMA = "cuda-workload-optimizer/diagnosis-policy-v1"
CATEGORIES = (
    "kernel",
    "framework",
    "cpu_data",
    "transfer",
    "communication",
    "io",
    "environment",
)
METRICS = {
    "gpu_busy_pct",
    "kernel_time_pct",
    "cuda_api_time_pct",
    "launch_gap_pct",
    "cpu_busy_pct",
    "data_wait_pct",
    "io_wait_pct",
    "transfer_time_pct",
    "communication_time_pct",
    "graph_replay_pct",
}
PROBE_KINDS = {
    "environment",
    "timeline",
    "framework",
    "cpu_data",
    "transfer",
    "communication",
    "io",
    "custom",
}
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")


class DiagnosisError(ValueError):
    """Raised when probe evidence or diagnosis policy is ambiguous."""


def _duplicate_safe_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DiagnosisError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_json_constant(token: str):
    raise DiagnosisError(f"number must be finite: {token}")


def _strict_load(path: Path) -> dict:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_safe_pairs,
            parse_constant=_finite_json_constant,
        )
    except DiagnosisError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DiagnosisError(f"cannot load diagnosis policy {path}: {error}") from error
    if type(value) is not dict:
        raise DiagnosisError("diagnosis policy must be an object")
    return value


def _closed(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise DiagnosisError(f"{field} contains unknown fields: {', '.join(unknown)}")


def _required(value: Mapping[str, Any], required: set[str], field: str) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise DiagnosisError(f"{field} is missing required fields: {', '.join(missing)}")


def _object(value: Any, field: str) -> dict:
    if type(value) is not dict:
        raise DiagnosisError(f"{field} must be an object")
    return value


def _string(value: Any, field: str, *, maximum: int = 4096) -> str:
    if type(value) is not str or not value.strip():
        raise DiagnosisError(f"{field} must be a non-empty string")
    if len(value) > maximum:
        raise DiagnosisError(f"{field} exceeds {maximum} characters")
    return value


def _safe_id(value: Any, field: str) -> str:
    text = _string(value, field, maximum=128)
    if _SAFE_ID.fullmatch(text) is None:
        raise DiagnosisError(f"{field} must be a safe identifier")
    return text


def _percentage(value: Any, field: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DiagnosisError(f"{field} must be a percentage number")
    if not math.isfinite(float(value)) or not 0 <= value <= 100:
        raise DiagnosisError(f"{field} must be between 0 and 100")
    return value


def validate_probe(value: Mapping[str, Any]) -> dict:
    """Return a detached, closed normalized probe artifact."""
    probe = _object(value, "probe")
    fields = {
        "schema_version",
        "probe_id",
        "kind",
        "status",
        "metrics",
        "issues",
        "artifacts",
    }
    _closed(probe, fields, "probe")
    _required(probe, fields, "probe")
    if probe["schema_version"] != PROBE_SCHEMA:
        raise DiagnosisError(f"probe.schema_version must be {PROBE_SCHEMA}")
    _safe_id(probe["probe_id"], "probe.probe_id")
    if probe["kind"] not in PROBE_KINDS:
        raise DiagnosisError("probe.kind is unsupported")
    if probe["status"] not in {"ok", "degraded", "failed", "unavailable"}:
        raise DiagnosisError("probe.status is unsupported")

    metrics = _object(probe["metrics"], "probe.metrics")
    unknown_metrics = sorted(set(metrics) - METRICS)
    if unknown_metrics:
        raise DiagnosisError(
            "probe.metrics contains unknown fields: " + ", ".join(unknown_metrics)
        )
    for name, number in metrics.items():
        _percentage(number, f"probe.metrics.{name}")

    issues = probe["issues"]
    if type(issues) is not list:
        raise DiagnosisError("probe.issues must be an array")
    for index, item in enumerate(issues):
        issue = _object(item, f"probe.issues[{index}]")
        issue_fields = {"id", "category", "severity", "message"}
        _closed(issue, issue_fields, f"probe.issues[{index}]")
        _required(issue, issue_fields, f"probe.issues[{index}]")
        _safe_id(issue["id"], f"probe.issues[{index}].id")
        if issue["category"] not in CATEGORIES:
            raise DiagnosisError(f"probe.issues[{index}].category is unsupported")
        if issue["severity"] not in {"info", "warning", "error"}:
            raise DiagnosisError(f"probe.issues[{index}].severity is unsupported")
        _string(issue["message"], f"probe.issues[{index}].message")

    artifacts = probe["artifacts"]
    if type(artifacts) is not list:
        raise DiagnosisError("probe.artifacts must be an array")
    for index, item in enumerate(artifacts):
        artifact = _object(item, f"probe.artifacts[{index}]")
        artifact_fields = {"name", "sha256"}
        _closed(artifact, artifact_fields, f"probe.artifacts[{index}]")
        _required(artifact, artifact_fields, f"probe.artifacts[{index}]")
        name = _string(artifact["name"], f"probe.artifacts[{index}].name")
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise DiagnosisError(f"probe.artifacts[{index}].name must be run-relative")
        if type(artifact["sha256"]) is not str or _SHA256.fullmatch(
            artifact["sha256"]
        ) is None:
            raise DiagnosisError(f"probe.artifacts[{index}].sha256 is invalid")
    return copy.deepcopy(probe)


def _policy_digest(policy: Mapping[str, Any]) -> str:
    payload = json.dumps(
        policy, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_policy(path: str | Path) -> dict:
    """Load and validate a versioned deterministic diagnosis policy."""
    policy = _strict_load(Path(path))
    fields = {
        "schema_version",
        "mixed_score_delta",
        "environment_issue_score",
        "rules",
        "suggested_probes",
    }
    _closed(policy, fields, "policy")
    _required(policy, fields, "policy")
    if policy["schema_version"] != POLICY_SCHEMA:
        raise DiagnosisError(f"policy.schema_version must be {POLICY_SCHEMA}")
    for name in ("mixed_score_delta", "environment_issue_score"):
        value = policy[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DiagnosisError(f"policy.{name} must be numeric")
        if not math.isfinite(float(value)) or value < 0:
            raise DiagnosisError(f"policy.{name} must be finite and non-negative")
    if type(policy["rules"]) is not list or not policy["rules"]:
        raise DiagnosisError("policy.rules must be a non-empty array")
    rule_ids = set()
    for index, item in enumerate(policy["rules"]):
        rule = _object(item, f"policy.rules[{index}]")
        rule_fields = {"id", "category", "score", "all"}
        _closed(rule, rule_fields, f"policy.rules[{index}]")
        _required(rule, rule_fields, f"policy.rules[{index}]")
        rule_id = _safe_id(rule["id"], f"policy.rules[{index}].id")
        if rule_id in rule_ids:
            raise DiagnosisError("policy rule ids must be unique")
        rule_ids.add(rule_id)
        if rule["category"] not in CATEGORIES or rule["category"] == "environment":
            raise DiagnosisError(f"policy.rules[{index}].category is unsupported")
        score = rule["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise DiagnosisError(f"policy.rules[{index}].score must be numeric")
        if not math.isfinite(float(score)) or score <= 0:
            raise DiagnosisError(f"policy.rules[{index}].score must be positive")
        conditions = rule["all"]
        if type(conditions) is not list or not conditions:
            raise DiagnosisError(f"policy.rules[{index}].all must be non-empty")
        for condition_index, item in enumerate(conditions):
            condition = _object(
                item, f"policy.rules[{index}].all[{condition_index}]"
            )
            condition_fields = {"metric", "operator", "threshold"}
            _closed(
                condition,
                condition_fields,
                f"policy.rules[{index}].all[{condition_index}]",
            )
            _required(
                condition,
                condition_fields,
                f"policy.rules[{index}].all[{condition_index}]",
            )
            if condition["metric"] not in METRICS:
                raise DiagnosisError("policy condition metric is unsupported")
            if condition["operator"] not in {"gte", "lte"}:
                raise DiagnosisError("policy condition operator is unsupported")
            _percentage(condition["threshold"], "policy condition threshold")
    suggested = policy["suggested_probes"]
    if type(suggested) is not list or not suggested:
        raise DiagnosisError("policy.suggested_probes must be non-empty")
    for item in suggested:
        if item not in PROBE_KINDS:
            raise DiagnosisError("policy suggested probe is unsupported")
    return copy.deepcopy(policy)


def _condition_matches(value: float | int, operator: str, threshold: float) -> bool:
    return value >= threshold if operator == "gte" else value <= threshold


def diagnose(probes: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]) -> dict:
    """Classify verified probe evidence without model-generated facts."""
    validated_policy = copy.deepcopy(policy)
    # Validate in-memory policies through the same structural checks by applying
    # the checks that influence execution; load_policy remains the file boundary.
    if validated_policy.get("schema_version") != POLICY_SCHEMA:
        raise DiagnosisError(f"policy.schema_version must be {POLICY_SCHEMA}")
    if not isinstance(probes, Sequence) or isinstance(probes, (str, bytes)):
        raise DiagnosisError("probes must be a sequence")
    normalized = [validate_probe(item) for item in probes]
    if not normalized:
        raise DiagnosisError("at least one probe is required")

    metrics = {}
    evidence_paths = {}
    for probe in normalized:
        for name, value in probe["metrics"].items():
            if name not in metrics or value > metrics[name]:
                metrics[name] = value
                evidence_paths[name] = (
                    f"probes/{probe['probe_id']}.json#/metrics/{name}"
                )

    scores = {category: 0.0 for category in CATEGORIES}
    matches = []
    diagnosis_ids = []
    for rule in validated_policy["rules"]:
        conditions = []
        paths = []
        matched = True
        for condition in rule["all"]:
            name = condition["metric"]
            if name not in metrics or not _condition_matches(
                metrics[name], condition["operator"], condition["threshold"]
            ):
                matched = False
                break
            conditions.append(
                {
                    "metric": name,
                    "operator": condition["operator"],
                    "threshold": condition["threshold"],
                    "observed": metrics[name],
                }
            )
            paths.append(evidence_paths[name])
        if matched:
            scores[rule["category"]] += float(rule["score"])
            matches.append(
                {
                    "rule_id": rule["id"],
                    "category": rule["category"],
                    "score": rule["score"],
                    "conditions": conditions,
                    "evidence_paths": paths,
                }
            )
            diagnosis_ids.append(rule["id"])

    for probe in normalized:
        for index, issue in enumerate(probe["issues"]):
            if issue["severity"] != "error":
                continue
            score = float(validated_policy["environment_issue_score"])
            scores[issue["category"]] += score
            matches.append(
                {
                    "rule_id": issue["id"],
                    "category": issue["category"],
                    "score": score,
                    "conditions": [{"severity": "error", "message": issue["message"]}],
                    "evidence_paths": [
                        f"probes/{probe['probe_id']}.json#/issues/{index}"
                    ],
                }
            )
            diagnosis_ids.append(issue["id"])

    ranked = [
        {"category": category, "score": score}
        for category, score in scores.items()
        if score > 0
    ]
    ranked.sort(key=lambda item: (-item["score"], item["category"]))
    status = "inconclusive"
    primary = None
    confidence = "inconclusive"
    if ranked:
        primary = ranked[0]["category"]
        status = "classified"
        top = ranked[0]["score"]
        confidence = "high" if top >= 100 else "medium" if top >= 50 else "low"
        if (
            len(ranked) > 1
            and top - ranked[1]["score"]
            <= float(validated_policy["mixed_score_delta"])
        ):
            status = "mixed"
            primary = "mixed"
            confidence = "low"

    return {
        "schema_version": DIAGNOSIS_SCHEMA,
        "policy_schema_version": validated_policy["schema_version"],
        "policy_digest": _policy_digest(validated_policy),
        "status": status,
        "primary_category": primary,
        "confidence": confidence,
        "ranked_categories": ranked,
        "scores": scores,
        "matched_rules": matches,
        "diagnosis_ids": diagnosis_ids,
        "coverage": {
            "probe_count": len(normalized),
            "available_probe_count": sum(
                probe["status"] in {"ok", "degraded"} for probe in normalized
            ),
            "known_metrics": sorted(metrics),
        },
        "suggested_probes": (
            copy.deepcopy(validated_policy["suggested_probes"])
            if status == "inconclusive"
            else []
        ),
    }
