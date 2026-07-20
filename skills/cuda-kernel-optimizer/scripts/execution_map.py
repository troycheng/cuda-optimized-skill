#!/usr/bin/env python3
"""Validate compact, epoch-bound V3.1 workload execution maps."""

from __future__ import annotations

import copy
import importlib.util
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


MAP_SCHEMA = "cuda-optimizer/execution-map-v1"
LAYERS = (
    "cpu",
    "gpu",
    "framework",
    "transfer",
    "communication",
    "io",
    "synchronization",
    "idle",
)
_RELATIONS = {
    "calls",
    "waits_for",
    "transfers_to",
    "synchronizes",
    "precedes",
    "unknown_dependency",
}
_ATTRIBUTION = {"explained", "unexplained", "not_applicable"}
_COVERAGE = {"observed", "not_observed", "unavailable"}
_CONCLUSIONS = {"observed", "inconclusive"}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


class ValidationError(ValueError):
    """Raised when an execution map can hide missing or stale evidence."""


def _load_epoch_module():
    path = Path(__file__).with_name("analysis_epoch.py")
    name = "cuda_optimizer_analysis_epoch_execution_map"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load epoch validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EPOCH = _load_epoch_module()


def epoch_digest(value: Mapping[str, Any]) -> str:
    return _EPOCH.epoch_digest(value)


def _closed(value: Any, fields: set[str], label: str) -> dict:
    if type(value) is not dict:
        raise ValidationError(f"{label} must be an object")
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing:
        raise ValidationError(f"{label} is missing fields: {sorted(missing)}")
    if unknown:
        raise ValidationError(f"{label} contains unknown fields: {sorted(unknown)}")
    return value


def _text(value: Any, label: str, *, maximum: int = 512) -> str:
    if type(value) is not str or not value.strip():
        raise ValidationError(f"{label} must be a non-empty string")
    if len(value) > maximum:
        raise ValidationError(f"{label} exceeds {maximum} characters")
    return value


def _identifier(value: Any, label: str) -> str:
    text = _text(value, label, maximum=128)
    if _IDENTIFIER.fullmatch(text) is None:
        raise ValidationError(f"{label} must be a safe identifier")
    return text


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be lowercase SHA-256")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValidationError(f"{label} must be finite")
    number = float(value)
    if number < 0 or (positive and number <= 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValidationError(f"{label} must be {qualifier}")
    return number


def _evidence_ids(value: Any, label: str, catalog: Mapping[str, dict], epoch_id: str) -> list[str]:
    if type(value) is not list or not value:
        raise ValidationError(f"{label} must be a non-empty array")
    result = []
    for index, raw in enumerate(value):
        evidence_id = _identifier(raw, f"{label}[{index}]")
        if evidence_id in result:
            raise ValidationError(f"{label} must not contain duplicates")
        if evidence_id not in catalog:
            raise ValidationError(f"{label} cites unknown evidence {evidence_id}")
        if catalog[evidence_id]["epoch_id"] != epoch_id:
            raise ValidationError(f"{label} evidence is not from the current epoch")
        result.append(evidence_id)
    return result


def _catalog(value: Mapping[str, Any]) -> dict[str, dict]:
    if not isinstance(value, Mapping):
        raise ValidationError("evidence_catalog must be an object")
    result = {}
    for raw_id, raw in value.items():
        evidence_id = _identifier(raw_id, "evidence_catalog id")
        item = _closed(
            raw,
            {"epoch_id", "kind", "artifact_sha256"},
            f"evidence_catalog.{evidence_id}",
        )
        result[evidence_id] = {
            "epoch_id": _identifier(item["epoch_id"], "evidence epoch_id"),
            "kind": _identifier(item["kind"], "evidence kind"),
            "artifact_sha256": _sha(item["artifact_sha256"], "evidence artifact_sha256"),
        }
    return result


def validate_execution_map(
    value: Mapping[str, Any], *, epoch: Mapping[str, Any], evidence_catalog: Mapping[str, Any]
) -> dict:
    """Return a detached map plus Controller-derived coverage flags."""
    try:
        normalized_epoch = _EPOCH.validate_epoch(epoch)
    except ValueError as exc:
        raise ValidationError(f"invalid Controller epoch: {exc}") from exc
    epoch_id = normalized_epoch["epoch_id"]
    catalog = _catalog(evidence_catalog)
    result = _closed(
        value,
        {
            "schema_version",
            "map_id",
            "epoch_id",
            "epoch_sha256",
            "identities",
            "window",
            "coverage",
            "nodes",
            "edges",
            "hot_path",
            "uncovered_intervals",
            "conclusion_level",
        },
        "execution_map",
    )
    if result["schema_version"] != MAP_SCHEMA:
        raise ValidationError(f"execution_map.schema_version must be {MAP_SCHEMA}")
    _identifier(result["map_id"], "execution_map.map_id")
    if result["epoch_id"] != epoch_id:
        raise ValidationError("execution_map epoch_id does not match Controller")
    if _sha(result["epoch_sha256"], "execution_map.epoch_sha256") != epoch_digest(normalized_epoch):
        raise ValidationError("execution_map epoch digest does not match Controller")
    identities = _closed(
        result["identities"],
        set(normalized_epoch["identities"]),
        "execution_map.identities",
    )
    for field, expected in normalized_epoch["identities"].items():
        if _sha(identities[field], f"execution_map.identities.{field}") != expected:
            label = field.removesuffix("_sha256")
            raise ValidationError(f"execution_map {label} identity does not match Controller")

    window = _closed(
        result["window"],
        {"start_us", "end_us", "boundary_ambiguous"},
        "execution_map.window",
    )
    start = _number(window["start_us"], "execution_map.window.start_us")
    end = _number(window["end_us"], "execution_map.window.end_us")
    if end <= start:
        raise ValidationError("execution_map window end must be after start")
    if type(window["boundary_ambiguous"]) is not bool:
        raise ValidationError("execution_map window boundary_ambiguous must be boolean")
    if window["boundary_ambiguous"] != normalized_epoch["boundary_ambiguous"]:
        raise ValidationError("execution_map boundary does not match Controller epoch")

    coverage = result["coverage"]
    if type(coverage) is not list or len(coverage) != len(LAYERS):
        raise ValidationError("execution_map coverage must contain all layers")
    coverage_by_layer = {}
    for index, raw in enumerate(coverage):
        item = _closed(raw, {"layer", "status", "reason"}, f"coverage[{index}]")
        layer = item["layer"]
        if layer not in LAYERS or layer in coverage_by_layer:
            raise ValidationError("execution_map coverage must contain all layers exactly once")
        if item["status"] not in _COVERAGE:
            raise ValidationError(f"coverage[{index}].status is unsupported")
        if item["status"] == "observed":
            if item["reason"] is not None:
                raise ValidationError("observed coverage reason must be null")
        else:
            _text(item["reason"], f"coverage[{index}].reason")
        coverage_by_layer[layer] = item["status"]
    if set(coverage_by_layer) != set(LAYERS):
        raise ValidationError("execution_map coverage must contain all layers")

    nodes = result["nodes"]
    if type(nodes) is not list or not nodes or len(nodes) > 256:
        raise ValidationError("execution_map nodes must contain 1 to 256 entries")
    node_ids = set()
    node_layers = {layer: 0 for layer in LAYERS}
    unexplained = False
    for index, raw in enumerate(nodes):
        node = _closed(
            raw,
            {
                "node_id",
                "layer",
                "lane",
                "kind",
                "label",
                "duration_us",
                "occurrences",
                "attribution_status",
                "evidence_ids",
            },
            f"nodes[{index}]",
        )
        node_id = _identifier(node["node_id"], f"nodes[{index}].node_id")
        if node_id in node_ids:
            raise ValidationError("execution_map node ids must be unique")
        node_ids.add(node_id)
        layer = node["layer"]
        if layer not in LAYERS:
            raise ValidationError(f"nodes[{index}].layer is unsupported")
        if coverage_by_layer[layer] != "observed":
            raise ValidationError(f"node uses {coverage_by_layer[layer]} layer {layer}")
        node_layers[layer] += 1
        _identifier(node["lane"], f"nodes[{index}].lane")
        _identifier(node["kind"], f"nodes[{index}].kind")
        _text(node["label"], f"nodes[{index}].label")
        _number(node["duration_us"], f"nodes[{index}].duration_us", positive=True)
        if type(node["occurrences"]) is not int or node["occurrences"] < 1:
            raise ValidationError(f"nodes[{index}].occurrences must be positive")
        if node["attribution_status"] not in _ATTRIBUTION:
            raise ValidationError(f"nodes[{index}].attribution_status is unsupported")
        unexplained = unexplained or node["attribution_status"] == "unexplained"
        _evidence_ids(node["evidence_ids"], f"nodes[{index}].evidence_ids", catalog, epoch_id)
    for layer, status in coverage_by_layer.items():
        if status == "observed" and node_layers[layer] == 0:
            raise ValidationError(f"observed layer {layer} requires at least one node")

    edges = result["edges"]
    if type(edges) is not list or len(edges) > 1024:
        raise ValidationError("execution_map edges must be an array of at most 1024 entries")
    edge_keys = set()
    for index, raw in enumerate(edges):
        edge = _closed(raw, {"source", "target", "relation", "evidence_ids"}, f"edges[{index}]")
        source = _identifier(edge["source"], f"edges[{index}].source")
        target = _identifier(edge["target"], f"edges[{index}].target")
        if source not in node_ids or target not in node_ids:
            raise ValidationError("execution_map edge references an unknown node")
        if source == target:
            raise ValidationError("execution_map edge must connect distinct nodes")
        relation = edge["relation"]
        if relation not in _RELATIONS:
            raise ValidationError(f"edges[{index}].relation is unsupported")
        key = (source, target, relation)
        if key in edge_keys:
            raise ValidationError("execution_map edges must be unique")
        edge_keys.add(key)
        unexplained = unexplained or relation == "unknown_dependency"
        _evidence_ids(edge["evidence_ids"], f"edges[{index}].evidence_ids", catalog, epoch_id)

    hot_path = result["hot_path"]
    if type(hot_path) is not list or not hot_path:
        raise ValidationError("execution_map hot_path must be non-empty")
    seen_hot = set()
    for index, raw in enumerate(hot_path):
        node_id = _identifier(raw, f"hot_path[{index}]")
        if node_id not in node_ids:
            raise ValidationError("execution_map hot_path references an unknown node")
        if node_id in seen_hot:
            raise ValidationError("execution_map hot_path must not contain duplicates")
        seen_hot.add(node_id)

    intervals = result["uncovered_intervals"]
    if type(intervals) is not list or len(intervals) > 128:
        raise ValidationError("uncovered_intervals must be an array of at most 128 entries")
    for index, raw in enumerate(intervals):
        interval = _closed(raw, {"start_us", "end_us", "reason"}, f"uncovered_intervals[{index}]")
        interval_start = _number(interval["start_us"], f"uncovered_intervals[{index}].start_us")
        interval_end = _number(interval["end_us"], f"uncovered_intervals[{index}].end_us")
        if interval_start < start or interval_end > end or interval_end <= interval_start:
            raise ValidationError("uncovered interval must be positive and inside the window")
        _text(interval["reason"], f"uncovered_intervals[{index}].reason")
    if result["conclusion_level"] not in _CONCLUSIONS:
        raise ValidationError("execution_map conclusion_level is unsupported")

    requires_unmodeled = (
        unexplained
        or bool(intervals)
        or any(status == "unavailable" for status in coverage_by_layer.values())
        or bool(window["boundary_ambiguous"])
    )
    return {
        "execution_map": copy.deepcopy(dict(result)),
        "window_duration_us": end - start,
        "requires_unmodeled_hypothesis": requires_unmodeled,
    }
