#!/usr/bin/env python3
"""Admit finite, epoch-bound competing performance hypotheses."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


HYPOTHESIS_SCHEMA = "cuda-optimizer/hypothesis-set-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_KINDS = {"mechanism", "unmodeled"}
_DISPOSITIONS = {"active", "rejected", "undifferentiable"}
_CONFIDENCE = {"inconclusive", "plausible", "direction_supported"}
_RELATIONS = {"exclusive", "depends_on", "coexists_with"}


class ValidationError(ValueError):
    """Raised when a hypothesis set can invent or misbind evidence."""


def _load_sibling(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load hypothesis dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXECUTION_MAP = _load_sibling(
    "execution_map.py", "cuda_optimizer_execution_map_hypothesis_runtime"
)
_EPOCH = _load_sibling("analysis_epoch.py", "cuda_optimizer_epoch_hypothesis_runtime")


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


def _text(value: Any, label: str, *, maximum: int = 1024) -> str:
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


def _identifier_list(
    value: Any, label: str, *, allow_empty: bool = False
) -> list[str]:
    if type(value) is not list or (not value and not allow_empty):
        qualifier = "an array" if allow_empty else "a non-empty array"
        raise ValidationError(f"{label} must be {qualifier}")
    result = [_identifier(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise ValidationError(f"{label} must not contain duplicates")
    return sorted(result)


def _catalog(value: Mapping[str, Any], epoch_id: str) -> dict[str, dict]:
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
        item_epoch = _identifier(item["epoch_id"], "evidence epoch_id")
        if item_epoch != epoch_id:
            raise ValidationError("evidence_catalog contains evidence outside the current epoch")
        result[evidence_id] = {
            "epoch_id": item_epoch,
            "kind": _identifier(item["kind"], "evidence kind"),
            "artifact_sha256": _sha(item["artifact_sha256"], "evidence artifact_sha256"),
        }
    return result


def _evidence_list(value: Any, label: str, catalog: Mapping[str, dict]) -> list[str]:
    result = _identifier_list(value, label, allow_empty=True)
    for evidence_id in result:
        if evidence_id not in catalog:
            raise ValidationError(f"{label} cites unknown evidence {evidence_id}")
    return result


def _has_cycle(vertices: set[str], edges: list[tuple[str, str]]) -> bool:
    adjacency = {item: [] for item in vertices}
    indegree = {item: 0 for item in vertices}
    for dependent, dependency in edges:
        adjacency[dependency].append(dependent)
        indegree[dependent] += 1
    ready = sorted(item for item, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        current = ready.pop(0)
        visited += 1
        for target in sorted(adjacency[current]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    return visited != len(vertices)


def _digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_hypothesis_set(
    value: Mapping[str, Any],
    *,
    epoch: Mapping[str, Any],
    execution_map: Mapping[str, Any],
    evidence_catalog: Mapping[str, Any],
) -> dict:
    """Replay map and evidence identities, then admit a canonical hypothesis graph."""
    try:
        normalized_epoch = _EPOCH.validate_epoch(epoch)
        map_result = _EXECUTION_MAP.validate_execution_map(
            execution_map, epoch=normalized_epoch, evidence_catalog=evidence_catalog
        )
        expected_map_digest = _EXECUTION_MAP.execution_map_digest(
            execution_map, epoch=normalized_epoch, evidence_catalog=evidence_catalog
        )
    except ValueError as exc:
        raise ValidationError(f"invalid Controller execution map: {exc}") from exc
    epoch_id = normalized_epoch["epoch_id"]
    catalog = _catalog(evidence_catalog, epoch_id)
    root = _closed(
        value,
        {
            "schema_version",
            "set_id",
            "epoch_id",
            "epoch_sha256",
            "execution_map_sha256",
            "hypotheses",
            "relationships",
        },
        "hypothesis_set",
    )
    if root["schema_version"] != HYPOTHESIS_SCHEMA:
        raise ValidationError(f"hypothesis_set.schema_version must be {HYPOTHESIS_SCHEMA}")
    _identifier(root["set_id"], "hypothesis_set.set_id")
    if root["epoch_id"] != epoch_id:
        raise ValidationError("hypothesis_set epoch_id does not match Controller")
    if _sha(root["epoch_sha256"], "hypothesis_set.epoch_sha256") != _EPOCH.epoch_digest(normalized_epoch):
        raise ValidationError("hypothesis_set epoch digest does not match Controller")
    if _sha(root["execution_map_sha256"], "hypothesis_set.execution_map_sha256") != expected_map_digest:
        raise ValidationError("hypothesis_set execution map digest does not match Controller")

    map_nodes = {
        item["node_id"] for item in map_result["execution_map"]["nodes"]
    }
    raw_hypotheses = root["hypotheses"]
    if type(raw_hypotheses) is not list or not 1 <= len(raw_hypotheses) <= 12:
        raise ValidationError("hypothesis_set must contain 1 to 12 hypotheses")
    hypotheses = []
    by_id = {}
    for index, raw in enumerate(raw_hypotheses):
        item = _closed(
            raw,
            {
                "hypothesis_id",
                "kind",
                "scope_node_ids",
                "statement",
                "mechanism",
                "disposition",
                "confidence",
                "support_evidence_ids",
                "oppose_evidence_ids",
                "missing_evidence_kinds",
                "falsification_question",
            },
            f"hypotheses[{index}]",
        )
        hypothesis_id = _identifier(item["hypothesis_id"], f"hypotheses[{index}].hypothesis_id")
        if hypothesis_id in by_id:
            raise ValidationError("hypothesis ids must be unique")
        if item["kind"] not in _KINDS:
            raise ValidationError(f"hypotheses[{index}].kind is unsupported")
        scope = _identifier_list(item["scope_node_ids"], f"hypotheses[{index}].scope_node_ids")
        if not set(scope).issubset(map_nodes):
            raise ValidationError("hypothesis scope references an unknown execution-map node")
        _text(item["statement"], f"hypotheses[{index}].statement")
        _identifier(item["mechanism"], f"hypotheses[{index}].mechanism")
        disposition = item["disposition"]
        confidence = item["confidence"]
        if disposition not in _DISPOSITIONS:
            raise ValidationError(f"hypotheses[{index}].disposition is unsupported")
        if confidence not in _CONFIDENCE:
            raise ValidationError(f"hypotheses[{index}].confidence is unsupported")
        support = _evidence_list(
            item["support_evidence_ids"], f"hypotheses[{index}].support_evidence_ids", catalog
        )
        oppose = _evidence_list(
            item["oppose_evidence_ids"], f"hypotheses[{index}].oppose_evidence_ids", catalog
        )
        if set(support) & set(oppose):
            raise ValidationError("the same evidence cannot support and oppose a hypothesis")
        missing = _identifier_list(
            item["missing_evidence_kinds"],
            f"hypotheses[{index}].missing_evidence_kinds",
            allow_empty=True,
        )
        _text(item["falsification_question"], f"hypotheses[{index}].falsification_question")
        evidence_kinds = {catalog[evidence_id]["kind"] for evidence_id in support}
        if confidence == "plausible" and not support:
            raise ValidationError("plausible hypothesis requires supporting evidence")
        if confidence == "direction_supported":
            if normalized_epoch["boundary_ambiguous"]:
                raise ValidationError("ambiguous epoch cannot support a direction")
            if len(evidence_kinds) < 2:
                raise ValidationError("direction requires two independent evidence kinds")
        if item["kind"] == "unmodeled" and confidence != "inconclusive":
            raise ValidationError("unmodeled hypothesis must remain inconclusive")
        if disposition == "rejected":
            if not oppose:
                raise ValidationError("rejected hypothesis requires opposing evidence")
            if confidence != "inconclusive":
                raise ValidationError("rejected hypothesis must be inconclusive")
        if disposition == "undifferentiable" and confidence != "inconclusive":
            raise ValidationError("undifferentiable hypothesis must be inconclusive")
        normalized = {
            **copy.deepcopy(dict(item)),
            "scope_node_ids": scope,
            "support_evidence_ids": support,
            "oppose_evidence_ids": oppose,
            "missing_evidence_kinds": missing,
        }
        hypotheses.append(normalized)
        by_id[hypothesis_id] = normalized
    hypotheses.sort(key=lambda item: item["hypothesis_id"])

    if map_result["requires_unmodeled_hypothesis"] and not any(
        item["kind"] == "unmodeled" and item["disposition"] == "active"
        for item in hypotheses
    ):
        raise ValidationError("execution-map gap requires an active unmodeled hypothesis")

    raw_relationships = root["relationships"]
    if type(raw_relationships) is not list or len(raw_relationships) > 66:
        raise ValidationError("relationships must be an array of at most 66 entries")
    relationships = []
    symmetric_pairs = {}
    dependency_edges = []
    relationship_keys = set()
    for index, raw in enumerate(raw_relationships):
        item = _closed(raw, {"relation", "left", "right"}, f"relationships[{index}]")
        relation = item["relation"]
        if relation not in _RELATIONS:
            raise ValidationError(f"relationships[{index}].relation is unsupported")
        left = _identifier(item["left"], f"relationships[{index}].left")
        right = _identifier(item["right"], f"relationships[{index}].right")
        if left not in by_id or right not in by_id:
            raise ValidationError("relationship references an unknown hypothesis")
        if left == right:
            raise ValidationError("relationship must connect distinct hypotheses")
        if relation in {"exclusive", "coexists_with"}:
            if left >= right:
                raise ValidationError("symmetric relationship pair must use canonical order")
            pair = (left, right)
            prior = symmetric_pairs.get(pair)
            if prior is not None and prior != relation:
                raise ValidationError("hypothesis pair has conflicting symmetric relationships")
            symmetric_pairs[pair] = relation
            if relation == "exclusive" and not (
                set(by_id[left]["scope_node_ids"]) & set(by_id[right]["scope_node_ids"])
            ):
                raise ValidationError("exclusive hypotheses must overlap in scope")
        else:
            dependency_edges.append((left, right))
        key = (relation, left, right)
        if key in relationship_keys:
            raise ValidationError("relationships must be unique")
        relationship_keys.add(key)
        relationships.append(dict(item))
    if _has_cycle(set(by_id), dependency_edges):
        raise ValidationError("depends_on relationship graph contains a cycle")
    relationships.sort(key=lambda item: (item["relation"], item["left"], item["right"]))

    normalized_set = {
        "schema_version": HYPOTHESIS_SCHEMA,
        "set_id": root["set_id"],
        "epoch_id": root["epoch_id"],
        "epoch_sha256": root["epoch_sha256"],
        "execution_map_sha256": root["execution_map_sha256"],
        "hypotheses": hypotheses,
        "relationships": relationships,
    }
    return {
        "hypothesis_set": normalized_set,
        "hypothesis_set_sha256": _digest(normalized_set),
        "active_hypothesis_ids": sorted(
            item["hypothesis_id"]
            for item in hypotheses
            if item["disposition"] == "active"
        ),
    }
