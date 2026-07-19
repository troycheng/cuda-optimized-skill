#!/usr/bin/env python3
"""Select a bounded set of hash-bound GPU optimization playbooks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SKILL_ROOT = Path(__file__).resolve().parent.parent
CAPABILITY_ROOT = SKILL_ROOT / "references" / "capabilities"
REGISTRY_PATH = CAPABILITY_ROOT / "registry.json"
SHUFFLED_REGISTRY_PATH = CAPABILITY_ROOT / "registry.shuffled.json"
SOURCES_PATH = CAPABILITY_ROOT / "sources.json"

_CAPABILITY_FIELDS = {
    "id",
    "version",
    "status",
    "task",
    "layer",
    "axes",
    "architectures",
    "required_features",
    "frameworks",
    "signal_groups",
    "counter_signals_any",
    "requires_evidence",
    "gate_requirements",
    "contract_binding_required",
    "conflicts",
    "context_cost_bytes",
    "playbook",
    "playbook_sha256",
    "source_ids",
    "last_reviewed",
    "risk",
    "method_ids",
}
_SOURCE_FIELDS = {
    "id",
    "title",
    "url",
    "kind",
    "license",
    "last_reviewed",
    "commit_sha",
}
_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){0,3}$")
_CAPABILITY_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){1,3}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_PRE_EXECUTION_GATES = frozenset(
    {"correctness_reference", "dispatch_identity", "target_compile_probe"}
)
_PROMOTION_GATES = frozenset(
    {"candidate_correctness", "paired_measurement", "workload_replay"}
)


def _reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _safe_root(root: Path) -> Path:
    root = Path(os.path.abspath(root))
    try:
        mode = root.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"missing capability root: {root}") from exc
    if stat.S_ISLNK(mode):
        raise ValueError(f"capability root must not be a symlink: {root}")
    if not stat.S_ISDIR(mode):
        raise ValueError(f"capability root must be a directory: {root}")
    return root


def _read_regular_snapshot(path: Path, trusted_root: Path) -> Tuple[bytes, str]:
    """Read one stable snapshot through descriptor-relative, no-follow opens."""
    root = _safe_root(trusted_root)
    target = Path(os.path.abspath(path))
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes capability root: {path}") from exc
    if not relative.parts:
        raise ValueError(f"expected a regular file: {path}")
    common_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    directory_flags = common_flags | getattr(os, "O_DIRECTORY", 0)
    descriptors: List[int] = []
    try:
        parent = os.open(root, directory_flags)
        descriptors.append(parent)
        for part in relative.parts[:-1]:
            parent = os.open(part, directory_flags, dir_fd=parent)
            descriptors.append(parent)
        descriptor = os.open(relative.parts[-1], common_flags, dir_fd=parent)
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"capability path must be a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError(
            f"capability path contains a symlink or unsafe component: {path}"
        ) from exc
    finally:
        for opened in reversed(descriptors):
            os.close(opened)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ValueError(f"capability file changed while reading: {path}")
    payload = b"".join(chunks)
    if len(payload) != after.st_size:
        raise ValueError(f"capability file size changed while reading: {path}")
    return payload, hashlib.sha256(payload).hexdigest()


def _read_json_snapshot(path: Path, trusted_root: Path) -> Tuple[Dict[str, Any], str]:
    raw, digest = _read_regular_snapshot(path, trusted_root)
    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload, digest


def _closed_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing:
        raise ValueError(f"{label} missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"{label} has unknown fields: {sorted(extra)}")


def _canonical_digest(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _strings(value: Any, label: str, *, allow_empty: bool = True) -> List[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ValueError(f"{label} must be a{' non-empty' if not allow_empty else ''} list")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must not contain duplicates")
    return value


def _parse_date(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date") from exc


def _version_tuple(value: str) -> Tuple[int, ...]:
    if not isinstance(value, str) or not _VERSION_RE.fullmatch(value):
        raise ValueError(f"unsupported framework version: {value!r}")
    return tuple(int(part) for part in value.split("."))


def _version_in_range(value: str, rule: Mapping[str, Any]) -> bool:
    if set(rule) != {"min_inclusive", "max_exclusive"}:
        raise ValueError("framework range must contain min_inclusive and max_exclusive")
    current = _version_tuple(value)
    minimum = _version_tuple(rule["min_inclusive"])
    maximum = _version_tuple(rule["max_exclusive"])
    width = max(len(current), len(minimum), len(maximum))
    pad = lambda item: item + (0,) * (width - len(item))
    if pad(minimum) >= pad(maximum):
        raise ValueError("framework version range must have min_inclusive < max_exclusive")
    return pad(minimum) <= pad(current) < pad(maximum)


def _read_playbook_snapshot(
    root: Path, relative: str, trusted_root: Optional[Path] = None
) -> Tuple[bytes, str]:
    if (
        not isinstance(relative, str)
        or not relative
        or not relative.endswith(".md")
        or Path(relative).is_absolute()
    ):
        raise ValueError(f"unsafe playbook path: {relative!r}")
    if ".." in Path(relative).parts:
        raise ValueError(f"unsafe playbook path: {relative!r}")
    return _read_regular_snapshot(Path(root) / relative, trusted_root or root)


def _validate_sources(payload: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    if set(payload) != {"schema_version", "sources"}:
        raise ValueError("sources manifest must be closed")
    if payload.get("schema_version") != "cuda-optimizer/capability-sources-v1":
        raise ValueError("unsupported capability sources schema")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list")
    by_id: Dict[str, Mapping[str, Any]] = {}
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"source[{index}] must be an object")
        allowed = set(_SOURCE_FIELDS)
        required = allowed - {"commit_sha"}
        missing = required - set(source)
        extra = set(source) - allowed
        if missing or extra:
            raise ValueError(f"invalid source[{index}] fields")
        source_id = source["id"]
        if not isinstance(source_id, str) or not _ID_RE.fullmatch(source_id):
            raise ValueError(f"invalid source id: {source_id!r}")
        if source_id in by_id:
            raise ValueError(f"duplicate source id: {source_id}")
        for field in ("title", "url", "kind", "license"):
            if not isinstance(source[field], str) or not source[field]:
                raise ValueError(f"source {source_id} has invalid {field}")
        _parse_date(source["last_reviewed"], f"source {source_id} last_reviewed")
        if "commit_sha" in source and not re.fullmatch(r"[0-9a-f]{40}", source["commit_sha"]):
            raise ValueError(f"source {source_id} has invalid commit_sha")
        by_id[source_id] = source
    return by_id


def _validate_capability(
    capability: Mapping[str, Any],
    index: int,
    known_architectures: Mapping[str, Any],
    source_ids: set[str],
    capability_root: Path,
    trusted_root: Optional[Path] = None,
) -> None:
    _closed_fields(capability, _CAPABILITY_FIELDS, f"capability[{index}]")
    capability_id = capability["id"]
    if not isinstance(capability_id, str) or not _ID_RE.fullmatch(capability_id):
        raise ValueError(f"invalid capability id: {capability_id!r}")
    if not isinstance(capability["version"], str) or not _CAPABILITY_VERSION_RE.fullmatch(
        capability["version"]
    ):
        raise ValueError(f"invalid capability version for {capability_id}")
    if capability["status"] not in {"experimental", "verified", "deprecated"}:
        raise ValueError(f"invalid status for {capability_id}")
    for field in ("task", "layer"):
        if not isinstance(capability[field], str) or not capability[field]:
            raise ValueError(f"invalid {field} for {capability_id}")
    if capability["risk"] not in {"low", "medium", "high"}:
        raise ValueError(f"invalid risk for {capability_id}")
    for field in (
        "axes",
        "architectures",
        "required_features",
        "counter_signals_any",
        "requires_evidence",
        "conflicts",
        "source_ids",
        "method_ids",
    ):
        _strings(
            capability[field],
            f"{capability_id}.{field}",
            allow_empty=field
            not in {"axes", "architectures", "source_ids"},
        )
    signal_groups = capability["signal_groups"]
    if not isinstance(signal_groups, list) or not signal_groups:
        raise ValueError(f"{capability_id}.signal_groups must be a non-empty list")
    normalized_groups = []
    for group_index, group in enumerate(signal_groups):
        normalized_groups.append(
            tuple(
                _strings(
                    group,
                    f"{capability_id}.signal_groups[{group_index}]",
                    allow_empty=False,
                )
            )
        )
    if len(normalized_groups) != len(set(normalized_groups)):
        raise ValueError(f"{capability_id}.signal_groups must not contain duplicates")
    gates = capability["gate_requirements"]
    if not isinstance(gates, dict) or set(gates) != {"pre_execution", "promotion"}:
        raise ValueError(
            f"{capability_id}.gate_requirements must contain pre_execution and promotion"
        )
    expected_gates = {
        "pre_execution": _PRE_EXECUTION_GATES,
        "promotion": _PROMOTION_GATES,
    }
    for phase, expected in expected_gates.items():
        actual = set(
            _strings(
                gates[phase],
                f"{capability_id}.gate_requirements.{phase}",
                allow_empty=False,
            )
        )
        if actual != expected:
            raise ValueError(
                f"invalid gate set for {capability_id}.{phase}: "
                f"expected={sorted(expected)} actual={sorted(actual)}"
            )
    if capability["contract_binding_required"] is not True:
        raise ValueError(f"{capability_id} must require contract binding")
    unknown_arches = set(capability["architectures"]) - set(known_architectures)
    if unknown_arches:
        raise ValueError(f"{capability_id} uses unknown architectures: {sorted(unknown_arches)}")
    for arch in capability["architectures"]:
        missing_features = set(capability["required_features"]) - set(known_architectures[arch])
        if missing_features:
            raise ValueError(f"{capability_id} requires unavailable features on {arch}")
    frameworks = capability["frameworks"]
    if not isinstance(frameworks, dict):
        raise ValueError(f"{capability_id}.frameworks must be an object")
    for framework, rule in frameworks.items():
        if not isinstance(framework, str) or not framework or not isinstance(rule, dict):
            raise ValueError(f"invalid framework rule for {capability_id}")
        _version_in_range(rule["min_inclusive"], rule)
    cost = capability["context_cost_bytes"]
    if isinstance(cost, bool) or not isinstance(cost, int) or cost < 1:
        raise ValueError(f"invalid context cost for {capability_id}")
    if not _HASH_RE.fullmatch(str(capability["playbook_sha256"])):
        raise ValueError(f"invalid playbook hash for {capability_id}")
    playbook_bytes, playbook_digest = _read_playbook_snapshot(
        capability_root, capability["playbook"], trusted_root
    )
    if playbook_digest != capability["playbook_sha256"]:
        raise ValueError(f"playbook hash mismatch for {capability_id}")
    if capability["context_cost_bytes"] != len(playbook_bytes):
        raise ValueError(
            f"context byte cost mismatch for {capability_id}: "
            f"declared={capability['context_cost_bytes']} actual={len(playbook_bytes)}"
        )
    unknown_sources = set(capability["source_ids"]) - source_ids
    if unknown_sources:
        raise ValueError(f"{capability_id} has unknown sources: {sorted(unknown_sources)}")
    _parse_date(capability["last_reviewed"], f"{capability_id}.last_reviewed")


def _validated_registry_snapshot(
    registry_path: Path = REGISTRY_PATH,
    sources_path: Path = SOURCES_PATH,
    capability_root: Path = CAPABILITY_ROOT,
    trusted_root: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Mapping[str, Any]]]:
    registry_path = Path(registry_path)
    sources_path = Path(sources_path)
    capability_root = Path(os.path.abspath(capability_root))
    trust = _safe_root(trusted_root or capability_root)
    sources_payload, sources_digest = _read_json_snapshot(sources_path, trust)
    registry, registry_digest = _read_json_snapshot(registry_path, trust)
    sources = _validate_sources(sources_payload)
    if set(registry) != {"schema_version", "known_architectures", "capabilities"}:
        raise ValueError("capability registry must be closed")
    if registry.get("schema_version") != "cuda-optimizer/capability-registry-v1":
        raise ValueError("unsupported capability registry schema")
    known = registry.get("known_architectures")
    if not isinstance(known, dict) or not known:
        raise ValueError("known_architectures must be a non-empty object")
    for arch, features in known.items():
        if not isinstance(arch, str) or not re.fullmatch(r"sm_[0-9]+", arch):
            raise ValueError(f"invalid architecture: {arch!r}")
        _strings(features, f"known_architectures.{arch}")
    capabilities = registry.get("capabilities")
    if not isinstance(capabilities, list):
        raise ValueError("capabilities must be a list")
    seen: set[str] = set()
    for index, capability in enumerate(capabilities):
        if not isinstance(capability, dict):
            raise ValueError(f"capability[{index}] must be an object")
        _validate_capability(
            capability,
            index,
            known,
            set(sources),
            capability_root,
            trust,
        )
        if capability["id"] in seen:
            raise ValueError(f"duplicate capability id: {capability['id']}")
        seen.add(capability["id"])
    validation = {
        "schema_version": "cuda-optimizer/capability-validation-v1",
        "status": "PASS",
        "registry_sha256": registry_digest,
        "sources_sha256": sources_digest,
        "capability_count": len(capabilities),
        "source_count": len(sources),
    }
    return validation, registry, sources


def validate_registry(
    registry_path: Path = REGISTRY_PATH,
    sources_path: Path = SOURCES_PATH,
    capability_root: Path = CAPABILITY_ROOT,
    trusted_root: Optional[Path] = None,
) -> Dict[str, Any]:
    validation, _, _ = _validated_registry_snapshot(
        registry_path=registry_path,
        sources_path=sources_path,
        capability_root=capability_root,
        trusted_root=trusted_root,
    )
    return validation


def _normal_set(values: Iterable[str], label: str) -> set[str]:
    result = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must contain non-empty strings")
        result.add(value.strip().lower())
    return result


def query(
    *,
    arch: str,
    task: str,
    layer: str,
    signals: Sequence[str],
    available_evidence: Sequence[str],
    framework_versions: Mapping[str, str],
    context_budget_bytes: int = 12000,
    limit: int = 3,
    as_of: Optional[str] = None,
    max_review_age_days: int = 365,
    registry_variant: str = "real",
    allow_ablation: bool = False,
) -> Dict[str, Any]:
    if registry_variant not in {"real", "shuffled"}:
        raise ValueError("registry_variant must be real or shuffled")
    if registry_variant == "shuffled" and not allow_ablation:
        raise ValueError("shuffled registry is an ablation fixture; set allow_ablation")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10")
    if (
        isinstance(context_budget_bytes, bool)
        or not isinstance(context_budget_bytes, int)
        or context_budget_bytes < 1
    ):
        raise ValueError("context_budget_bytes must be a positive integer")
    if isinstance(max_review_age_days, bool) or not isinstance(max_review_age_days, int) or max_review_age_days < 1:
        raise ValueError("max_review_age_days must be a positive integer")
    if not isinstance(framework_versions, Mapping):
        raise ValueError("framework_versions must be an object")
    for framework, version in framework_versions.items():
        if not isinstance(framework, str) or not framework:
            raise ValueError("framework names must be non-empty strings")
        _version_tuple(version)

    registry_path = REGISTRY_PATH if registry_variant == "real" else SHUFFLED_REGISTRY_PATH
    validation, registry, sources = _validated_registry_snapshot(
        registry_path=registry_path,
        trusted_root=SKILL_ROOT,
    )
    known = registry["known_architectures"]
    if arch not in known:
        raise ValueError(f"Unknown architecture {arch}; exact capability data is required")

    observed_signals = _normal_set(signals, "signals")
    evidence = _normal_set(available_evidence, "available_evidence")
    today = _parse_date(as_of or date.today().isoformat(), "as_of")
    ranked: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
    rejected: List[Dict[str, Any]] = []

    def reject(capability_id: str, reason: str) -> None:
        rejected.append({"id": capability_id, "reason": reason})

    for capability in registry["capabilities"]:
        capability_id = capability["id"]
        if capability["task"] != task or capability["layer"] != layer:
            continue
        if arch not in capability["architectures"]:
            reject(capability_id, "architecture_mismatch")
            continue
        if not set(capability["required_features"]).issubset(set(known[arch])):
            reject(capability_id, "feature_mismatch")
            continue
        framework_mismatch = any(
            framework not in framework_versions
            or not _version_in_range(framework_versions[framework], rule)
            for framework, rule in capability["frameworks"].items()
        )
        if framework_mismatch:
            reject(capability_id, "framework_version_mismatch")
            continue
        counter = observed_signals & _normal_set(
            capability["counter_signals_any"], "counter_signals_any"
        )
        if counter:
            reject(capability_id, "counter_signal_hit")
            continue
        matched_groups = [
            sorted(group)
            for group in (
                _normal_set(values, "signal_groups")
                for values in capability["signal_groups"]
            )
            if group.issubset(observed_signals)
        ]
        if not matched_groups:
            reject(capability_id, "no_complete_signal_group")
            continue
        positive = set().union(*(set(group) for group in matched_groups))
        missing = sorted(
            _normal_set(capability["requires_evidence"], "requires_evidence") - evidence
        )
        review_dates = [
            _parse_date(capability["last_reviewed"], "last_reviewed"),
            *(
                _parse_date(
                    sources[source_id]["last_reviewed"],
                    f"source {source_id} last_reviewed",
                )
                for source_id in capability["source_ids"]
            ),
        ]
        future_knowledge = any(review_date > today for review_date in review_dates)
        stale = any(
            (today - review_date).days > max_review_age_days
            for review_date in review_dates
            if review_date <= today
        )
        if future_knowledge:
            knowledge_status = "unverified_future"
        elif stale:
            knowledge_status = "unverified_stale"
        else:
            knowledge_status = capability["status"]
        item = {
            "id": capability_id,
            "version": capability["version"],
            "status": capability["status"],
            "task": capability["task"],
            "layer": capability["layer"],
            "axes": capability["axes"],
            "matched_signals": sorted(positive),
            "matched_signal_groups": matched_groups,
            "missing_evidence": missing,
            "retrieval_status": "needs_evidence" if missing else "ready",
            "knowledge_status": knowledge_status,
            "knowledge_time_violation": future_knowledge,
            "gate_requirements": capability["gate_requirements"],
            "contract_binding_required": capability["contract_binding_required"],
            "risk": capability["risk"],
            "conflicts": capability["conflicts"],
            "method_ids": capability["method_ids"],
            "context_cost_bytes": capability["context_cost_bytes"],
            "playbook": capability["playbook"],
            "playbook_sha256": capability["playbook_sha256"],
            "source_ids": capability["source_ids"],
            "last_reviewed": capability["last_reviewed"],
        }
        rank = (
            1 if missing else 0,
            1 if (stale or future_knowledge) else 0,
            -len(positive),
            capability["context_cost_bytes"],
            capability_id,
        )
        ranked.append((rank, item))

    selected: List[Dict[str, Any]] = []
    consumed = 0
    for _, item in sorted(ranked, key=lambda pair: pair[0]):
        if len(selected) >= limit:
            reject(item["id"], "limit_exceeded")
            continue
        if consumed + item["context_cost_bytes"] > context_budget_bytes:
            reject(item["id"], "context_budget_exceeded")
            continue
        selected.append(item)
        consumed += item["context_cost_bytes"]

    result = {
        "schema_version": "cuda-optimizer/capability-query-v1",
        "registry_variant": registry_variant,
        "registry_sha256": validation["registry_sha256"],
        "sources_sha256": validation["sources_sha256"],
        "arch": arch,
        "task": task,
        "layer": layer,
        "observed_signals": sorted(observed_signals),
        "available_evidence": sorted(evidence),
        "framework_versions": dict(sorted(framework_versions.items())),
        "as_of": today.isoformat(),
        "max_review_age_days": max_review_age_days,
        "context_budget_bytes": context_budget_bytes,
        "limit": limit,
        "selected_context_bytes": consumed,
        "capabilities": selected,
        "rejected": rejected,
        "context_cost_model": "utf8_bytes",
        "execution_authority": "none",
        "promotion_authority": "local_correctness_and_measurement_only",
    }
    result["query_sha256"] = _canonical_digest(result)
    return result


def validate_query_result(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Replay a query against the current hash-bound registry."""
    if type(value) is not dict:
        raise ValueError("query result must be an object")
    recorded = value.get("query_sha256")
    if type(recorded) is not str or _HASH_RE.fullmatch(recorded) is None:
        raise ValueError("query result has an invalid query_sha256")
    unsigned = dict(value)
    unsigned.pop("query_sha256")
    if _canonical_digest(unsigned) != recorded:
        raise ValueError("query result digest changed")
    required = {
        "arch",
        "task",
        "layer",
        "observed_signals",
        "available_evidence",
        "framework_versions",
        "context_budget_bytes",
        "limit",
        "as_of",
        "max_review_age_days",
        "registry_variant",
    }
    if not required.issubset(value):
        raise ValueError("query result is missing replay inputs")
    replayed = query(
        arch=value["arch"],
        task=value["task"],
        layer=value["layer"],
        signals=value["observed_signals"],
        available_evidence=value["available_evidence"],
        framework_versions=value["framework_versions"],
        context_budget_bytes=value["context_budget_bytes"],
        limit=value["limit"],
        as_of=value["as_of"],
        max_review_age_days=value["max_review_age_days"],
        registry_variant=value["registry_variant"],
        allow_ablation=value["registry_variant"] == "shuffled",
    )
    if replayed != value:
        raise ValueError("query result does not match registry replay")
    return replayed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Query hash-bound GPU optimization playbooks")
    parser.add_argument("--arch")
    parser.add_argument("--task")
    parser.add_argument("--layer")
    parser.add_argument("--signals", help="Comma-separated normalized signals")
    parser.add_argument("--evidence", default="", help="Comma-separated evidence kinds")
    parser.add_argument("--framework-versions", help="JSON object or file")
    parser.add_argument("--context-budget-bytes", type=int, default=12000)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--registry-variant", choices=("real", "shuffled"), default="real")
    parser.add_argument("--allow-ablation", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        print(
            json.dumps(
                validate_registry(trusted_root=SKILL_ROOT),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    missing = [
        name
        for name in ("arch", "task", "layer", "signals", "framework_versions")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            "query requires: "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
        )
    raw_frameworks = args.framework_versions
    path = Path(raw_frameworks)
    if path.is_file():
        frameworks, _ = _read_json_snapshot(path, path.parent)
    else:
        frameworks = json.loads(raw_frameworks)
    result = query(
        arch=args.arch,
        task=args.task,
        layer=args.layer,
        signals=[item for item in args.signals.split(",") if item],
        available_evidence=[item for item in args.evidence.split(",") if item],
        framework_versions=frameworks,
        context_budget_bytes=args.context_budget_bytes,
        limit=args.limit,
        registry_variant=args.registry_variant,
        allow_ablation=args.allow_ablation,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
