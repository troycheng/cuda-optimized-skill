#!/usr/bin/env python3
"""Keep optimization rounds performance-first without running the target."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import artifact_store  # noqa: E402
import evidence_protocol  # noqa: E402


RECORD_SCHEMA = "cuda-optimizer/performance-iteration-v1"
REGISTRY_SCHEMA = "cuda-optimizer/measurement-path-registry-v1"
ANCHOR_SCHEMA = "cuda-optimizer/iteration-lineage-v1"
BINDING_SCHEMA = "cuda-optimizer/iteration-binding-v1"
DECISION_SCHEMA = "cuda-optimizer/iteration-decision-v1"
NON_CANDIDATE_CLASSES = {"measurement_blocked", "infrastructure_only"}
TERMINAL_STATES = {
    "valid",
    "invalid_contaminated",
    "invalid_identity",
    "partial",
    "superseded",
}
VERDICTS = {"confirmed_win", "confirmed_loss", "inconclusive", "failed", "unknown"}


def _pairs_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _json_bytes(raw: bytes, field: str) -> dict:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {field}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{field} root must be an object")
    return payload


def load_json_strict(path: Path | str) -> dict:
    return _json_bytes(artifact_store.read_regular_bytes(path), str(path))


def _closed(value: object, *, keys: set[str], field: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    missing = sorted(keys - set(value))
    unknown = sorted(set(value) - keys)
    if missing or unknown:
        raise ValueError(f"{field} has missing keys {missing} and unknown keys {unknown}")
    return dict(value)


def _string(value: object, field: str, *, min_length: int = 1) -> str:
    if not isinstance(value, str) or len(value.strip()) < min_length:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _sha256(value: object, field: str) -> str:
    text = _string(value, field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _finite(value: object, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return number


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _safe_relative(value: object, field: str) -> str:
    text = _string(value, field)
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or "\\" in text
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ValueError(f"{field} must be a safe relative path")
    return text


def _leaf(value: object, field: str) -> str:
    text = _safe_relative(value, field)
    if len(PurePosixPath(text).parts) != 1:
        raise ValueError(f"{field} must name one file in the evidence directory")
    return text


def _validate_path(value: object, field: str) -> dict:
    item = _closed(
        value,
        keys={"id", "version", "definition_sha256"},
        field=field,
    )
    return {
        "id": _string(item["id"], f"{field}.id"),
        "version": _string(item["version"], f"{field}.version"),
        "definition_sha256": _sha256(
            item["definition_sha256"], f"{field}.definition_sha256"
        ),
    }


def _validate_registry(value: object) -> list[dict]:
    registry = _closed(value, keys={"schema_version", "paths"}, field="registry")
    if registry["schema_version"] != REGISTRY_SCHEMA:
        raise ValueError(f"registry.schema_version must be {REGISTRY_SCHEMA}")
    if not isinstance(registry["paths"], list) or not registry["paths"]:
        raise ValueError("registry.paths must be a non-empty array")
    paths = []
    identities = set()
    for index, value in enumerate(registry["paths"]):
        field = f"registry.paths[{index}]"
        item = _closed(
            value,
            keys={"id", "version", "definition_sha256", "status"},
            field=field,
        )
        path = _validate_path(
            {key: item[key] for key in ("id", "version", "definition_sha256")},
            field,
        )
        if item["status"] != "validated":
            raise ValueError(f"{field}.status must be validated before lineage init")
        identity = (path["id"], path["version"])
        if identity in identities:
            raise ValueError(f"duplicate measurement path: {identity[0]}@{identity[1]}")
        identities.add(identity)
        paths.append(path)
    return paths


def freeze_lineage(
    registry_payload: Mapping,
    *,
    baseline_source_sha256: str,
    environment_sha256: str,
    initial_measurement_path: Mapping,
) -> dict:
    """Freeze baseline, environment and prevalidated paths before round one."""
    paths = _validate_registry(registry_payload)
    initial = _validate_path(initial_measurement_path, "initial_measurement_path")
    if paths.count(initial) != 1:
        raise ValueError("initial measurement path is not a validated registry entry")
    return {
        "schema_version": ANCHOR_SCHEMA,
        "baseline_source_sha256": _sha256(
            baseline_source_sha256, "baseline_source_sha256"
        ),
        "environment_sha256": _sha256(environment_sha256, "environment_sha256"),
        "measurement_paths": paths,
        "initial_measurement_path": initial,
    }


def _validate_anchor(value: object) -> tuple[dict, str]:
    anchor = _closed(
        value,
        keys={
            "schema_version",
            "baseline_source_sha256",
            "environment_sha256",
            "measurement_paths",
            "initial_measurement_path",
        },
        field="anchor",
    )
    if anchor["schema_version"] != ANCHOR_SCHEMA:
        raise ValueError(f"anchor.schema_version must be {ANCHOR_SCHEMA}")
    paths = anchor["measurement_paths"]
    if not isinstance(paths, list) or not paths:
        raise ValueError("anchor.measurement_paths must be a non-empty array")
    normalized_paths = [
        _validate_path(path, f"anchor.measurement_paths[{index}]")
        for index, path in enumerate(paths)
    ]
    if len({(path["id"], path["version"]) for path in normalized_paths}) != len(
        normalized_paths
    ):
        raise ValueError("anchor measurement path identities must be unique")
    initial = _validate_path(anchor["initial_measurement_path"], "anchor.initial_measurement_path")
    if normalized_paths.count(initial) != 1:
        raise ValueError("anchor initial measurement path is not frozen in the registry")
    normalized = {
        "schema_version": ANCHOR_SCHEMA,
        "baseline_source_sha256": _sha256(
            anchor["baseline_source_sha256"], "anchor.baseline_source_sha256"
        ),
        "environment_sha256": _sha256(
            anchor["environment_sha256"], "anchor.environment_sha256"
        ),
        "measurement_paths": normalized_paths,
        "initial_measurement_path": initial,
    }
    return normalized, _canonical_digest(normalized)


def _validate_ref(value: object, field: str) -> dict:
    ref = _closed(value, keys={"path", "sha256"}, field=field)
    return {
        "path": _leaf(ref["path"], f"{field}.path"),
        "sha256": _sha256(ref["sha256"], f"{field}.sha256"),
    }


def _literal_size(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _verify_seal_no_follow(
    evidence_root: Path,
    *,
    seal_raw: bytes,
    seal: Mapping,
    audit: Mapping,
) -> dict[str, dict]:
    sealed = _closed(
        seal,
        keys={
            "schema_version",
            "attempt_id",
            "attempt_state",
            "claim_layer",
            "attempt_manifest",
            "artifacts",
            "gate_results",
            "gate_errors",
            "evidence_integrity",
        },
        field="seal",
    )
    attempt_id = _string(sealed["attempt_id"], "seal.attempt_id")
    attempt_state = sealed["attempt_state"]
    if (
        sealed["schema_version"] != "cuda-evidence/seal-v1"
        or attempt_state not in TERMINAL_STATES
        or sealed["claim_layer"]
        not in {"isolated_operator", "matched_runtime", "serving_endpoint"}
        or sealed["evidence_integrity"] != "not_audited"
        or not isinstance(sealed["gate_results"], Mapping)
        or not isinstance(sealed["gate_errors"], Mapping)
    ):
        raise ValueError("sealed V2.5 seal contract is invalid")

    manifest_ref = _closed(
        sealed["attempt_manifest"],
        keys={"path", "sha256", "size_bytes"},
        field="seal.attempt_manifest",
    )
    manifest_path = evidence_root / _safe_relative(
        manifest_ref["path"], "seal.attempt_manifest.path"
    )
    manifest_raw = artifact_store.read_regular_bytes(manifest_path)
    manifest_sha = _sha256(manifest_ref["sha256"], "seal.attempt_manifest.sha256")
    manifest_size = _literal_size(
        manifest_ref["size_bytes"], "seal.attempt_manifest.size_bytes"
    )
    if _sha256_bytes(manifest_raw) != manifest_sha or len(manifest_raw) != manifest_size:
        raise ValueError("sealed V2.5 attempt manifest integrity failed")
    attempt = _closed(
        _json_bytes(manifest_raw, "attempt manifest"),
        keys={"schema_version", "attempt_id", "state", "claim_layer", "artifacts"},
        field="attempt manifest",
    )
    if (
        attempt["schema_version"] != "cuda-evidence/attempt-v1"
        or attempt["attempt_id"] != attempt_id
        or attempt["state"] != attempt_state
        or attempt["claim_layer"] != sealed["claim_layer"]
        or not isinstance(attempt["artifacts"], list)
    ):
        raise ValueError("sealed V2.5 attempt manifest binding failed")

    artifacts = sealed["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("sealed V2.5 artifact list is invalid")
    expected_refs = []
    actual_refs = []
    ids = set()
    kinds = set()
    paths = set()
    verified_artifacts = {}
    for index, item in enumerate(attempt["artifacts"]):
        row = _closed(
            item,
            keys={"id", "kind", "path"},
            field=f"attempt.artifacts[{index}]",
        )
        expected_refs.append(row)
    for index, item in enumerate(artifacts):
        row = _closed(
            item,
            keys={"id", "kind", "path", "sha256", "size_bytes"},
            field=f"seal.artifacts[{index}]",
        )
        artifact_id = _string(row["id"], f"seal.artifacts[{index}].id")
        kind = _string(row["kind"], f"seal.artifacts[{index}].kind")
        relative = _safe_relative(row["path"], f"seal.artifacts[{index}].path")
        if artifact_id in ids or kind in kinds or relative in paths:
            raise ValueError("sealed V2.5 artifact identities must be unique")
        ids.add(artifact_id)
        kinds.add(kind)
        paths.add(relative)
        raw = artifact_store.read_regular_bytes(evidence_root / relative)
        digest = _sha256(row["sha256"], f"seal.artifacts[{index}].sha256")
        size = _literal_size(row["size_bytes"], f"seal.artifacts[{index}].size_bytes")
        if _sha256_bytes(raw) != digest or len(raw) != size:
            raise ValueError(f"sealed V2.5 artifact integrity failed: {kind}")
        verified_artifacts[kind] = {"record": dict(row), "raw": raw}
        actual_refs.append({"id": artifact_id, "kind": kind, "path": relative})
    if expected_refs != actual_refs:
        raise ValueError("sealed V2.5 attempt and seal artifact bindings differ")

    audited = _closed(
        audit,
        keys={
            "schema_version",
            "attempt_id",
            "attempt_state",
            "seal_sha256",
            "evidence_integrity",
            "artifact_count",
            "reasons",
        },
        field="audit",
    )
    if (
        audited["schema_version"] != "cuda-evidence/audit-v1"
        or audited["attempt_id"] != attempt_id
        or audited["attempt_state"] != attempt_state
        or audited["seal_sha256"] != _sha256_bytes(seal_raw)
        or audited["evidence_integrity"] != "PASS"
        or audited["artifact_count"] != len(artifacts)
        or audited["reasons"] != []
    ):
        raise ValueError("sealed V2.5 audit does not match no-follow rehash")
    return verified_artifacts


def _recompute_v25_semantics(seal: Mapping, verified_artifacts: Mapping) -> dict:
    """Run existing V2.5 semantic gates against an isolated byte snapshot."""
    with tempfile.TemporaryDirectory(prefix="cuda-iteration-evidence-") as temporary:
        root = Path(temporary)
        by_kind = {}
        for index, (kind, item) in enumerate(sorted(verified_artifacts.items())):
            snapshot = root / f"artifact-{index:04d}"
            snapshot.write_bytes(item["raw"])
            by_kind[kind] = {
                "file_path": snapshot,
                "record": dict(item["record"]),
            }
        return evidence_protocol.recompute_sealed_semantics(seal, by_kind)


def _validate_v25_performance_verdict(value: object) -> dict:
    verdict = _closed(
        value,
        keys={
            "schema_version",
            "status",
            "promotional_eligible",
            "analysis_sha256",
            "experiment_design_sha256",
            "raw_rows_sha256",
        },
        field="performance verdict",
    )
    if verdict["schema_version"] != "cuda-evidence/performance-verdict-v1":
        raise ValueError("sealed V2.5 performance verdict schema is unsupported")
    if verdict["status"] not in {"confirmed_win", "confirmed_loss", "inconclusive", "failed"}:
        raise ValueError("sealed V2.5 performance verdict status is invalid")
    if type(verdict["promotional_eligible"]) is not bool:
        raise ValueError("sealed V2.5 promotional_eligible must be a boolean")
    for field in ("analysis_sha256", "experiment_design_sha256", "raw_rows_sha256"):
        _sha256(verdict[field], f"performance verdict.{field}")
    if verdict["promotional_eligible"] != (verdict["status"] == "confirmed_win"):
        raise ValueError("sealed V2.5 promotional eligibility is inconsistent")
    return verdict


def load_v25_closure(path: Path | str, expected_sha256: str) -> dict:
    """Rehash a V2.5 closure and return only facts needed by the thin guard."""
    manifest_path = Path(path)
    expected = _sha256(expected_sha256, "evidence_manifest_sha256")
    first_manifest = artifact_store.read_regular_bytes(manifest_path)
    if _sha256_bytes(first_manifest) != expected:
        raise ValueError("sealed V2.5 evidence manifest digest mismatch")
    manifest = _closed(
        _json_bytes(first_manifest, "evidence manifest"),
        keys={"schema_version", "attempt_id", "evidence_refs"},
        field="evidence manifest",
    )
    if manifest["schema_version"] != "cuda-evidence/manifest-v1":
        raise ValueError("sealed V2.5 evidence manifest schema is unsupported")
    attempt_id = _string(manifest["attempt_id"], "evidence manifest.attempt_id")
    refs = _closed(
        manifest["evidence_refs"],
        keys={"seal", "audit", "decision"},
        field="evidence manifest.evidence_refs",
    )
    refs = {name: _validate_ref(refs[name], f"evidence_refs.{name}") for name in refs}
    names = [manifest_path.name] + [refs[name]["path"] for name in ("seal", "audit", "decision")]
    if len(names) != len(set(names)):
        raise ValueError("sealed V2.5 closure file names must be unique")
    bundle = artifact_store.read_regular_bundle(manifest_path.parent, names)
    if bundle[manifest_path.name] != first_manifest:
        raise ValueError("sealed V2.5 evidence manifest changed during validation")
    for name, ref in refs.items():
        if _sha256_bytes(bundle[ref["path"]]) != ref["sha256"]:
            raise ValueError(f"sealed V2.5 {name} digest mismatch")

    seal = _json_bytes(bundle[refs["seal"]["path"]], "seal")
    audit = _json_bytes(bundle[refs["audit"]["path"]], "audit")
    decision = _closed(
        _json_bytes(bundle[refs["decision"]["path"]], "decision"),
        keys={
            "schema_version",
            "attempt_id",
            "attempt_state",
            "performance_verdict",
            "evidence_integrity",
            "decision",
            "reasons",
            "evidence_refs",
        },
        field="decision",
    )
    verified_artifacts = _verify_seal_no_follow(
        manifest_path.parent,
        seal_raw=bundle[refs["seal"]["path"]],
        seal=seal,
        audit=audit,
    )
    semantics = _recompute_v25_semantics(seal, verified_artifacts)
    if (
        semantics["gate_results"] != seal.get("gate_results")
        or semantics["gate_errors"] != seal.get("gate_errors")
    ):
        raise ValueError("sealed V2.5 semantic gates do not match recomputation")
    if decision["schema_version"] != "cuda-evidence/decision-v1":
        raise ValueError("sealed V2.5 decision schema is unsupported")
    if decision["decision"] not in {"promote", "retain"}:
        raise ValueError("sealed V2.5 promotion decision is invalid")
    if decision["attempt_id"] != attempt_id or seal.get("attempt_id") != attempt_id:
        raise ValueError("sealed V2.5 attempt identity mismatch")
    if decision["attempt_state"] != seal.get("attempt_state") or decision[
        "attempt_state"
    ] not in TERMINAL_STATES:
        raise ValueError("sealed V2.5 attempt state mismatch")
    if decision["evidence_integrity"] != "PASS":
        raise ValueError("sealed V2.5 decision did not pass evidence integrity")
    decision_refs = _closed(
        decision["evidence_refs"],
        keys={"seal", "audit", "performance_verdict"},
        field="decision.evidence_refs",
    )
    if _validate_ref(decision_refs["seal"], "decision.evidence_refs.seal") != refs["seal"]:
        raise ValueError("sealed V2.5 decision seal reference mismatch")
    if _validate_ref(decision_refs["audit"], "decision.evidence_refs.audit") != refs["audit"]:
        raise ValueError("sealed V2.5 decision audit reference mismatch")

    artifacts = seal.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("sealed V2.5 artifacts are invalid")
    sources = [item for item in artifacts if isinstance(item, Mapping) and item.get("kind") == "source"]
    if len(sources) != 1:
        raise ValueError("sealed V2.5 evidence must contain exactly one source artifact")
    source_sha256 = _sha256(sources[0].get("sha256"), "seal.source.sha256")
    source_path = _safe_relative(sources[0].get("path"), "seal.source.path")

    runners = [
        item
        for item in artifacts
        if isinstance(item, Mapping) and item.get("kind") == "runner"
    ]
    bindings = [
        item
        for item in artifacts
        if isinstance(item, Mapping) and item.get("kind") == "iteration_binding"
    ]
    if len(runners) != 1 or len(bindings) != 1:
        raise ValueError(
            "sealed V2.5 evidence requires one runner and one iteration binding"
        )
    runner_sha256 = _sha256(runners[0].get("sha256"), "seal.runner.sha256")
    binding_record = bindings[0]
    binding_path = manifest_path.parent / _safe_relative(
        binding_record.get("path"), "seal.iteration_binding.path"
    )
    binding_raw = artifact_store.read_regular_bytes(binding_path)
    if _sha256_bytes(binding_raw) != _sha256(
        binding_record.get("sha256"), "seal.iteration_binding.sha256"
    ):
        raise ValueError("sealed V2.5 iteration binding digest mismatch")
    binding = _closed(
        _json_bytes(binding_raw, "iteration binding"),
        keys={
            "schema_version",
            "anchor_sha256",
            "environment_sha256",
            "measurement_path",
            "hypothesis_sha256",
            "source_path",
        },
        field="iteration binding",
    )
    if binding["schema_version"] != BINDING_SCHEMA:
        raise ValueError("sealed V2.5 iteration binding schema is unsupported")
    normalized_binding = {
        "schema_version": BINDING_SCHEMA,
        "anchor_sha256": _sha256(binding["anchor_sha256"], "iteration_binding.anchor_sha256"),
        "environment_sha256": _sha256(
            binding["environment_sha256"], "iteration_binding.environment_sha256"
        ),
        "measurement_path": _validate_path(
            binding["measurement_path"], "iteration_binding.measurement_path"
        ),
        "hypothesis_sha256": _sha256(
            binding["hypothesis_sha256"], "iteration_binding.hypothesis_sha256"
        ),
        "source_path": _safe_relative(binding["source_path"], "iteration_binding.source_path"),
    }
    if normalized_binding["measurement_path"]["definition_sha256"] != runner_sha256:
        raise ValueError("iteration binding measurement path does not match sealed runner")
    if normalized_binding["source_path"] != source_path:
        raise ValueError("iteration binding source path does not match sealed source")

    performance_records = [
        item
        for item in artifacts
        if isinstance(item, Mapping) and item.get("kind") == "performance_verdict"
    ]
    performance_verdict = "unknown"
    if performance_records:
        if len(performance_records) != 1:
            raise ValueError("sealed V2.5 evidence has duplicate performance verdicts")
        performance_record = performance_records[0]
        performance_path = manifest_path.parent / _safe_relative(
            performance_record.get("path"), "seal.performance_verdict.path"
        )
        performance_raw = artifact_store.read_regular_bytes(performance_path)
        if _sha256_bytes(performance_raw) != _sha256(
            performance_record.get("sha256"), "seal.performance_verdict.sha256"
        ):
            raise ValueError("sealed V2.5 performance verdict digest mismatch")
        performance = _validate_v25_performance_verdict(
            _json_bytes(performance_raw, "performance verdict")
        )
        performance_verdict = performance["status"]
    if performance_verdict not in VERDICTS or decision["performance_verdict"] != performance_verdict:
        raise ValueError("sealed V2.5 performance verdict does not match decision")
    expected_decision = (
        "promote" if semantics["promotion_without_integrity"] else "retain"
    )
    if (
        semantics["performance_verdict"] != performance_verdict
        or decision["decision"] != expected_decision
    ):
        raise ValueError("sealed V2.5 decision does not match semantic recomputation")
    return {
        "manifest_sha256": expected,
        "attempt_id": attempt_id,
        "attempt_state": decision["attempt_state"],
        "source_sha256": source_sha256,
        "performance_verdict": performance_verdict,
        "promotion_decision": decision["decision"],
        "iteration_binding": normalized_binding,
    }


def _validate_hypothesis(value: object) -> dict:
    hypothesis = _closed(
        value,
        keys={
            "statement",
            "mechanism",
            "target_metric",
            "direction",
            "minimum_effect_pct",
            "mutation_scope",
        },
        field="record.hypothesis",
    )
    if hypothesis["direction"] not in {"lower", "higher"}:
        raise ValueError("record.hypothesis.direction must be lower or higher")
    scope = hypothesis["mutation_scope"]
    if not isinstance(scope, list) or not scope:
        raise ValueError("record.hypothesis.mutation_scope must be non-empty")
    normalized_scope = [
        _safe_relative(path, f"record.hypothesis.mutation_scope[{index}]")
        for index, path in enumerate(scope)
    ]
    if len(normalized_scope) != len(set(normalized_scope)):
        raise ValueError("record.hypothesis.mutation_scope contains duplicates")
    return {
        "statement": _string(hypothesis["statement"], "record.hypothesis.statement", min_length=12),
        "mechanism": _string(hypothesis["mechanism"], "record.hypothesis.mechanism"),
        "target_metric": _string(hypothesis["target_metric"], "record.hypothesis.target_metric"),
        "direction": hypothesis["direction"],
        "minimum_effect_pct": _finite(
            hypothesis["minimum_effect_pct"],
            "record.hypothesis.minimum_effect_pct",
            minimum=0.0000001,
        ),
        "mutation_scope": normalized_scope,
    }


def _path_is_in_scope(path: str, scope: str) -> bool:
    return path == scope or path.startswith(scope.rstrip("/") + "/")


def make_iteration_binding(
    anchor_payload: Mapping,
    record_payload: Mapping,
    *,
    source_path: str,
) -> dict:
    """Build the small context artifact that V2.5 seals with the candidate."""
    anchor, anchor_sha256 = _validate_anchor(anchor_payload)
    if not isinstance(record_payload, Mapping):
        raise ValueError("record must be an object")
    if _sha256(record_payload.get("anchor_sha256"), "record.anchor_sha256") != anchor_sha256:
        raise ValueError("record anchor does not match iteration binding anchor")
    hypothesis = _validate_hypothesis(record_payload.get("hypothesis"))
    measurement_path = _validate_path(
        record_payload.get("measurement_path"), "record.measurement_path"
    )
    if anchor["measurement_paths"].count(measurement_path) != 1:
        raise ValueError("iteration binding measurement path is not frozen")
    normalized_source = _safe_relative(source_path, "iteration binding.source_path")
    if not any(
        _path_is_in_scope(normalized_source, scope)
        for scope in hypothesis["mutation_scope"]
    ):
        raise ValueError("iteration binding source_path escapes mutation_scope")
    return {
        "schema_version": BINDING_SCHEMA,
        "anchor_sha256": anchor_sha256,
        "environment_sha256": anchor["environment_sha256"],
        "measurement_path": measurement_path,
        "hypothesis_sha256": _canonical_digest(hypothesis),
        "source_path": normalized_source,
    }


def _validate_budget(value: object) -> dict:
    budget = _closed(
        value,
        keys={"round_seconds", "infrastructure_seconds", "infrastructure_repairs"},
        field="record.budget",
    )
    return {
        "round_seconds": _finite(budget["round_seconds"], "record.budget.round_seconds", minimum=0.0000001),
        "infrastructure_seconds": _finite(
            budget["infrastructure_seconds"], "record.budget.infrastructure_seconds", minimum=0
        ),
        "infrastructure_repairs": _integer(
            budget["infrastructure_repairs"], "record.budget.infrastructure_repairs"
        ),
    }


def _validate_previous(previous: object, *, anchor_sha256: str, round_index: int, expected_sha256: str | None) -> dict | None:
    if round_index == 1:
        if previous is not None or expected_sha256 is not None:
            raise ValueError("round one must not provide a previous decision")
        return None
    if previous is None or expected_sha256 is None:
        raise ValueError("rounds after one require the bound previous decision")
    row = _closed(
        previous,
        keys={
            "schema_version",
            "round_id",
            "round_index",
            "lineage_id",
            "anchor_sha256",
            "previous_decision_sha256",
            "record_sha256",
            "work_class",
            "performance_result",
            "budget",
            "measurement_path",
            "fallback_measurement_path",
            "reasons",
            "next_action",
        },
        field="previous decision",
    )
    if _canonical_digest(row) != expected_sha256:
        raise ValueError("previous decision digest mismatch")
    if row["schema_version"] != DECISION_SCHEMA:
        raise ValueError("previous decision schema mismatch")
    if row["anchor_sha256"] != anchor_sha256 or row["lineage_id"] != anchor_sha256:
        raise ValueError("previous decision anchor mismatch")
    if row["round_index"] != round_index - 1:
        raise ValueError("previous decision round order mismatch")
    if row["work_class"] not in {
        "candidate_evaluated",
        "measurement_blocked",
        "infrastructure_only",
    }:
        raise ValueError("previous decision work class is invalid")
    if row["next_action"] not in {
        "continue_candidate_search",
        "proceed_to_existing_promotion_gate",
        "return_to_candidate",
        "switch_measurement_path",
        "stop_direction",
    }:
        raise ValueError("previous decision next action is invalid")
    return row


def classify_iteration(
    record_payload: Mapping,
    anchor_payload: Mapping,
    *,
    evidence_manifest: Path | str | None = None,
    previous: Mapping | None = None,
) -> dict:
    anchor, anchor_sha256 = _validate_anchor(anchor_payload)
    record = _closed(
        record_payload,
        keys={
            "schema_version",
            "round_id",
            "round_index",
            "anchor_sha256",
            "previous_decision_sha256",
            "hypothesis",
            "budget",
            "measurement_path",
            "candidate_declared",
            "evidence_manifest_sha256",
        },
        field="record",
    )
    if record["schema_version"] != RECORD_SCHEMA:
        raise ValueError(f"record.schema_version must be {RECORD_SCHEMA}")
    round_id = _string(record["round_id"], "record.round_id")
    round_index = _integer(record["round_index"], "record.round_index", minimum=1)
    if _sha256(record["anchor_sha256"], "record.anchor_sha256") != anchor_sha256:
        raise ValueError("record.anchor_sha256 does not match the frozen anchor")
    previous_sha = record["previous_decision_sha256"]
    if previous_sha is not None:
        previous_sha = _sha256(previous_sha, "record.previous_decision_sha256")
    previous_row = _validate_previous(
        previous,
        anchor_sha256=anchor_sha256,
        round_index=round_index,
        expected_sha256=previous_sha,
    )
    hypothesis = _validate_hypothesis(record["hypothesis"])
    budget = _validate_budget(record["budget"])
    path = _validate_path(record["measurement_path"], "record.measurement_path")
    if anchor["measurement_paths"].count(path) != 1:
        raise ValueError("record measurement path is not in the frozen anchor")
    if previous_row is not None and previous_row["next_action"] == "stop_direction":
        raise ValueError("the previous decision stopped this optimization direction")
    if previous_row is not None and previous_row["next_action"] == "switch_measurement_path":
        expected_fallback = previous_row["fallback_measurement_path"]
        if expected_fallback is None or path != expected_fallback:
            raise ValueError("the current round must use the previous frozen fallback path")
    if type(record["candidate_declared"]) is not bool:
        raise ValueError("record.candidate_declared must be a boolean")
    candidate_declared = record["candidate_declared"]
    evidence_sha = record["evidence_manifest_sha256"]
    if evidence_sha is not None:
        evidence_sha = _sha256(evidence_sha, "record.evidence_manifest_sha256")
    if not candidate_declared and evidence_sha is not None:
        raise ValueError("candidate evidence requires candidate_declared=true")
    if evidence_sha is not None and evidence_manifest is None:
        raise ValueError("candidate_evaluated requires sealed V2.5 evidence")
    if evidence_manifest is not None:
        if evidence_sha is None:
            raise ValueError("sealed V2.5 evidence path requires a manifest digest")
        facts = load_v25_closure(evidence_manifest, evidence_sha)
        if not candidate_declared or facts["manifest_sha256"] != evidence_sha:
            raise ValueError("sealed V2.5 evidence does not match the round record")
        if facts["source_sha256"] == anchor["baseline_source_sha256"]:
            raise ValueError("sealed candidate source does not differ from frozen baseline")
        expected_binding = make_iteration_binding(
            anchor,
            record,
            source_path=facts["iteration_binding"]["source_path"],
        )
        if facts["iteration_binding"] != expected_binding:
            raise ValueError("sealed V2.5 iteration binding does not match this round")
        work_class = "candidate_evaluated"
        performance_result = facts["performance_verdict"]
    elif candidate_declared:
        work_class = "measurement_blocked"
        performance_result = "not_measured"
    else:
        if budget["infrastructure_seconds"] == 0 and budget["infrastructure_repairs"] == 0:
            raise ValueError("a round without a candidate must record infrastructure work")
        work_class = "infrastructure_only"
        performance_result = "not_measured"

    cap = min(1200, math.floor(budget["round_seconds"] * 0.15))
    reasons = []
    if budget["infrastructure_seconds"] > cap:
        reasons.append("infrastructure_budget_exceeded")
    if budget["infrastructure_repairs"] > 1:
        reasons.append("infrastructure_repair_limit_exceeded")
    if (
        work_class in NON_CANDIDATE_CLASSES
        and previous_row is not None
        and previous_row["work_class"] in NON_CANDIDATE_CLASSES
    ):
        reasons.append("two_consecutive_non_candidate_rounds")

    fallback = next(
        (
            candidate
            for candidate in anchor["measurement_paths"]
            if candidate["definition_sha256"] != path["definition_sha256"]
        ),
        None,
    )
    if reasons or work_class == "measurement_blocked":
        next_action = "switch_measurement_path" if fallback else "stop_direction"
    elif work_class == "infrastructure_only":
        next_action = "return_to_candidate"
    elif performance_result == "confirmed_win" and facts["promotion_decision"] == "promote":
        next_action = "proceed_to_existing_promotion_gate"
    else:
        next_action = "continue_candidate_search"

    normalized_record = {
        "schema_version": RECORD_SCHEMA,
        "round_id": round_id,
        "round_index": round_index,
        "anchor_sha256": anchor_sha256,
        "previous_decision_sha256": previous_sha,
        "hypothesis": hypothesis,
        "budget": budget,
        "measurement_path": path,
        "candidate_declared": candidate_declared,
        "evidence_manifest_sha256": evidence_sha,
    }
    return {
        "schema_version": DECISION_SCHEMA,
        "round_id": round_id,
        "round_index": round_index,
        "lineage_id": anchor_sha256,
        "anchor_sha256": anchor_sha256,
        "previous_decision_sha256": previous_sha,
        "record_sha256": _canonical_digest(normalized_record),
        "work_class": work_class,
        "performance_result": performance_result,
        "budget": {
            "round_seconds": budget["round_seconds"],
            "infrastructure_seconds": budget["infrastructure_seconds"],
            "infrastructure_cap_seconds": cap,
            "infrastructure_repairs": budget["infrastructure_repairs"],
            "infrastructure_repair_cap": 1,
        },
        "measurement_path": path,
        "fallback_measurement_path": fallback,
        "reasons": reasons,
        "next_action": next_action,
    }


def _select_initial(registry: Mapping, selector: str) -> dict:
    if selector.count("@") != 1:
        raise ValueError("measurement path must use id@version")
    path_id, version = selector.split("@", 1)
    paths = _validate_registry(registry)
    matches = [path for path in paths if path["id"] == path_id and path["version"] == version]
    if len(matches) != 1:
        raise ValueError("measurement path selector is not a validated registry entry")
    return matches[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze and classify performance-first optimization rounds without running the target."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="freeze one optimization lineage before round one")
    init.add_argument("--registry", required=True)
    init.add_argument("--baseline-source-sha256", required=True)
    init.add_argument("--environment-sha256", required=True)
    init.add_argument("--measurement-path", required=True, help="validated path as id@version")
    init.add_argument("--out", required=True)

    binding = commands.add_parser(
        "binding", help="create the iteration context artifact before V2.5 seal"
    )
    binding.add_argument("--anchor", required=True)
    binding.add_argument("--record", required=True)
    binding.add_argument("--source-path", required=True)
    binding.add_argument("--out", required=True)

    check = commands.add_parser("check", help="validate and classify one iteration record")
    check.add_argument("--anchor", required=True)
    check.add_argument("--record", required=True)
    check.add_argument("--evidence-manifest")
    check.add_argument("--out", required=True)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            output = Path(args.out)
            if output.name != "iteration-anchor.json":
                raise ValueError("lineage anchor output must be named iteration-anchor.json")
            registry = load_json_strict(args.registry)
            anchor = freeze_lineage(
                registry,
                baseline_source_sha256=args.baseline_source_sha256,
                environment_sha256=args.environment_sha256,
                initial_measurement_path=_select_initial(registry, args.measurement_path),
            )
            artifact_store.create_regular_json(output, anchor)
            result = anchor
        elif args.command == "binding":
            output = Path(args.out)
            if output.name != "iteration-binding.json":
                raise ValueError("iteration binding output must be named iteration-binding.json")
            result = make_iteration_binding(
                load_json_strict(args.anchor),
                load_json_strict(args.record),
                source_path=args.source_path,
            )
            artifact_store.create_regular_json(output, result)
        else:
            anchor_path = Path(args.anchor)
            record = load_json_strict(args.record)
            round_index = _integer(record.get("round_index"), "record.round_index", minimum=1)
            output = Path(args.out)
            expected_name = f"round-{round_index:04d}-decision.json"
            if output.name != expected_name or os.path.abspath(output.parent) != os.path.abspath(
                anchor_path.parent
            ):
                raise ValueError(
                    f"decision output must be {expected_name} beside iteration-anchor.json"
                )
            if anchor_path.name != "iteration-anchor.json":
                raise ValueError("anchor must be named iteration-anchor.json")
            anchor = load_json_strict(anchor_path)
            evidence_sha = record.get("evidence_manifest_sha256")
            if (evidence_sha is None) != (args.evidence_manifest is None):
                raise ValueError("--evidence-manifest must match the round evidence reference")
            previous = None
            if round_index > 1:
                previous_path = anchor_path.parent / f"round-{round_index - 1:04d}-decision.json"
                previous = load_json_strict(previous_path)
            result = classify_iteration(
                record,
                anchor,
                evidence_manifest=args.evidence_manifest,
                previous=previous,
            )
            artifact_store.create_regular_json(output, result)
    except (FileExistsError, OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
