#!/usr/bin/env python3
"""Global state manager for the optimization loop (v2 — roofline-driven).

Subcommands:
  init               create run_YYYYMMDD_HHMMSS/, seed state.json
  update             after a successful iteration, merge methods into
                     selected / effective / ineffective / implementation_failed
                     lists using attribution + SASS verification data
  set-baseline-metric  called by run_iteration.py seed-baseline
  set-best-ncu-rep   helper called by profile_ncu after promoting best
  show               pretty-print current state (debug)

state.json schema (all paths stored absolute):
{
  "run_dir": str,
  "baseline_file": str,
  "ref_file": str,
  "best_file": str,
  "best_metric_ms": float | null,
  "best_ncu_rep": str | null,
  "env": {...},
  "iterations_total": int,
  "ncu_num": int,
  "branches": int,
  "budget": {...},
  "confidence": float,
  "min_effect_pct": float,
  "noise_threshold_pct": float,  # legacy init compatibility only
  "ptr_size": int,
  "dims": dict,
  "selected_methods":   [ {id, name, axis, iter} ],
  "effective_methods":  [ {id, name, axis, iter, attribution_ms} ],
  "ineffective_methods":[ {id, name, axis, iter} ],
  "implementation_failed_methods": [ {id, name, axis, iter, note} ],
  "history": [ per-iteration records ],
  "roofline_history": [ {iter, delta_c, delta_m, delta_l, bound, budget} ],
  "frontier": [ {iter, branch, kernel, ms, methods} ]
}
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path


# Keep sibling imports working both as a CLI script and via importlib file specs.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from artifact_store import (  # noqa: E402
    ArtifactStore,
    CURRENT_SCHEMA_VERSION,
    atomic_write_json,
    read_regular_bytes,
    sha256_file,
)
from budget import BudgetPolicy, resolve_budget  # noqa: E402
import decision as decision_engine  # noqa: E402
import paired_stats  # noqa: E402
import workload_evaluate  # noqa: E402


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _read(path: str) -> dict:
    try:
        return json.loads(
            read_regular_bytes(path).decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"artifact JSON is malformed: {error}") from error


def _sha256_nofollow(path: str | os.PathLike) -> str:
    return hashlib.sha256(read_regular_bytes(path)).hexdigest()


def validate_state(payload: dict) -> dict:
    """Validate a v2.2 state without mutating the caller's payload."""
    if not isinstance(payload, dict):
        raise ValueError("state payload must be a JSON object")
    if type(payload.get("schema_version")) is not int or payload.get(
        "schema_version"
    ) != CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"state schema_version must be {CURRENT_SCHEMA_VERSION}; "
            "start a new v2.2 run"
        )
    for key in ("run_dir", "input_hash", "budget", "candidates"):
        if key not in payload:
            raise ValueError(f"v2.2 state is missing required field: {key}")
    terminal = payload.get("terminal_decision")
    if terminal is not None:
        _validate_terminal_snapshot(terminal, state=payload, verify_artifacts=True)
    return payload


def _read_state(path: str) -> dict:
    return validate_state(_read(path))


def _write(path: str, payload: dict) -> None:
    atomic_write_json(path, payload)


_DECISION_STATUSES = {
    "confirmed_win",
    "confirmed_loss",
    "inconclusive",
    "no_confirmed_kernel_win",
    "workload_failed",
    "invalid",
    "kernel_only_win",
    "end_to_end_win",
    "rejected_compile",
    "rejected_correctness",
    "rejected_constraint",
    "pareto_frontier",
}
_WIN_STATUSES = {"confirmed_win", "kernel_only_win", "end_to_end_win"}
_STATISTIC_FIELDS = (
    "statistic",
    "estimate_pct",
    "ci_low_pct",
    "ci_high_pct",
    "status",
)
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


def _resolved_path(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("decision candidate_file and kernel must be non-empty paths")
    return str(Path(path).expanduser().resolve(strict=False))


def _absolute_candidate_path(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("decision candidate_file and kernel must be non-empty paths")
    return os.path.abspath(os.path.expanduser(path))


def _validate_candidate_contract(
    *, status: str, declared_candidate, candidate_file
) -> None:
    if status in _WIN_STATUSES:
        if _absolute_candidate_path(declared_candidate) != _absolute_candidate_path(
            candidate_file
        ):
            raise ValueError(
                "decision.json candidate_file does not match the update kernel"
            )
        return

    if declared_candidate is None and candidate_file is None:
        return
    if declared_candidate is None or candidate_file is None:
        raise ValueError(
            "decision.json candidate_file conflicts with the supplied kernel"
        )
    if _absolute_candidate_path(declared_candidate) != _absolute_candidate_path(
        candidate_file
    ):
        raise ValueError(
            "decision.json candidate_file does not match the update kernel"
        )


def _regular_candidate(path: str) -> tuple[Path, os.stat_result]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"decision candidate must not be a symlink: {candidate}")
    try:
        candidate_stat = candidate.lstat()
    except OSError as error:
        raise ValueError(
            f"decision candidate is missing or unreadable: {candidate}"
        ) from error
    if not stat.S_ISREG(candidate_stat.st_mode):
        raise ValueError(f"decision candidate must be a regular file: {candidate}")
    return candidate.resolve(strict=True), candidate_stat


def _validate_candidate_hash(
    decision: Mapping, *, status: str, candidate_file: str | None
) -> str | None:
    declared_hash = decision.get("candidate_sha256")
    if candidate_file is None:
        if declared_hash is not None:
            raise ValueError(
                "decision candidate_sha256 requires a candidate_file"
            )
        return None

    candidate, _ = _regular_candidate(candidate_file)
    if declared_hash is None:
        if status in _WIN_STATUSES:
            raise ValueError("winning decision requires candidate_sha256")
        return _sha256_nofollow(candidate)
    if not isinstance(declared_hash, str) or not _SHA256.fullmatch(declared_hash):
        raise ValueError("decision candidate_sha256 must be 64 hexadecimal characters")

    actual_hash = _sha256_nofollow(candidate)
    if actual_hash != declared_hash.lower():
        raise ValueError("decision candidate_sha256 does not match candidate content")
    return actual_hash


def _validate_decision_path(path: str, *, iter_dir: str) -> str:
    expected = Path(os.path.abspath(os.path.join(iter_dir, "decision.json")))
    actual = Path(os.path.abspath(os.path.expanduser(path)))
    if actual != expected:
        raise ValueError(
            "decision.json must be the decision for the current iteration"
        )
    if actual.is_symlink():
        raise ValueError("decision.json must not be a symlink")
    try:
        decision_stat = actual.lstat()
    except OSError as error:
        raise ValueError(f"decision.json missing: {actual}") from error
    if not stat.S_ISREG(decision_stat.st_mode):
        raise ValueError("decision.json must be a regular file")
    return str(actual)


def _capture_candidate_binding(
    decision: Mapping,
    *,
    status: str,
    candidate_file: str | None,
    iter_dir: str,
) -> dict | None:
    if candidate_file is None:
        return None

    candidate, candidate_stat = _regular_candidate(candidate_file)
    iteration = Path(iter_dir).expanduser().resolve()
    if candidate.parent != iteration:
        raise ValueError(
            "decision candidate must be a file in the current iteration"
        )
    actual_hash = _validate_candidate_hash(
        decision, status=status, candidate_file=str(candidate)
    )
    return {
        "path": str(candidate),
        "sha256": actual_hash,
        "device": candidate_stat.st_dev,
        "inode": candidate_stat.st_ino,
        "size": candidate_stat.st_size,
        "mtime_ns": candidate_stat.st_mtime_ns,
    }


def _verify_candidate_binding(binding: dict | None) -> None:
    """Revalidate the exact candidate immediately before state persistence."""
    if binding is None:
        return
    try:
        candidate, candidate_stat = _regular_candidate(binding["path"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("candidate changed before state write") from error
    identity = (
        candidate_stat.st_dev,
        candidate_stat.st_ino,
        candidate_stat.st_size,
        candidate_stat.st_mtime_ns,
    )
    expected_identity = (
        binding.get("device"),
        binding.get("inode"),
        binding.get("size"),
        binding.get("mtime_ns"),
    )
    if identity != expected_identity:
        raise ValueError("candidate changed before state write")
    if _sha256_nofollow(candidate) != binding.get("sha256"):
        raise ValueError("candidate sha256 changed before state write")


def _validate_decision_statistics(
    payload, *, required: bool, field_name: str = "statistics"
) -> dict | None:
    if payload is None and not required:
        return None
    required_status = "confirmed_win" if required else None
    try:
        return decision_engine.validate_paired_statistics(
            payload,
            f"decision.json {field_name}",
            required_status=required_status,
        )
    except ValueError:
        raise


_PAIRED_ARTIFACT_FIELDS = {
    "schema_version",
    "kind",
    "path",
    "sha256",
    "pairs",
    "input_hash",
    "iteration",
    "candidate_id",
    "candidate_file",
    "candidate_sha256",
    "classifier",
}


def _strict_json_copy(value, field: str):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be strict JSON") from error


def _workload_failure_snapshot(value, field: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    snapshot = {
        key: value[key]
        for key in ("status", "reason", "primary", "constraints", "failure")
        if key in value
    }
    clean = _strict_json_copy(snapshot, field)
    if clean.get("status") != "workload_failed":
        raise ValueError(f"{field}.status must be workload_failed")
    if type(clean.get("reason")) is not str or not clean["reason"].strip():
        raise ValueError(f"{field}.reason must be non-empty")
    if not isinstance(clean.get("primary"), Mapping):
        raise ValueError(f"{field}.primary must be a mapping")
    constraints = clean.get("constraints")
    if not isinstance(constraints, list) or not all(
        isinstance(item, Mapping) for item in constraints
    ):
        raise ValueError(f"{field}.constraints must be a sequence of mappings")
    return clean


def _validate_paired_artifact(
    payload,
    *,
    kind: str,
    input_hash: str,
    iteration: int,
    candidate_id: str,
    candidate_sha256: str,
    expected_pairs: int | None,
    expected_statistics: Mapping | None,
    expected_constraints,
    expected_objective,
    expected_workload_status: str | None,
    expected_workload_failure,
    expected_confidence: float,
    expected_kernel_min_effect: float,
    expected_bootstrap_samples: int,
    expected_seed: int,
    verify_file: bool,
) -> dict:
    field = f"{kind}_paired_samples"
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field} must be a mapping")
    missing = sorted(_PAIRED_ARTIFACT_FIELDS - set(payload))
    if missing:
        raise ValueError(f"{field} missing required field: {missing[0]}")
    clean = _strict_json_copy(dict(payload), field)
    if clean["schema_version"] != CURRENT_SCHEMA_VERSION:
        raise ValueError(f"{field}.schema_version is invalid")
    if clean["kind"] != kind:
        raise ValueError(f"{field}.kind does not match its evidence role")
    if clean["input_hash"] != input_hash:
        raise ValueError(f"{field}.input_hash does not match frozen input")
    if clean["iteration"] != iteration:
        raise ValueError(f"{field}.iteration does not match terminal iteration")
    if clean["candidate_sha256"] != candidate_sha256:
        raise ValueError(f"{field}.candidate_sha256 does not match decision")
    if clean["candidate_id"] != candidate_id:
        raise ValueError(f"{field}.candidate_id does not match decision")
    if (
        type(clean["candidate_id"]) is not str
        or not clean["candidate_id"].strip()
    ):
        raise ValueError(f"{field}.candidate_id must be non-empty")
    if type(clean["pairs"]) is not int or clean["pairs"] < 0:
        raise ValueError(f"{field}.pairs must be a non-negative integer")
    if expected_pairs is not None and clean["pairs"] != expected_pairs:
        raise ValueError(f"{field}.pairs does not match paired statistics")
    for hash_field in ("sha256", "candidate_sha256"):
        digest = clean[hash_field]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"{field}.{hash_field} must be a sha256 digest")
    if not isinstance(clean["path"], str) or not Path(clean["path"]).is_absolute():
        raise ValueError(f"{field}.path must be absolute")
    if (
        not isinstance(clean["candidate_file"], str)
        or not Path(clean["candidate_file"]).is_absolute()
    ):
        raise ValueError(f"{field}.candidate_file must be absolute")
    classifier = clean.get("classifier")
    if not isinstance(classifier, Mapping):
        raise ValueError(f"{field}.classifier must be a mapping")
    if not verify_file:
        return clean

    artifact = Path(clean["path"])
    artifact_bytes = read_regular_bytes(artifact)
    if hashlib.sha256(artifact_bytes).hexdigest() != clean["sha256"].lower():
        raise ValueError(f"{field}.sha256 does not match artifact content")
    sample_candidate = Path(clean["candidate_file"])
    if hashlib.sha256(read_regular_bytes(sample_candidate)).hexdigest() != candidate_sha256:
        raise ValueError(f"{field}.candidate_file does not match decision content")

    records = []
    try:
        text = artifact_bytes.decode("utf-8")
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                raise ValueError(f"{field} contains blank line {line_number}")
            records.append(
                json.loads(
                    line,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON constant: {token}")
                    ),
                )
            )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is malformed: {error}") from error
    if len(records) != clean["pairs"]:
        raise ValueError(f"{field}.pairs does not match JSONL record count")
    bindings = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "kind": kind,
        "input_hash": input_hash,
        "iteration": iteration,
        "candidate_id": candidate_id,
        "candidate_file": clean["candidate_file"],
        "candidate_sha256": candidate_sha256,
    }
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("pair_index") != index:
            raise ValueError(f"{field} record {index} has invalid pair_index")
        if any(record.get(key) != value for key, value in bindings.items()):
            raise ValueError(f"{field} record {index} has drifted bindings")
        if "pair" not in record:
            raise ValueError(f"{field} record {index} is missing pair")
        if record.get("classifier") != classifier:
            raise ValueError(f"{field} record {index} has drifted classifier")
    raw_pairs = [record["pair"] for record in records]
    if kind == "kernel":
        required_classifier = {
            "direction",
            "min_effect_pct",
            "confidence",
            "bootstrap_samples",
            "seed",
        }
        if set(classifier) != required_classifier:
            raise ValueError(f"{field}.classifier fields are invalid")
        if classifier["direction"] != "lower":
            raise ValueError(f"{field}.classifier direction does not match frozen state")
        if classifier["min_effect_pct"] != expected_kernel_min_effect:
            raise ValueError(f"{field}.classifier min_effect does not match frozen state")
        if classifier["confidence"] != expected_confidence:
            raise ValueError(f"{field}.classifier confidence does not match frozen state")
        if classifier["bootstrap_samples"] != expected_bootstrap_samples:
            raise ValueError(f"{field}.classifier bootstrap policy does not match frozen state")
        if classifier["seed"] != expected_seed:
            raise ValueError(f"{field}.classifier seed does not match frozen state")
        recomputed = paired_stats.classify_pairs(
            raw_pairs,
            direction=classifier["direction"],
            min_effect_pct=classifier["min_effect_pct"],
            confidence=classifier["confidence"],
            bootstrap_samples=classifier["bootstrap_samples"],
            seed=classifier["seed"],
        )
        if recomputed != expected_statistics:
            raise ValueError(f"{field} raw pairs do not recompute declared statistics")
    else:
        required_classifier = {
            "objective",
            "objective_sha256",
            "confidence",
            "bootstrap_samples",
            "seed",
        }
        if set(classifier) != required_classifier:
            raise ValueError(f"{field}.classifier fields are invalid")
        if classifier["confidence"] != expected_confidence:
            raise ValueError(f"{field}.classifier confidence does not match frozen state")
        if classifier["bootstrap_samples"] != expected_bootstrap_samples:
            raise ValueError(f"{field}.classifier bootstrap policy does not match frozen state")
        if classifier["seed"] != expected_seed:
            raise ValueError(f"{field}.classifier seed does not match frozen state")
        objective = classifier["objective"]
        encoded = json.dumps(
            objective,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != classifier["objective_sha256"]:
            raise ValueError(f"{field}.classifier objective hash is invalid")
        if objective != expected_objective:
            raise ValueError(f"{field}.classifier objective does not match frozen workload")
        recomputed = workload_evaluate.classify_recorded_pairs(
            objective,
            raw_pairs,
            confidence=classifier["confidence"],
            bootstrap_samples=classifier["bootstrap_samples"],
            seed=classifier["seed"],
        )
        if recomputed.get("status") != expected_workload_status:
            raise ValueError(f"{field} raw pairs do not recompute workload status")
        if expected_workload_status == "evaluated":
            if recomputed.get("primary") != expected_statistics:
                raise ValueError(f"{field} raw pairs do not recompute declared statistics")
            if recomputed.get("constraints") != expected_constraints:
                raise ValueError(f"{field} raw pairs do not recompute declared constraints")
        elif expected_workload_status == "workload_failed":
            recomputed_failure = _workload_failure_snapshot(
                recomputed, f"{field} recomputed failure"
            )
            if recomputed_failure != expected_workload_failure:
                raise ValueError(f"{field} raw pairs do not recompute workload failure")
        else:
            raise ValueError(f"{field} terminal workload status is missing")
    return clean


def _paired_count(statistics: Mapping | None) -> int | None:
    if not isinstance(statistics, Mapping):
        return None
    valid = statistics.get("valid_pairs")
    invalid = statistics.get("invalid_pairs")
    if type(valid) is int and type(invalid) is int:
        return valid + invalid
    return None


def _checkpoint_candidate_binding(checkpoint: Mapping) -> tuple[str | None, str | None]:
    candidate_id = checkpoint.get("candidate_id")
    if candidate_id is not None and (
        type(candidate_id) is not str or not candidate_id.strip()
    ):
        raise ValueError("checkpoint candidate_id must be null or non-empty")
    candidate_status = checkpoint.get("candidate_status")
    if candidate_status is not None and (
        type(candidate_status) is not str or not candidate_status.strip()
    ):
        raise ValueError("checkpoint candidate_status must be null or non-empty")
    return candidate_id, candidate_status


def _validate_terminal_snapshot(
    terminal, *, state: Mapping, verify_artifacts: bool
) -> dict:
    if not isinstance(terminal, Mapping):
        raise ValueError("state terminal_decision must be a mapping")
    clean = _strict_json_copy(dict(terminal), "terminal_decision")
    iteration = clean.get("iteration")
    if type(iteration) is not int or iteration <= 0:
        raise ValueError("terminal_decision.iteration must be positive")
    if clean.get("input_hash") != state.get("input_hash"):
        raise ValueError("terminal_decision.input_hash does not match state")
    mode = clean.get("mode")
    if mode not in {"full", "kernel-only"}:
        raise ValueError("terminal_decision.mode is invalid")
    if mode != _state_mode(dict(state)):
        raise ValueError("terminal_decision.mode does not match state")
    status = clean.get("status")
    if type(status) is not str or status not in _DECISION_STATUSES:
        raise ValueError("terminal_decision.status is invalid")
    statistics = _validate_decision_statistics(
        clean.get("statistics"), required=status in _WIN_STATUSES,
        field_name="terminal_decision.statistics",
    )
    workload_statistics = _validate_decision_statistics(
        clean.get("workload_statistics"),
        required=status == "end_to_end_win",
        field_name="terminal_decision.workload_statistics",
    )
    if status in _WIN_STATUSES and statistics["status"] != "confirmed_win":
        raise ValueError("terminal_decision win requires confirmed kernel evidence")
    if status == "end_to_end_win" and workload_statistics["status"] != "confirmed_win":
        raise ValueError("terminal_decision end_to_end_win requires workload evidence")
    workload_status = clean.get("workload_status")
    if workload_status is None and workload_statistics is not None:
        workload_status = "evaluated"
    if workload_status not in {None, "evaluated", "workload_failed"}:
        raise ValueError("terminal_decision.workload_status is invalid")
    workload_failure = clean.get("workload_failure")
    if workload_status == "evaluated":
        if workload_statistics is None:
            raise ValueError("evaluated workload requires workload_statistics")
        if workload_failure is not None:
            raise ValueError("evaluated workload must not contain workload_failure")
    elif workload_status == "workload_failed":
        if workload_statistics is not None:
            raise ValueError("workload_failed must not contain workload_statistics")
        workload_failure = _workload_failure_snapshot(
            workload_failure, "terminal_decision.workload_failure"
        )
    elif workload_failure is not None:
        raise ValueError("workload_failure requires workload_status")
    if status == "end_to_end_win" and workload_status != "evaluated":
        raise ValueError("end_to_end_win requires evaluated workload status")
    expected_confidence = state.get("confidence", 0.95)
    for field_name, stats in (
        ("statistics", statistics),
        ("workload_statistics", workload_statistics),
    ):
        if stats is not None and stats.get("confidence") != expected_confidence:
            raise ValueError(
                f"terminal_decision.{field_name}.confidence does not match frozen state"
            )
    expected_bootstrap_samples = state.get(
        "bootstrap_samples", workload_evaluate.DEFAULT_BOOTSTRAP_SAMPLES
    )
    expected_seed = state.get("seed", 0)
    expected_kernel_min_effect = state.get("min_effect_pct", 0.5)
    candidate_sha256 = clean.get("candidate_sha256")
    if candidate_sha256 is not None and (
        not isinstance(candidate_sha256, str) or not _SHA256.fullmatch(candidate_sha256)
    ):
        raise ValueError("terminal_decision.candidate_sha256 is invalid")
    candidate_file = clean.get("candidate_file")
    if candidate_file is not None:
        if type(candidate_file) is not str or not os.path.isabs(candidate_file):
            raise ValueError("terminal_decision.candidate_file must be absolute")
        if candidate_sha256 is None:
            raise ValueError("terminal_decision.candidate_file requires candidate_sha256")
        if verify_artifacts and _sha256_nofollow(candidate_file) != candidate_sha256:
            raise ValueError("terminal_decision candidate file does not match sha256")
    elif candidate_sha256 is not None:
        raise ValueError("terminal_decision.candidate_sha256 requires candidate_file")
    candidate_id = clean.get("candidate_id")
    has_candidate_evidence = any(
        value is not None
        for value in (
            clean.get("candidate_file"),
            candidate_sha256,
            statistics,
            workload_statistics,
            clean.get("kernel_paired_samples"),
            clean.get("workload_paired_samples"),
        )
    )
    if candidate_id is not None and (
        type(candidate_id) is not str or not candidate_id.strip()
    ):
        raise ValueError("terminal_decision.candidate_id must be non-empty")
    if has_candidate_evidence and candidate_id is None:
        raise ValueError("terminal_decision candidate evidence requires candidate_id")
    decision_sha256 = clean.get("decision_sha256")
    if not isinstance(decision_sha256, str) or not _SHA256.fullmatch(decision_sha256):
        raise ValueError("terminal_decision.decision_sha256 is invalid")
    decision_json = clean.get("decision_json")
    if type(decision_json) is not str or not os.path.isabs(decision_json):
        raise ValueError("terminal_decision.decision_json must be an absolute path")
    if verify_artifacts:
        if _sha256_nofollow(decision_json) != decision_sha256:
            raise ValueError("terminal_decision.decision_sha256 does not match file")
    resume = clean.get("resume", {"status": "not_recorded"})
    if not isinstance(resume, Mapping):
        raise ValueError("terminal_decision.resume must be a mapping")
    resume = _strict_json_copy(dict(resume), "terminal_decision.resume")
    resume_status = resume.get("status")
    if candidate_id is not None and resume_status == "not_recorded":
        raise ValueError(
            "terminal_decision candidate requires a recorded checkpoint resume"
        )
    if resume_status != "not_recorded":
        if type(resume_status) is not str or not resume_status.strip():
            raise ValueError("terminal_decision.resume.status must be non-empty")
        resume_stage = resume.get("stage")
        if type(resume_stage) is not str or not resume_stage.strip():
            raise ValueError("terminal_decision.resume.stage must be non-empty")
        resume_iteration = resume.get("iteration")
        if type(resume_iteration) is not int or resume_iteration < 0:
            raise ValueError("terminal_decision.resume.iteration must be non-negative")
        if resume.get("input_hash") != state.get("input_hash"):
            raise ValueError("terminal_decision.resume.input_hash does not match state")
        if not isinstance(resume.get("budget"), Mapping):
            raise ValueError("terminal_decision.resume.budget must be a mapping")
        checkpoint_path = resume.get("checkpoint")
        expected_checkpoint = os.path.abspath(
            os.path.join(os.fspath(state.get("run_dir")), "checkpoint.json")
        )
        if (
            type(checkpoint_path) is not str
            or not os.path.isabs(checkpoint_path)
            or os.path.abspath(checkpoint_path) != expected_checkpoint
        ):
            raise ValueError(
                "terminal_decision.resume.checkpoint must bind run checkpoint.json"
            )
        resume_candidate_id = resume.get("candidate_id")
        if resume_candidate_id is not None and (
            type(resume_candidate_id) is not str or not resume_candidate_id.strip()
        ):
            raise ValueError(
                "terminal_decision.resume.candidate_id must be null or non-empty"
            )
        resume_candidate_status = resume.get("candidate_status")
        if resume_candidate_status is not None and (
            type(resume_candidate_status) is not str
            or not resume_candidate_status.strip()
        ):
            raise ValueError(
                "terminal_decision.resume.candidate_status must be null or non-empty"
            )
        if candidate_id is not None:
            if resume_iteration != iteration:
                raise ValueError(
                    "terminal_decision.resume.iteration must match terminal iteration"
                )
            if resume_candidate_id is None:
                raise ValueError(
                    "terminal_decision.resume.candidate_id is required for candidate terminal"
                )
            if resume_candidate_id != candidate_id:
                raise ValueError(
                    "terminal_decision.resume.candidate_id does not match terminal"
                )
            if resume_candidate_status is None:
                raise ValueError(
                    "terminal_decision.resume.candidate_status is required for candidate terminal"
                )
            if resume_candidate_status != status:
                raise ValueError(
                    "terminal_decision.resume.candidate_status does not match terminal"
                )
        if verify_artifacts and candidate_id is not None:
            checkpoint = _read(checkpoint_path)
            if not isinstance(checkpoint, Mapping):
                raise ValueError("terminal_decision checkpoint must contain a mapping")
            expected_bindings = {
                "schema_version": CURRENT_SCHEMA_VERSION,
                "input_hash": state.get("input_hash"),
                "iteration": resume_iteration,
                "stage": resume_stage,
                "status": resume_status,
                "candidate_id": resume_candidate_id,
                "candidate_status": resume_candidate_status,
            }
            for field, expected in expected_bindings.items():
                if checkpoint.get(field) != expected:
                    raise ValueError(
                        f"terminal_decision.resume.{field} does not match checkpoint"
                    )
            if checkpoint.get("budget", {}) != resume.get("budget"):
                raise ValueError(
                    "terminal_decision.resume.budget does not match checkpoint"
                )
    clean["resume"] = resume
    for kind, stats in (("kernel", statistics), ("workload", workload_statistics)):
        field = f"{kind}_paired_samples"
        evidence = clean.get(field)
        required_evidence = (
            kind == "kernel" and status in _WIN_STATUSES
        ) or (kind == "workload" and workload_status is not None)
        if evidence is None and required_evidence:
            raise ValueError(f"terminal_decision win requires {field}")
        if evidence is not None:
            if candidate_sha256 is None or type(candidate_id) is not str:
                raise ValueError(f"{field} requires candidate binding")
            clean[field] = _validate_paired_artifact(
                evidence,
                kind=kind,
                input_hash=state["input_hash"],
                iteration=iteration,
                candidate_id=candidate_id,
                candidate_sha256=candidate_sha256,
                expected_pairs=_paired_count(stats),
                expected_statistics=stats,
                expected_constraints=clean.get("constraints", []),
                expected_objective=(
                    state.get("workload", {}).get("objective")
                    if isinstance(state.get("workload"), Mapping)
                    else None
                ),
                expected_workload_status=(
                    workload_status if kind == "workload" else None
                ),
                expected_workload_failure=(
                    workload_failure if kind == "workload" else None
                ),
                expected_confidence=expected_confidence,
                expected_kernel_min_effect=expected_kernel_min_effect,
                expected_bootstrap_samples=expected_bootstrap_samples,
                expected_seed=expected_seed,
                verify_file=verify_artifacts,
            )
    clean["statistics"] = statistics
    clean["workload_statistics"] = workload_statistics
    clean["workload_status"] = workload_status
    clean["workload_failure"] = workload_failure
    return clean


def _checkpoint_runtime_evidence(state: Mapping, *, iteration: int) -> tuple[dict, dict]:
    path = Path(state["run_dir"]) / "checkpoint.json"
    if path.is_symlink() or not path.is_file():
        return {"status": "not_recorded"}, {"status": "not_recorded"}
    payload = _read(path)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint.json must contain a mapping")
    if payload.get("input_hash") != state["input_hash"]:
        raise ValueError("checkpoint.json does not match frozen input")
    checkpoint_iteration = payload.get("iteration")
    if checkpoint_iteration not in {0, iteration}:
        raise ValueError("checkpoint.json iteration does not match state update")
    stage = payload.get("stage")
    status = payload.get("status")
    if type(stage) is not str or type(status) is not str:
        raise ValueError("checkpoint.json stage and status must be strings")
    candidate_id, candidate_status = _checkpoint_candidate_binding(payload)
    resume = {
        "status": status,
        "stage": stage,
        "iteration": checkpoint_iteration,
        "checkpoint": str(path.resolve(strict=True)),
        "input_hash": state["input_hash"],
        "budget": _strict_json_copy(payload.get("budget", {}), "checkpoint budget"),
        "candidate_id": candidate_id,
        "candidate_status": candidate_status,
    }
    stage_evidence = payload.get("stage_evidence", {})
    if not isinstance(stage_evidence, Mapping):
        raise ValueError("checkpoint.json stage_evidence must be a mapping")
    sanitizer = stage_evidence.get("candidate_sanitizer")
    if sanitizer is None:
        sanitizer_snapshot = {"status": "not_recorded"}
    elif not isinstance(sanitizer, Mapping):
        raise ValueError("checkpoint sanitizer evidence must be a mapping")
    else:
        sanitizer_snapshot = _strict_json_copy(
            dict(sanitizer), "checkpoint sanitizer evidence"
        )
    return resume, sanitizer_snapshot


def persist_checkpoint_snapshot(
    state_path: str | os.PathLike,
    checkpoint: Mapping,
    checkpoint_path: str | os.PathLike,
) -> dict:
    """Persist the checkpoint resume binding before a summary is rendered."""
    state = _read(str(state_path))
    state_without_terminal = dict(state)
    state_without_terminal.pop("terminal_decision", None)
    validate_state(state_without_terminal)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    if checkpoint.get("input_hash") != state["input_hash"]:
        raise ValueError("checkpoint does not match state frozen input")
    stage = checkpoint.get("stage")
    status = checkpoint.get("status")
    iteration = checkpoint.get("iteration")
    if type(stage) is not str or type(status) is not str:
        raise ValueError("checkpoint stage and status must be strings")
    if type(iteration) is not int or iteration < 0:
        raise ValueError("checkpoint iteration must be non-negative")
    candidate_id, candidate_status = _checkpoint_candidate_binding(checkpoint)
    path = Path(checkpoint_path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint path must be a regular non-symlink file")
    resume = {
        "status": status,
        "stage": stage,
        "iteration": iteration,
        "checkpoint": str(path.resolve(strict=True)),
        "input_hash": state["input_hash"],
        "budget": _strict_json_copy(checkpoint.get("budget", {}), "checkpoint budget"),
        "candidate_id": candidate_id,
        "candidate_status": candidate_status,
    }
    state["resume"] = resume
    terminal = state.get("terminal_decision")
    if isinstance(terminal, Mapping):
        terminal = _strict_json_copy(dict(terminal), "terminal_decision")
        terminal["resume"] = resume
        state["terminal_decision"] = _validate_terminal_snapshot(
            terminal, state=state, verify_artifacts=True
        )
    _write(str(state_path), state)
    return resume


def _load_decision(
    path: str, *, candidate_file: str | None
) -> tuple[dict, str, dict | None, dict | None]:
    if not os.path.isfile(path):
        raise ValueError(f"decision.json missing: {path}")
    try:
        decision = _read(path)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"decision.json is malformed: {error}") from error
    if not isinstance(decision, Mapping):
        raise ValueError("decision.json must contain a JSON object")
    status = decision.get("status")
    if type(status) is not str or status not in _DECISION_STATUSES:
        raise ValueError(
            "decision.json status must be a recognized terminal status"
        )
    declared_candidate = decision.get("candidate_file")
    _validate_candidate_contract(
        status=status,
        declared_candidate=declared_candidate,
        candidate_file=candidate_file,
    )
    _validate_candidate_hash(
        decision, status=status, candidate_file=candidate_file
    )
    statistics = _validate_decision_statistics(
        decision.get("statistics"), required=status in _WIN_STATUSES
    )
    workload_statistics = _validate_decision_statistics(
        decision.get("workload_statistics"),
        required=status == "end_to_end_win",
        field_name="workload_statistics",
    )
    if status in _WIN_STATUSES and statistics["status"] != "confirmed_win":
        raise ValueError(
            "decision.json statistics.status conflicts with decision status "
            f"{status}; requires confirmed_win evidence"
        )
    if (
        status == "end_to_end_win"
        and workload_statistics["status"] != "confirmed_win"
    ):
        raise ValueError(
            "decision.json workload_statistics.status conflicts with decision "
            "status end_to_end_win; requires confirmed_win evidence"
        )
    if status in {"confirmed_loss", "inconclusive", "invalid"} and (
        statistics is not None and statistics["status"] != status
    ):
        raise ValueError(
            "decision.json statistics.status conflicts with decision status"
        )
    return dict(decision), status, statistics, workload_statistics


def _state_mode(state: dict) -> str:
    raw = state.get("mode")
    if raw is None:
        raw = "full" if state.get("workload") else "kernel-only"
    if raw == "kernel_only":
        raw = "kernel-only"
    if raw not in {"full", "kernel-only"}:
        raise ValueError("state mode must be full or kernel-only")
    return raw


def _promotion_for(status: str, mode: str) -> tuple[str, bool]:
    if status == "confirmed_win":
        return "kernel_only_win", mode == "kernel-only"
    if status == "kernel_only_win":
        return status, mode == "kernel-only"
    if status == "end_to_end_win":
        if mode != "full":
            raise ValueError("end_to_end_win requires full mode")
        return status, True
    return status, False


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

def _new_run_dir(output_root: str | os.PathLike) -> Path:
    root = Path(output_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("output_root must be a directory")
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for suffix in range(1000):
        name = f"run_{stamp}" if suffix == 0 else f"run_{stamp}_{suffix}"
        candidate = root / name
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise ValueError("could not allocate a unique run directory")


def _strict_json_value(text: str, field: str):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{field} contains duplicate key: {key}")
            result[key] = value
        return result

    def nonfinite(token):
        raise ValueError(f"{field} contains non-finite JSON constant: {token}")

    try:
        return json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=nonfinite,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} must be valid strict JSON: {error}") from error


def _budget_from_json(text: str) -> dict:
    payload = _strict_json_value(text, "--budget-json")
    if not isinstance(payload, dict):
        raise ValueError("--budget-json must be a JSON object")
    fields = set(BudgetPolicy.__dataclass_fields__)
    if "soft_target_seconds" not in payload and isinstance(
        payload.get("max_seconds"), int
    ):
        payload["soft_target_seconds"] = max(1, payload["max_seconds"] // 3)
    missing = sorted(fields - set(payload))
    unknown = sorted(set(payload) - fields)
    if missing:
        raise ValueError("--budget-json missing required field: " + missing[0])
    if unknown:
        raise ValueError("--budget-json contains unknown field: " + unknown[0])
    if not isinstance(payload["name"], str) or not payload["name"].strip():
        raise ValueError("--budget-json name must be a non-empty string")
    try:
        declared = BudgetPolicy(**payload)
        validated = resolve_budget(
            "custom",
            max_seconds=declared.max_seconds,
            branches=declared.branches,
            max_rounds=declared.max_rounds,
            min_pairs=declared.min_pairs,
            max_pairs=declared.max_pairs,
            outer_candidates=declared.outer_candidates,
            max_cases=declared.max_cases,
            sanitizer_mode=declared.sanitizer_mode,
            reserve_seconds=declared.reserve_seconds,
            soft_target_seconds=declared.soft_target_seconds,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"--budget-json is invalid: {error}") from error
    # The preset name is descriptive; all executable fields were validated
    # through budget.py's public policy resolver.
    clean = asdict(validated)
    clean["name"] = declared.name
    return clean


def _frozen_input_hash(
    manifest: Mapping,
    *,
    workload,
    dims,
    backend,
    budget,
    confidence,
    min_effect_pct,
    ptr_size,
) -> str:
    frozen = {
        "inputs": {
            key: value["sha256"]
            for key, value in sorted(manifest["inputs"].items())
        },
        "workload": workload,
        "dims": dims,
        "backend": backend,
        "budget": budget,
        "confidence": confidence,
        "min_effect_pct": min_effect_pct,
        "ptr_size": ptr_size,
    }
    encoded = json.dumps(
        frozen,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def initialize_state(
    *,
    run_dir: str | os.PathLike,
    baseline: str | os.PathLike,
    ref: str | os.PathLike,
    manifest: Mapping,
    budget: Mapping,
    environment: Mapping | None,
    env_path: str | os.PathLike | None,
    dims: Mapping,
    ptr_size: int,
    ncu_num: int,
    mode: str,
    workload,
    started_at: float,
    backend: str = "auto",
    confidence: float = 0.95,
    min_effect_pct: float = 0.5,
    noise_threshold_pct: float | None = None,
) -> tuple[dict, Path]:
    """Create state.json from an already durable frozen manifest."""
    root = Path(run_dir).expanduser().resolve(strict=True)
    baseline_path = Path(baseline).expanduser().resolve(strict=True)
    ref_path = Path(ref).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("run_dir must be a directory")
    if not baseline_path.is_file():
        raise ValueError(f"baseline not found: {baseline_path}")
    if not ref_path.is_file():
        raise ValueError(f"ref not found: {ref_path}")
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")
    if manifest.get("schema_version") != CURRENT_SCHEMA_VERSION:
        raise ValueError("manifest schema_version is invalid")
    input_hash = manifest.get("input_hash")
    if not isinstance(input_hash, str) or not input_hash:
        raise ValueError("manifest input_hash must be non-empty")
    if mode not in {"full", "kernel-only"}:
        raise ValueError("mode must be full or kernel-only")

    clean_budget = json.loads(json.dumps(budget, allow_nan=False))
    iterations_total = clean_budget.get(
        "max_rounds", clean_budget.get("iterations_total")
    )
    branches = clean_budget.get("branches")
    if (
        isinstance(iterations_total, bool)
        or not isinstance(iterations_total, int)
        or iterations_total <= 0
    ):
        raise ValueError("budget max_rounds must be a positive integer")
    if isinstance(branches, bool) or not isinstance(branches, int) or branches <= 0:
        raise ValueError("budget branches must be a positive integer")

    baseline_copy_dir = root / "baseline"
    baseline_copy_dir.mkdir(parents=True, exist_ok=True)
    baseline_copy = baseline_copy_dir / baseline_path.name
    shutil.copy2(baseline_path, baseline_copy)

    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "run_dir": str(root),
        "input_hash": input_hash,
        "budget": clean_budget,
        "mode": mode,
        "workload": json.loads(json.dumps(workload, allow_nan=False)),
        "started_at": float(started_at),
        "confidence": float(confidence),
        "min_effect_pct": float(min_effect_pct),
        "bootstrap_samples": workload_evaluate.DEFAULT_BOOTSTRAP_SAMPLES,
        "seed": 0,
        "backend": backend,
        "baseline_file": str(baseline_copy),
        "baseline_file_original": str(baseline_path),
        "ref_file": str(ref_path),
        "best_file": str(baseline_copy),
        "best_kernel_statistics": None,
        "best_workload_statistics": None,
        "best_metric_ms": None,
        "best_ncu_rep": None,
        "env": json.loads(json.dumps(environment or {}, allow_nan=False)),
        "env_path": str(Path(env_path).expanduser().resolve()) if env_path else None,
        "iterations_total": iterations_total,
        "ncu_num": int(ncu_num),
        "branches": branches,
        "ptr_size": int(ptr_size),
        "dims": json.loads(json.dumps(dims, allow_nan=False)),
        "selected_methods": [],
        "effective_methods": [],
        "ineffective_methods": [],
        "implementation_failed_methods": [],
        "candidates": {},
        "history": [],
        "roofline_history": [],
        "frontier": [],
        "created_at": _dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    if noise_threshold_pct is not None:
        state["noise_threshold_pct"] = float(noise_threshold_pct)
    state_path = root / "state.json"
    _write(str(state_path), state)
    for iteration in range(1, iterations_total + 1):
        (root / f"iterv{iteration}").mkdir(exist_ok=True)
    return state, state_path

def cmd_init(args: argparse.Namespace) -> None:
    baseline = os.path.abspath(args.baseline)
    ref = os.path.abspath(args.ref)
    if not os.path.isfile(baseline):
        sys.exit(f"baseline not found: {baseline}")
    if not os.path.isfile(ref):
        sys.exit(f"ref not found: {ref}")

    env = {}
    if args.env and os.path.isfile(args.env):
        env = _read(args.env)

    dims = _strict_json_value(args.dims, "--dims") if args.dims else {}
    if not isinstance(dims, dict):
        raise ValueError("--dims must be a JSON object")

    supplied_budget = getattr(args, "budget", None)
    budget_json = getattr(args, "budget_json", None)
    if supplied_budget is not None:
        budget = dict(supplied_budget)
    elif budget_json is not None:
        budget = _budget_from_json(budget_json)
    else:
        iterations = int(getattr(args, "iterations", 3))
        budget = {
            "iterations_total": iterations,
            "max_rounds": iterations,
            "ncu_num": int(args.ncu_num),
            "branches": int(getattr(args, "branches", 4)),
        }
    workload = getattr(args, "workload", None)
    mode = getattr(args, "mode", "kernel-only")
    started_at = getattr(args, "started_at", None) or time.time()
    backend = getattr(args, "backend", "auto")
    confidence = getattr(args, "confidence", 0.95)
    min_effect_pct = getattr(args, "min_effect_pct", 0.5)
    output_root = getattr(args, "output_root", None) or os.path.dirname(baseline)
    supplied_run_dir = getattr(args, "run_dir", None)
    if supplied_run_dir is None:
        run_dir = _new_run_dir(output_root)
    else:
        run_arg = Path(supplied_run_dir).expanduser()
        if run_arg.is_symlink():
            raise ValueError("run_dir must not be a symlink")
        run_dir = run_arg.resolve(strict=True)
        root = Path(output_root).expanduser().resolve(strict=True)
        if not run_dir.is_dir() or run_dir.parent != root:
            raise ValueError("run_dir must be a direct child of output_root")

    manifest_arg = getattr(args, "manifest", None)
    if manifest_arg is None:
        store = ArtifactStore(run_dir)
        manifest = store.initialize(
            inputs={"baseline": baseline, "ref": ref},
            budget=budget,
            environment=env,
        )
        manifest.update(
            {
                "mode": mode,
                "workload": workload,
                "confidence": confidence,
                "min_effect_pct": min_effect_pct,
                "started_at": started_at,
                "dims": dims,
                "backend": backend,
                "ptr_size": args.ptr_size,
            }
        )
        manifest["input_hash"] = _frozen_input_hash(
            manifest,
            workload=workload,
            dims=dims,
            backend=backend,
            budget=budget,
            confidence=confidence,
            min_effect_pct=min_effect_pct,
            ptr_size=args.ptr_size,
        )
        atomic_write_json(run_dir / "manifest.json", manifest)
    else:
        manifest_path = Path(manifest_arg).expanduser()
        if manifest_path.is_symlink():
            raise ValueError("manifest must not be a symlink")
        manifest_path = manifest_path.resolve(strict=True)
        if manifest_path != run_dir / "manifest.json" or not manifest_path.is_file():
            raise ValueError("manifest must be run_dir/manifest.json")
        manifest = _read(str(manifest_path))

    workload_file_arg = getattr(args, "workload_file", None)
    if workload_file_arg is not None:
        workload_path = Path(workload_file_arg).expanduser()
        if workload_path.is_symlink():
            raise ValueError("workload file must not be a symlink")
        workload_path = workload_path.resolve(strict=True)
        expected_workload_path = run_dir / "workload" / "spec.json"
        if workload_path != expected_workload_path or not workload_path.is_file():
            raise ValueError("workload file must be run_dir/workload/spec.json")
        workload = _strict_json_value(
            workload_path.read_text("utf-8"), "--workload-file"
        )

    state, state_path = initialize_state(
        run_dir=run_dir,
        baseline=baseline,
        ref=ref,
        manifest=manifest,
        budget=budget,
        environment=env,
        env_path=args.env if args.env and os.path.isfile(args.env) else None,
        dims=dims,
        ptr_size=args.ptr_size,
        ncu_num=args.ncu_num,
        mode=mode,
        workload=workload,
        started_at=started_at,
        backend=backend,
        confidence=confidence,
        min_effect_pct=min_effect_pct,
        noise_threshold_pct=getattr(args, "noise_threshold_pct", None),
    )
    print(json.dumps({"run_dir": str(run_dir), "state": str(state_path)}, indent=2))


# ---------------------------------------------------------------------------
# update  (v2: uses attribution + sass_check)
# ---------------------------------------------------------------------------

def _method_key(m: dict) -> str:
    if "id" in m and m["id"]:
        return str(m["id"]).strip().lower()
    return f"{str(m.get('name','')).strip().lower()}::{str(m.get('axis','')).strip().lower()}"


def _merge_unique(bag: list[dict], new_items: list[dict]) -> list[dict]:
    seen = {_method_key(m) for m in bag}
    for m in new_items:
        k = _method_key(m)
        if k not in seen:
            bag.append(m)
            seen.add(k)
    return bag


def cmd_update(args: argparse.Namespace) -> None:
    state = _read_state(args.state)
    bench = _read(args.bench)
    methods = _read(args.methods_json)

    iter_dir = os.path.join(state["run_dir"], f"iterv{args.iter}")
    decision_path = getattr(args, "decision", None) or os.path.join(
        iter_dir, "decision.json"
    )
    decision_path = _validate_decision_path(decision_path, iter_dir=iter_dir)
    (
        decision,
        decision_status,
        decision_statistics,
        workload_statistics,
    ) = _load_decision(
        decision_path, candidate_file=args.kernel
    )
    candidate_binding = _capture_candidate_binding(
        decision,
        status=decision_status,
        candidate_file=args.kernel,
        iter_dir=iter_dir,
    )
    mode = _state_mode(state)
    terminal_status, promote_best = _promotion_for(decision_status, mode)

    if not isinstance(methods, dict) or "methods" not in methods:
        sys.exit("methods-json must contain a top-level 'methods' list")
    methods_list = methods["methods"]

    # --- Priority-compliance validation ---
    if not args.skip_validation:
        validator = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "validate_methods.py")
        cmd = [
            sys.executable, validator,
            "--methods", args.methods_json,
            "--state", args.state,
        ]
        if args.allow_ineffective:
            cmd.append("--allow-ineffective")
        rv = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="ignore")
        if rv.returncode != 0:
            sys.stderr.write(
                "\n[state update] methods.json failed validation:\n"
            )
            sys.stderr.write(rv.stdout or "")
            sys.stderr.write(rv.stderr or "")
            sys.exit(1)

    validation_passed = bool(bench.get("correctness", {}).get("passed", True))
    new_ms = None
    ref_ms = None
    if bench.get("kernel"):
        new_ms = bench["kernel"].get("average_ms")
    if bench.get("reference"):
        ref_ms = bench["reference"].get("average_ms")

    # Load attribution and sass_check if provided
    attribution_data = {}
    if args.attribution and os.path.isfile(args.attribution):
        attr = _read(args.attribution)
        for a in attr.get("attributions", []):
            attribution_data[a["method_id"]] = a

    sass_data = {}
    sass_document = {}
    if args.sass_check and os.path.isfile(args.sass_check):
        sass_document = _read(args.sass_check)
        if not isinstance(sass_document, Mapping):
            raise ValueError("sass_check.json must contain a mapping")
        for c in sass_document.get("checks", []):
            sass_data[c["method_id"]] = c

    # A benchmark average is diagnostic only. Promotion comes exclusively from
    # the terminal decision and its unified paired statistic.
    if promote_best and not validation_passed:
        raise ValueError(
            "decision.json declares a win for a correctness-failed candidate"
        )
    improved = bool(promote_best and validation_passed)
    kernel_evidence_win = bool(
        validation_passed and terminal_status in {"kernel_only_win", "end_to_end_win"}
    )
    speedup_vs_best_before = None

    # Annotate each method
    for m in methods_list:
        m.setdefault("id", _method_key(m))
        m["iter"] = int(args.iter)

    # Always: add to selected
    _merge_unique(state["selected_methods"], methods_list)

    # Classify each method based on attribution + SASS
    for m in methods_list:
        mid = m["id"]
        attr_info = attribution_data.get(mid, {})
        sass_info = sass_data.get(mid, {})

        sass_status = sass_info.get("status", "unavailable")
        contributed = attr_info.get("contributed", None)
        attr_ms = attr_info.get("attribution_ms", None)

        m_entry = dict(m)

        if sass_status == "failed":
            # SASS signature missing — implementation failed
            m_entry["note"] = f"SASS patterns not found: {sass_info.get('patterns_missing', [])}"
            state["implementation_failed_methods"].append(m_entry)
        elif contributed is True or contributed is None:
            # Contributed (or no ablation data — assume effective if the
            # unified kernel evidence is a confirmed win).
            if kernel_evidence_win:
                if attr_ms is not None:
                    m_entry["attribution_ms"] = attr_ms
                if speedup_vs_best_before is not None:
                    m_entry["speedup_vs_best_before"] = speedup_vs_best_before
                state["effective_methods"].append(m_entry)
            elif validation_passed:
                state["ineffective_methods"].append(m_entry)
        elif contributed is False:
            # Attribution says it didn't help
            m_entry["note"] = f"attribution_ms={attr_ms}"
            state["ineffective_methods"].append(m_entry)

    # Update best
    if improved:
        state["best_file"] = _resolved_path(args.kernel)
        if new_ms is not None:
            state["best_metric_ms"] = new_ms
        state["best_kernel_statistics"] = decision_statistics
        if terminal_status == "end_to_end_win":
            state["best_workload_statistics"] = workload_statistics

    # Load roofline data if available
    roofline_path = os.path.join(iter_dir, "roofline.json")
    if os.path.isfile(roofline_path):
        roofline = _read(roofline_path)
        state["roofline_history"].append({
            "iter": int(args.iter),
            "delta_compute": roofline.get("delta_compute"),
            "delta_memory": roofline.get("delta_memory"),
            "delta_latency": roofline.get("delta_latency"),
            "analysis_model": roofline.get("analysis_model"),
            "analysis_quality": roofline.get("analysis_quality"),
            "bound": roofline.get("bound"),
            "axis_budget": roofline.get("axis_budget"),
        })

    # Load frontier from branch_results if available
    branch_results_path = os.path.join(iter_dir, "branch_results.json")
    if os.path.isfile(branch_results_path):
        br = _read(branch_results_path)
        for fe in br.get("frontier", []):
            fe["methods"] = [m["id"] for m in methods_list]
            state["frontier"].append(fe)

    state["history"].append({
        "iter": int(args.iter),
        "kernel_file": os.path.abspath(args.kernel),
        "methods": [m["id"] for m in methods_list],
        "method_names": [m.get("name") for m in methods_list],
        "ms": new_ms,
        "ref_ms": ref_ms,
        "speedup_vs_ref": (ref_ms / new_ms) if (ref_ms and new_ms and new_ms > 0) else None,
        "speedup_vs_best_before": speedup_vs_best_before,
        "validation_passed": validation_passed,
        "status": terminal_status,
        "decision_status": decision_status,
        "statistics": decision_statistics,
        "decision_json": os.path.abspath(decision_path),
        "retries": int(args.retries),
    })

    _verify_candidate_binding(candidate_binding)
    if candidate_binding is not None:
        state["candidates"][f"iter-{int(args.iter)}"] = {
            "candidate_file": candidate_binding["path"],
            "candidate_sha256": candidate_binding["sha256"],
            "status": terminal_status,
        }
        if mode == "full" and terminal_status == "kernel_only_win":
            frontier_entry = {
                "iter": int(args.iter),
                "candidate_file": candidate_binding["path"],
                "path": candidate_binding["path"],
                "kernel": candidate_binding["path"],
                "candidate_sha256": candidate_binding["sha256"],
                "sha256": candidate_binding["sha256"],
                "statistics": decision_statistics,
                "kernel_statistics": decision_statistics,
                "status": "kernel_only_win",
                "mode": "full",
            }
            identity = (
                frontier_entry["iter"],
                frontier_entry["candidate_file"],
                frontier_entry["candidate_sha256"],
                frontier_entry["status"],
            )
            if not any(
                isinstance(item, Mapping)
                and (
                    item.get("iter"),
                    item.get("candidate_file") or item.get("kernel"),
                    item.get("candidate_sha256"),
                    item.get("status"),
                ) == identity
                for item in state["frontier"]
            ):
                state["frontier"].append(frontier_entry)
    resume_evidence, sanitizer_evidence = _checkpoint_runtime_evidence(
        state, iteration=int(args.iter)
    )
    compiler_evidence = bench.get("compiler_evidence", {})
    if not isinstance(compiler_evidence, Mapping):
        raise ValueError("bench compiler_evidence must be a mapping")
    constraints = decision.get("constraints", [])
    if (
        isinstance(constraints, (str, bytes, bytearray, Mapping))
        or not isinstance(constraints, list)
        or not all(isinstance(item, Mapping) for item in constraints)
    ):
        raise ValueError("decision constraints must be a sequence of mappings")
    terminal_decision = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "iteration": int(args.iter),
        "input_hash": state["input_hash"],
        "mode": mode,
        "status": terminal_status,
        "decision_status": decision_status,
        "candidate_file": (
            candidate_binding["path"] if candidate_binding is not None else None
        ),
        "candidate_sha256": (
            candidate_binding["sha256"] if candidate_binding is not None else None
        ),
        "candidate_id": decision.get("candidate_id"),
        "decision_json": os.path.abspath(decision_path),
        "decision_sha256": _sha256_nofollow(decision_path),
        "statistics": decision_statistics,
        "workload_statistics": workload_statistics,
        "workload_status": decision.get("workload_status"),
        "workload_failure": decision.get("workload_failure"),
        "constraints": _strict_json_copy(constraints, "decision constraints"),
        "correctness": {
            "status": "passed" if validation_passed else "failed",
            "passed": validation_passed,
        },
        "compiler_evidence": _strict_json_copy(
            dict(compiler_evidence), "compiler evidence"
        ),
        "sass": _strict_json_copy(dict(sass_document), "SASS evidence"),
        "sanitizer": sanitizer_evidence,
        "resume": resume_evidence,
    }
    for field in ("kernel_paired_samples", "workload_paired_samples"):
        if field in decision:
            terminal_decision[field] = _strict_json_copy(
                decision[field], field
            )
    terminal_decision = _validate_terminal_snapshot(
        terminal_decision, state=state, verify_artifacts=True
    )
    state["terminal_decision"] = terminal_decision
    state["compiler_evidence"] = terminal_decision["compiler_evidence"]
    state["sass_verification"] = terminal_decision["sass"]
    state["sanitizer_coverage"] = terminal_decision["sanitizer"].get("coverage")
    state["resume"] = terminal_decision["resume"]
    _write(args.state, state)
    print(json.dumps({
        "iter": args.iter,
        "status": terminal_status,
        "new_ms": new_ms,
        "best_ms": state["best_metric_ms"],
        "improved": improved,
        "speedup_vs_best_before": speedup_vs_best_before,
        "decision": (
            {field: decision_statistics[field] for field in _STATISTIC_FIELDS}
            if decision_statistics is not None
            else None
        ),
    }, indent=2))


# ---------------------------------------------------------------------------
# set-best-ncu-rep
# ---------------------------------------------------------------------------

def cmd_set_best_ncu(args: argparse.Namespace) -> None:
    state = _read_state(args.state)
    state["best_ncu_rep"] = os.path.abspath(args.ncu_rep)
    _write(args.state, state)
    print(json.dumps({"best_ncu_rep": state["best_ncu_rep"]}, indent=2))


# ---------------------------------------------------------------------------
# seed baseline metric
# ---------------------------------------------------------------------------

def cmd_set_baseline_metric(args: argparse.Namespace) -> None:
    state = _read_state(args.state)
    bench = _read(args.bench)
    if not bench.get("correctness", {}).get("passed", True):
        sys.exit("Baseline failed correctness validation — cannot proceed.")
    ms = bench.get("kernel", {}).get("average_ms")
    if ms is None:
        sys.exit("Baseline bench has no kernel timing.")
    state["best_metric_ms"] = ms
    _write(args.state, state)
    print(json.dumps({"baseline_ms": ms}, indent=2))


def cmd_show(args: argparse.Namespace) -> None:
    state = _read_state(args.state)
    print(json.dumps(state, indent=2))


def cmd_record_decision(args: argparse.Namespace) -> None:
    state = _read_state(args.state)
    iteration = int(args.iter)
    if iteration <= 0:
        raise ValueError("iter must be a positive integer")
    iter_dir = os.path.join(state["run_dir"], f"iterv{iteration}")
    decision_path = _validate_decision_path(args.decision, iter_dir=iter_dir)
    decision = _read(decision_path)
    if not isinstance(decision, Mapping):
        raise ValueError("decision.json must contain a JSON object")
    status = decision.get("status")
    if type(status) is not str or status not in _DECISION_STATUSES:
        raise ValueError("decision.json status must be recognized")
    if status in _WIN_STATUSES:
        raise ValueError("winning decisions must be recorded through state update")
    decision, status, statistics, workload_statistics = _load_decision(
        decision_path,
        candidate_file=decision.get("candidate_file"),
    )
    candidate_binding = _capture_candidate_binding(
        decision,
        status=status,
        candidate_file=decision.get("candidate_file"),
        iter_dir=iter_dir,
    )
    constraints = decision.get("constraints", [])
    if (
        isinstance(constraints, (str, bytes, bytearray, Mapping))
        or not isinstance(constraints, list)
        or not all(isinstance(item, Mapping) for item in constraints)
    ):
        raise ValueError("decision constraints must be a sequence of mappings")
    resume_evidence, sanitizer_evidence = _checkpoint_runtime_evidence(
        state, iteration=iteration
    )
    decision_digest = _sha256_nofollow(decision_path)
    record = {
        "event": "decision_record",
        "iter": iteration,
        "status": status,
        "decision_json": decision_path,
        "decision_sha256": decision_digest,
        "statistics": statistics,
    }
    history = state.setdefault("history", [])
    if not any(
        isinstance(item, Mapping)
        and item.get("event") == "decision_record"
        and item.get("iter") == iteration
        and item.get("decision_sha256") == record["decision_sha256"]
        for item in history
    ):
        history.append(record)
    terminal = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "iteration": iteration,
        "input_hash": state["input_hash"],
        "mode": _state_mode(state),
        "status": status,
        "decision_status": status,
        "candidate_file": (
            candidate_binding["path"] if candidate_binding is not None else None
        ),
        "candidate_sha256": (
            candidate_binding["sha256"] if candidate_binding is not None else None
        ),
        "candidate_id": decision.get("candidate_id"),
        "decision_json": decision_path,
        "decision_sha256": decision_digest,
        "statistics": statistics,
        "workload_statistics": workload_statistics,
        "workload_status": decision.get("workload_status"),
        "workload_failure": decision.get("workload_failure"),
        "constraints": _strict_json_copy(constraints, "decision constraints"),
        "correctness": {"status": "not_recorded"},
        "compiler_evidence": {"status": "not_recorded"},
        "sass": {"status": "not_recorded"},
        "sanitizer": sanitizer_evidence,
        "resume": resume_evidence,
    }
    for field in ("kernel_paired_samples", "workload_paired_samples"):
        if field in decision:
            terminal[field] = _strict_json_copy(decision[field], field)
    state["terminal_decision"] = _validate_terminal_snapshot(
        terminal, state=state, verify_artifacts=True
    )
    state["resume"] = terminal["resume"]
    _write(args.state, state)
    print(json.dumps(record, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init")
    pi.add_argument("--baseline", required=True)
    pi.add_argument("--ref", required=True)
    pi.add_argument("--iterations", type=int, default=3)
    pi.add_argument("--ncu-num", type=int, default=5)
    pi.add_argument("--branches", type=int, default=4)
    pi.add_argument("--dims", type=str, default="{}", help="JSON dict of dim name -> int")
    pi.add_argument("--env", type=str, default="")
    pi.add_argument("--ptr-size", type=int, default=0)
    pi.add_argument("--output-root", default=None)
    pi.add_argument(
        "--budget-json",
        default=None,
        help="strict JSON serialization of a complete BudgetPolicy",
    )
    # The orchestrator has already allocated and frozen these artifacts.  Keep
    # the plumbing private while still using this public init command.
    pi.add_argument("--run-dir", default=None, help=argparse.SUPPRESS)
    pi.add_argument("--manifest", default=None, help=argparse.SUPPRESS)
    pi.add_argument("--mode", choices=("full", "kernel-only"), default="kernel-only", help=argparse.SUPPRESS)
    pi.add_argument("--workload-file", default=None, help=argparse.SUPPRESS)
    pi.add_argument("--started-at", type=float, default=None, help=argparse.SUPPRESS)
    pi.add_argument("--backend", default="auto", help=argparse.SUPPRESS)
    pi.add_argument("--confidence", type=float, default=0.95, help=argparse.SUPPRESS)
    pi.add_argument("--min-effect-pct", type=float, default=0.5, help=argparse.SUPPRESS)
    pi.set_defaults(func=cmd_init)

    pu = sub.add_parser("update")
    pu.add_argument("--state", required=True)
    pu.add_argument("--iter", required=True, type=int)
    pu.add_argument("--kernel", required=True)
    pu.add_argument("--bench", required=True)
    pu.add_argument("--methods-json", required=True)
    pu.add_argument("--attribution", type=str, default=None,
                    help="Path to attribution.json from ablation step")
    pu.add_argument("--sass-check", type=str, default=None,
                    help="Path to sass_check.json from SASS verification step")
    pu.add_argument("--retries", type=int, default=0)
    pu.add_argument("--skip-validation", action="store_true")
    pu.add_argument("--allow-ineffective", action="store_true")
    pu.add_argument(
        "--decision",
        default=None,
        help="Path to decision.json (default: itervN/decision.json)",
    )
    pu.set_defaults(func=cmd_update)

    pb = sub.add_parser("set-baseline-metric")
    pb.add_argument("--state", required=True)
    pb.add_argument("--bench", required=True)
    pb.set_defaults(func=cmd_set_baseline_metric)

    pbn = sub.add_parser("set-best-ncu-rep")
    pbn.add_argument("--state", required=True)
    pbn.add_argument("--ncu-rep", required=True)
    pbn.set_defaults(func=cmd_set_best_ncu)

    ps = sub.add_parser("show")
    ps.add_argument("--state", required=True)
    ps.set_defaults(func=cmd_show)

    pr = sub.add_parser("record-decision")
    pr.add_argument("--state", required=True)
    pr.add_argument("--iter", required=True, type=int)
    pr.add_argument("--decision", required=True)
    pr.set_defaults(func=cmd_record_decision)

    return p


def main() -> None:
    p = build_parser()

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
