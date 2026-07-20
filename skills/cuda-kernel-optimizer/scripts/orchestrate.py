#!/usr/bin/env python3
"""End-to-end orchestrator (v2 — roofline-driven, branch-and-select).

Subcommands:
  setup       Steps 0-2: env check, preflight, init, seed baseline, profile+roofline for iter 1
  open-iter   Prepare an iteration: profile best → ncu_top → roofline → axis budgets
              (the agent then writes K branch kernels + methods.json + analysis.md)
  close-iter  Steps 3e-3j: branch explore → champion → ncu champion → ablate → sass → update
  finalize    Step 4: emit summary.md
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from numbers import Real
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from artifact_store import (  # noqa: E402
    ArtifactStore,
    CURRENT_SCHEMA_VERSION,
    atomic_write_json,
    publish_regular_bundle,
    read_regular_bytes,
    read_regular_with_optional_sibling,
    sha256_file,
    write_paired_samples,
)
from budget import (  # noqa: E402
    BudgetClock,
    BudgetPolicy,
    resolve_budget,
    run_budgeted_command,
)
import decision as decision_engine  # noqa: E402
import preflight  # noqa: E402
import sanitize as sanitizer_engine  # noqa: E402
import state as state_manager  # noqa: E402
from workload_adapter import (  # noqa: E402
    WorkloadSpec,
    normalize_workload,
    run_spec_once,
    verify_frozen_spec,
)
import workload_evaluate  # noqa: E402

_BRANCH_RESULT_STATUSES = {"shortlist_ready", "no_confirmed_kernel_win"}

STAGES = (
    "baseline",
    "candidate_correctness",
    "candidate_paired",
    "candidate_profile",
    "candidate_sanitizer",
    "workload_paired",
    "decision",
    "complete",
)
_CHECKPOINT_STATUSES = {
    "ready",
    "in_progress",
    "stage_complete",
    "budget_exhausted",
    "interrupted",
    "failed",
    "complete",
}
_UNSET = object()


def _finite_real(value, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be at least {minimum:g}")
    return number


def _positive_int(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _strict_json_copy(value, field: str = "value", active: set[int] | None = None):
    if active is None:
        active = set()
    if value is None or type(value) in {bool, str, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field} numbers must be finite")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{field} must not contain a cycle")
        active.add(identity)
        result = {}
        try:
            for key, item in value.items():
                if type(key) is not str:
                    raise ValueError(f"{field} mappings must use string keys")
                result[key] = _strict_json_copy(
                    item, f"{field}.{key}", active
                )
            return result
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{field} must not contain a cycle")
        active.add(identity)
        try:
            return [
                _strict_json_copy(item, f"{field}[{index}]", active)
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(identity)
    raise ValueError(f"{field} must contain only strict JSON values")


def _workload_snapshot(spec: WorkloadSpec | None) -> dict | None:
    if spec is None:
        return None
    return _strict_json_copy(
        {
            "kind": spec.kind,
            "source": spec.source,
            "objective": spec.objective,
            "cases": list(spec.cases),
            "source_hash": spec.source_hash,
        },
        "workload",
    )


def _workload_from_snapshot(snapshot) -> WorkloadSpec | None:
    if snapshot is None:
        return None
    if not isinstance(snapshot, Mapping):
        raise ValueError("state workload must be a mapping")
    required = {"kind", "source", "objective", "cases", "source_hash"}
    if set(snapshot) != required:
        raise ValueError("state workload snapshot is incomplete")
    cases = snapshot["cases"]
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes, bytearray)):
        raise ValueError("state workload cases must be a sequence")
    return WorkloadSpec(
        kind=snapshot["kind"],
        source=_strict_json_copy(snapshot["source"], "workload.source"),
        objective=_strict_json_copy(snapshot["objective"], "workload.objective"),
        cases=tuple(_strict_json_copy(cases, "workload.cases")),
        source_hash=snapshot["source_hash"],
    )


def _budget_payload(policy: BudgetPolicy) -> dict:
    return _strict_json_copy(asdict(policy), "budget")


def resolve_setup_policy(args) -> BudgetPolicy:
    confidence = _finite_real(args.confidence, "confidence")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    _finite_real(args.min_effect_pct, "min_effect_pct", minimum=0.0)
    overrides = {}
    iterations = getattr(args, "iterations", None)
    max_rounds = getattr(args, "max_rounds", None)
    if iterations is not None and max_rounds is not None:
        raise ValueError("--iterations conflicts with --max-rounds")
    if iterations is not None:
        max_rounds = _positive_int(iterations, "iterations")
    for field in (
        "max_seconds",
        "max_rounds",
        "branches",
        "min_pairs",
        "max_pairs",
        "outer_candidates",
    ):
        value = max_rounds if field == "max_rounds" else getattr(args, field, None)
        if value is not None:
            overrides[field] = _positive_int(value, field)
    return resolve_budget(args.budget, **overrides)


def _validate_checkpoint(checkpoint, *, input_hash: str | None = None) -> dict:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    clean = _strict_json_copy(checkpoint, "checkpoint")
    required = {
        "schema_version",
        "input_hash",
        "iteration",
        "stage",
        "stage_index",
        "status",
        "candidate_id",
        "candidate_status",
        "budget",
        "updated_at",
    }
    missing = sorted(required - set(clean))
    if missing:
        raise ValueError(f"checkpoint missing required field: {missing[0]}")
    if type(clean["schema_version"]) is not int or clean["schema_version"] != CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint schema_version must be {CURRENT_SCHEMA_VERSION}; start a new run"
        )
    iteration = clean["iteration"]
    if type(iteration) is not int or iteration < 0:
        raise ValueError("checkpoint iteration must be a non-negative integer")
    stage = clean["stage"]
    if type(stage) is not str or stage not in STAGES:
        raise ValueError("checkpoint stage is unknown")
    stage_index = clean["stage_index"]
    if type(stage_index) is not int or stage_index != STAGES.index(stage):
        raise ValueError("checkpoint stage_index does not match stage")
    status = clean["status"]
    if type(status) is not str or status not in _CHECKPOINT_STATUSES:
        raise ValueError("checkpoint status is invalid")
    if stage == "complete" and status != "complete":
        raise ValueError("complete checkpoint stage requires complete status")
    if status == "complete" and stage != "complete":
        raise ValueError("complete checkpoint status requires complete stage")
    if not isinstance(clean["input_hash"], str) or not clean["input_hash"]:
        raise ValueError("checkpoint input_hash must be non-empty")
    if input_hash is not None and clean["input_hash"] != input_hash:
        raise ValueError("checkpoint does not match the frozen input")
    budget = clean["budget"]
    if not isinstance(budget, Mapping):
        raise ValueError("checkpoint budget must be a mapping")
    for field in ("elapsed_seconds", "remaining_seconds"):
        _finite_real(budget.get(field), f"checkpoint budget.{field}", minimum=0.0)
    _finite_real(clean["updated_at"], "checkpoint updated_at")
    candidate_id = clean["candidate_id"]
    if candidate_id is not None and (
        type(candidate_id) is not str or not candidate_id.strip()
    ):
        raise ValueError("checkpoint candidate_id must be null or a non-empty string")
    candidate_status = clean["candidate_status"]
    if candidate_status is not None and (
        type(candidate_status) is not str or not candidate_status.strip()
    ):
        raise ValueError(
            "checkpoint candidate_status must be null or a non-empty string"
        )
    stage_evidence = clean.get("stage_evidence", {})
    if not isinstance(stage_evidence, Mapping):
        raise ValueError("checkpoint stage_evidence must be a mapping")
    for evidence_stage, evidence in stage_evidence.items():
        if evidence_stage not in STAGES:
            raise ValueError("checkpoint stage_evidence contains an unknown stage")
        if not isinstance(evidence, Mapping):
            raise ValueError("checkpoint stage evidence must be a mapping")
        evidence_status = evidence.get("status")
        if type(evidence_status) is not str or not evidence_status.strip():
            raise ValueError("checkpoint stage evidence status must be non-empty")
    declared_run_dir = clean.get("run_dir")
    if declared_run_dir is not None and (
        type(declared_run_dir) is not str or not declared_run_dir.strip()
    ):
        raise ValueError("checkpoint run_dir must be a non-empty path")
    candidate_file = clean.get("candidate_file")
    candidate_hash = clean.get("candidate_sha256")
    if candidate_file is not None or candidate_hash is not None:
        if not isinstance(candidate_file, str) or not candidate_file:
            raise ValueError("checkpoint candidate_file must be a non-empty path")
        if (
            not isinstance(candidate_hash, str)
            or len(candidate_hash) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in candidate_hash)
        ):
            raise ValueError("checkpoint candidate_sha256 must be a sha256 digest")
        candidate = Path(candidate_file).expanduser()
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("checkpoint candidate file drifted or is unsafe")
        if sha256_file(candidate) != candidate_hash.lower():
            raise ValueError("checkpoint candidate file drifted from frozen sha256")
    return clean


def transition_checkpoint(
    checkpoint,
    stage: str,
    *,
    status: str = "stage_complete",
    candidate_id=_UNSET,
    candidate_status=_UNSET,
    budget=None,
    evidence=None,
    iteration=_UNSET,
    updated_at: float | None = None,
) -> dict:
    """Return an ordered, detached checkpoint transition."""
    current = _validate_checkpoint(checkpoint)
    if stage not in STAGES:
        raise ValueError(f"unknown checkpoint stage: {stage}")
    if status not in _CHECKPOINT_STATUSES:
        raise ValueError(f"unknown checkpoint status: {status}")
    current_index = current["stage_index"]
    target_index = STAGES.index(stage)
    wraps_round = (
        current["stage"] == "decision"
        and current["status"] == "stage_complete"
        and stage == "candidate_correctness"
    )
    if target_index < current_index and not wraps_round:
        raise ValueError("checkpoint stage order cannot move backward")
    if target_index > current_index + 1 and not wraps_round:
        raise ValueError("checkpoint stage order cannot skip stages")
    if (
        target_index == current_index
        and current["status"] in {"stage_complete", "complete"}
        and status != current["status"]
    ):
        raise ValueError("checkpoint stage order cannot reopen a completed stage")
    if target_index == current_index + 1 and current["status"] not in {
        "stage_complete",
        "complete",
    }:
        raise ValueError("checkpoint stage order requires the current stage to complete")
    inferred_iteration = current["iteration"]
    if current["stage"] == "baseline" and stage == "candidate_correctness":
        inferred_iteration = 1
    elif wraps_round:
        inferred_iteration = current["iteration"] + 1
    requested_iteration = (
        inferred_iteration if iteration is _UNSET else iteration
    )
    if requested_iteration != inferred_iteration:
        raise ValueError("checkpoint iteration does not match the legal stage wrap")
    result = copy.deepcopy(current)
    result.update(
        {
            "stage": stage,
            "stage_index": target_index,
            "iteration": requested_iteration,
            "status": status,
            "candidate_id": (
                current.get("candidate_id") if candidate_id is _UNSET else candidate_id
            ),
            "candidate_status": (
                current.get("candidate_status")
                if candidate_status is _UNSET
                else candidate_status
            ),
            "updated_at": time.time() if updated_at is None else _finite_real(
                updated_at, "updated_at"
            ),
        }
    )
    if budget is not None:
        result["budget"] = _strict_json_copy(budget, "budget")
    if evidence is not None:
        clean_evidence = _strict_json_copy(evidence, f"stage_evidence.{stage}")
        if not isinstance(clean_evidence, Mapping):
            raise ValueError("stage evidence must be a mapping")
        evidence_status = clean_evidence.get("status")
        if type(evidence_status) is not str or not evidence_status.strip():
            raise ValueError("stage evidence status must be a non-empty string")
        result.setdefault("stage_evidence", {})[stage] = clean_evidence
    if stage == "complete":
        result["status"] = "complete"
    return _validate_checkpoint(result)


checkpoint_transition = transition_checkpoint


def resume(checkpoint, *, input_hash: str, max_rounds: int | None = None) -> dict:
    """Validate and detach resumable state without replaying completed work."""
    current = _validate_checkpoint(checkpoint, input_hash=input_hash)
    result = copy.deepcopy(current)
    if current["stage"] == "complete":
        result["status"] = "complete"
        result["next_stage"] = "complete"
        result["next_iteration"] = current["iteration"]
        return result
    if max_rounds is not None:
        _positive_int(max_rounds, "max_rounds")
    if current["stage"] == "decision" and current["status"] == "stage_complete":
        if max_rounds is not None and current["iteration"] < max_rounds:
            result["next_stage"] = "candidate_correctness"
            result["next_iteration"] = current["iteration"] + 1
        else:
            result["next_stage"] = "complete"
            result["next_iteration"] = current["iteration"]
        return result
    if current["status"] == "stage_complete":
        result["next_stage"] = STAGES[current["stage_index"] + 1]
        result["next_iteration"] = (
            1 if current["stage"] == "baseline" else current["iteration"]
        )
    else:
        result["next_stage"] = current["stage"]
        result["next_iteration"] = current["iteration"]
    return result


def schedule_next(
    state,
    clock: BudgetClock,
    estimated_seconds,
    *,
    now: float | None = None,
    run_dir=None,
    store: ArtifactStore | None = None,
    candidate_id=None,
) -> dict:
    """Admit work against the execution deadline or durably stop the run."""
    if not isinstance(clock, BudgetClock):
        raise ValueError("clock must be a BudgetClock")
    estimate = _finite_real(estimated_seconds, "estimated_seconds", minimum=0.0)
    current_time = time.monotonic() if now is None else _finite_real(now, "now")
    elapsed = clock.elapsed(now=current_time)
    remaining = clock.remaining_seconds(now=current_time)
    if isinstance(state, Mapping) and "stage" not in state:
        checkpoint = {
            "schema_version": state.get("schema_version", CURRENT_SCHEMA_VERSION),
            "input_hash": state.get("input_hash"),
            "run_dir": state.get("run_dir"),
            "iteration": state.get("iteration", 0),
            "stage": STAGES[0],
            "stage_index": 0,
            "status": "ready",
            "candidate_id": None,
            "candidate_status": None,
            "budget": {
                "elapsed_seconds": elapsed,
                "remaining_seconds": remaining,
            },
            "updated_at": current_time,
        }
        checkpoint = _validate_checkpoint(checkpoint)
    else:
        checkpoint = _validate_checkpoint(state)
    if checkpoint["status"] in {"stage_complete", "complete"}:
        raise ValueError(
            "schedule_next requires the next unfinished stage, not a completed stage"
        )
    checkpoint["budget"] = {
        "elapsed_seconds": elapsed,
        "remaining_seconds": remaining,
    }
    checkpoint["updated_at"] = current_time
    normalized_candidate_id = candidate_id
    if normalized_candidate_id is not None:
        normalized_candidate_id = str(normalized_candidate_id).strip()
        if not normalized_candidate_id:
            raise ValueError("candidate_id must be null or a non-empty string")
    checkpoint["checkpoint_written"] = False
    if clock.can_start(now=current_time, estimated_seconds=estimate):
        checkpoint["status"] = "in_progress"
        checkpoint["candidate_id"] = normalized_candidate_id
        return checkpoint

    checkpoint["status"] = "budget_exhausted"
    checkpoint["candidate_id"] = normalized_candidate_id
    checkpoint["candidate_status"] = "inconclusive"
    selected_store = store
    if selected_store is None:
        selected_root = run_dir or checkpoint.get("run_dir")
        if selected_root is not None:
            selected_store = ArtifactStore(selected_root)
    if selected_store is not None:
        persisted = copy.deepcopy(checkpoint)
        persisted["checkpoint_written"] = True
        try:
            selected_store.write_checkpoint(persisted)
        except Exception as error:
            checkpoint["checkpoint_error"] = f"{type(error).__name__}: {error}"
        else:
            checkpoint = persisted
    return checkpoint


def _checkpoint_budget(clock: BudgetClock, now: float) -> dict:
    return {
        "elapsed_seconds": clock.elapsed(now=now),
        "remaining_seconds": clock.remaining_seconds(now=now),
    }


def _persist_checkpoint(
    store: ArtifactStore, checkpoint: Mapping, *, input_hash: str
) -> dict:
    """Atomically write, reload, and validate a checkpoint before continuing."""
    store.write_checkpoint(dict(checkpoint))
    reloaded = store.load_checkpoint(expected_input_hash=input_hash)
    return _validate_checkpoint(reloaded, input_hash=input_hash)


def _position_checkpoint(
    checkpoint: Mapping,
    stage: str,
    *,
    store: ArtifactStore,
    clock: BudgetClock,
    candidate_id=None,
    now: float | None = None,
) -> dict:
    current = _validate_checkpoint(checkpoint)
    current_time = time.monotonic() if now is None else _finite_real(now, "now")
    normalized_id = None if candidate_id is None else str(candidate_id).strip()
    if normalized_id == "":
        raise ValueError("candidate_id must be null or a non-empty string")
    target_index = STAGES.index(stage)
    if current["stage_index"] == target_index:
        if current["status"] == "stage_complete":
            return current
        positioned = copy.deepcopy(current)
        positioned["status"] = "ready"
        positioned["candidate_id"] = normalized_id
        positioned["candidate_status"] = None
        positioned["budget"] = _checkpoint_budget(clock, current_time)
        positioned["updated_at"] = current_time
    else:
        positioned = transition_checkpoint(
            current,
            stage,
            status="ready",
            candidate_id=normalized_id,
            candidate_status=None,
            budget=_checkpoint_budget(clock, current_time),
            updated_at=current_time,
        )
    return _persist_checkpoint(
        store, positioned, input_hash=positioned["input_hash"]
    )


def _admit_checkpoint_stage(
    checkpoint: Mapping,
    stage: str,
    *,
    store: ArtifactStore,
    clock: BudgetClock,
    estimated_seconds: float,
    candidate_id=None,
    now: float | None = None,
) -> tuple[bool, dict]:
    current_time = time.monotonic() if now is None else _finite_real(now, "now")
    positioned = _position_checkpoint(
        checkpoint,
        stage,
        store=store,
        clock=clock,
        candidate_id=candidate_id,
        now=current_time,
    )
    admitted = schedule_next(
        positioned,
        clock,
        estimated_seconds,
        now=current_time,
        store=store,
        candidate_id=candidate_id,
    )
    if admitted["status"] == "budget_exhausted":
        return False, _validate_checkpoint(
            store.load_checkpoint(expected_input_hash=positioned["input_hash"]),
            input_hash=positioned["input_hash"],
        )
    admitted.pop("checkpoint_written", None)
    admitted = _persist_checkpoint(
        store, admitted, input_hash=positioned["input_hash"]
    )
    return True, admitted


def _complete_checkpoint_stage(
    checkpoint: Mapping,
    stage: str,
    evidence: Mapping,
    *,
    store: ArtifactStore,
    clock: BudgetClock,
    candidate_id=None,
    candidate_status=None,
    now: float | None = None,
) -> dict:
    current_time = time.monotonic() if now is None else _finite_real(now, "now")
    positioned = _position_checkpoint(
        checkpoint,
        stage,
        store=store,
        clock=clock,
        candidate_id=candidate_id,
        now=current_time,
    )
    completed = transition_checkpoint(
        positioned,
        stage,
        status="stage_complete",
        candidate_id=(None if candidate_id is None else str(candidate_id).strip()),
        candidate_status=candidate_status,
        budget=_checkpoint_budget(clock, current_time),
        evidence=evidence,
        updated_at=current_time,
    )
    return _persist_checkpoint(
        store, completed, input_hash=completed["input_hash"]
    )


def execute_stage(
    checkpoint,
    stage: str,
    action,
    *,
    store: ArtifactStore,
    cleanup=None,
    updated_at: float | None = None,
):
    """Execute one stage and checkpoint success or a safe interruption."""
    if not callable(action):
        raise ValueError("action must be callable")
    if cleanup is not None and not callable(cleanup):
        raise ValueError("cleanup must be callable")
    current = _validate_checkpoint(checkpoint)
    if current["stage"] != stage:
        raise ValueError("execute_stage must run the checkpoint current stage")
    cleaned = False

    def clean_once():
        nonlocal cleaned
        if cleanup is not None and not cleaned:
            cleaned = True
            cleanup()

    try:
        value = action()
    except BaseException as error:
        try:
            clean_once()
        finally:
            failed = transition_checkpoint(
                current,
                stage,
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                candidate_id=current.get("candidate_id"),
                candidate_status=current.get("candidate_status"),
                updated_at=updated_at,
            )
            store.write_checkpoint(failed)
        raise
    completed = transition_checkpoint(
        current,
        stage,
        status="stage_complete",
        candidate_id=current.get("candidate_id"),
        candidate_status=current.get("candidate_status"),
        updated_at=updated_at,
    )
    store.write_checkpoint(completed)
    return value, completed


def select_outer_candidates(items, limit: int) -> list[dict]:
    """Return the stable top confirmed candidates without mutating input."""
    _positive_int(limit, "limit")
    if isinstance(items, (str, bytes, bytearray, Mapping)):
        raise ValueError("items must be a sequence of candidate mappings")
    try:
        source = list(items)
    except TypeError as error:
        raise ValueError("items must be a sequence of candidate mappings") from error
    selected = []
    for index, item in enumerate(source):
        if not isinstance(item, Mapping):
            raise ValueError(f"items[{index}] must be a mapping")
        if item.get("status") != "confirmed_win":
            continue
        statistics = item.get("statistics")
        if not isinstance(statistics, Mapping) or "estimate_pct" not in statistics:
            raise ValueError(f"items[{index}] confirmed win requires estimate_pct")
        estimate = _finite_real(
            statistics["estimate_pct"], f"items[{index}].statistics.estimate_pct"
        )
        selected.append((estimate, index, _strict_json_copy(item, f"items[{index}]")))
    selected.sort(key=lambda row: -row[0])
    return [item for _, _, item in selected[:limit]]


def _absolute_lexical_path(value: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(value)))
    parts = path.parts
    if len(parts) > 1:
        root_entry = Path(path.anchor) / parts[1]
        try:
            metadata = os.lstat(root_entry)
        except OSError:
            metadata = None
        if (
            metadata is not None
            and stat.S_ISLNK(metadata.st_mode)
            and metadata.st_uid == 0
        ):
            target = os.readlink(root_entry)
            root_target = Path(target)
            if not root_target.is_absolute():
                root_target = Path(path.anchor) / root_target
            path = Path(os.path.normpath(root_target)) / Path(*parts[2:])
    return path


def _candidate_snapshot(candidate: Mapping) -> dict:
    value = candidate.get("candidate_file") or candidate.get("kernel")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("candidate must provide candidate_file or kernel")
    path = _absolute_lexical_path(value)
    payload, bench_payload = read_regular_with_optional_sibling(
        path, "bench.json"
    )
    return {
        "path": str(path),
        "payload": payload,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bench": (
            None
            if bench_payload is None
            else {
                "payload": bench_payload,
                "sha256": hashlib.sha256(bench_payload).hexdigest(),
            }
        ),
    }


def _candidate_file(candidate: Mapping) -> str:
    return _candidate_snapshot(candidate)["path"]


def _candidate_identifier(candidate: Mapping, fallback: str | None = None):
    candidate_id = candidate.get("id")
    if candidate_id is None:
        candidate_id = candidate.get("branch_index")
    if candidate_id is None:
        candidate_id = fallback
    return candidate_id


def build_terminal_decision(
    *,
    mode: str,
    candidate: Mapping,
    workload_result=None,
    decide_fn=None,
    _snapshot=None,
) -> dict:
    """Build the authoritative terminal decision bound to candidate bytes."""
    selected_decider = decision_engine.decide if decide_fn is None else decide_fn
    kernel = {
        "status": candidate.get("status"),
        "statistics": _strict_json_copy(candidate.get("statistics"), "statistics"),
    }
    result = selected_decider(
        mode=mode,
        kernel=kernel,
        workload=workload_result,
        constraints=None,
    )
    if isinstance(kernel.get("statistics"), Mapping):
        result.setdefault(
            "statistics",
            _strict_json_copy(kernel["statistics"], "kernel statistics"),
        )
    snapshot = _candidate_snapshot(candidate) if _snapshot is None else _snapshot
    result["candidate_file"] = snapshot["path"]
    result["candidate_sha256"] = snapshot["sha256"]
    kernel_samples = candidate.get("paired_samples") or candidate.get(
        "kernel_paired_samples"
    )
    if isinstance(kernel_samples, Mapping):
        result["kernel_paired_samples"] = _strict_json_copy(
            kernel_samples, "kernel paired samples"
        )
    candidate_id = _candidate_identifier(candidate)
    if candidate_id is None and isinstance(kernel_samples, Mapping):
        candidate_id = kernel_samples.get("candidate_id")
    if candidate_id is None:
        candidate_id = result["candidate_sha256"][:16]
    if isinstance(candidate_id, bool) or not str(candidate_id).strip():
        raise ValueError("candidate_id must be non-empty")
    result["candidate_id"] = str(candidate_id).strip()
    workload_samples = workload_result if isinstance(workload_result, Mapping) else None
    if workload_samples is not None and isinstance(
        workload_samples.get("paired_samples"), Mapping
    ):
        result["workload_paired_samples"] = _strict_json_copy(
            workload_samples["paired_samples"], "workload paired samples"
        )
    return _strict_json_copy(result, "decision")


def evaluate_outer_candidate(
    candidate: Mapping,
    *,
    mode: str,
    workload_spec: WorkloadSpec | None,
    baseline,
    policy: BudgetPolicy,
    confidence: float,
    evaluator=None,
    remaining_seconds: float | None = None,
    estimated_seconds_per_pair: float = 0.0,
    budget_clock: BudgetClock | None = None,
    now: float | None = None,
    candidate_root=None,
    retries: int = 2,
    seed: int = 0,
    workload_runner=None,
    input_hash: str | None = None,
    iteration: int | None = None,
) -> dict:
    """Evaluate one confirmed inner winner through the applicable outer loop."""
    candidate_snapshot = _candidate_snapshot(candidate)
    candidate_path = Path(candidate_snapshot["path"])
    if candidate_path.suffix not in {".cu", ".py"}:
        raise ValueError("outer candidate must be a .cu or .py kernel")
    iteration_root = None
    if candidate_root is not None:
        iteration_root = Path(candidate_root).expanduser().resolve(strict=True)
        try:
            candidate_path.relative_to(iteration_root)
        except ValueError as error:
            raise ValueError("outer candidate escapes the current iteration") from error
    normalized_candidate = dict(candidate)
    normalized_candidate["candidate_file"] = str(candidate_path)
    if mode == "kernel-only":
        return build_terminal_decision(
            mode=mode,
            candidate=normalized_candidate,
            _snapshot=candidate_snapshot,
        )
    if mode != "full" or workload_spec is None:
        raise ValueError("full mode requires a frozen WorkloadSpec")
    blocks = policy.max_pairs
    pair_estimate = _finite_real(
        estimated_seconds_per_pair, "estimated_seconds_per_pair", minimum=0.0
    )
    budget_exhausted = False
    if budget_clock is not None:
        if not isinstance(budget_clock, BudgetClock):
            raise ValueError("budget_clock must be a BudgetClock")
        current_time = time.monotonic() if now is None else _finite_real(now, "now")
        if not budget_clock.can_start(
            now=current_time,
            estimated_seconds=policy.min_pairs * pair_estimate,
        ):
            budget_exhausted = True
        if pair_estimate > 0.0:
            execution_remaining = budget_clock.execution_seconds_available(
                now=current_time
            )
            blocks = min(blocks, int(execution_remaining // pair_estimate))
    if remaining_seconds is not None:
        remaining = _finite_real(
            remaining_seconds, "remaining_seconds", minimum=0.0
        )
        if pair_estimate > 0.0:
            blocks = min(blocks, int(remaining // pair_estimate))
    if budget_exhausted or blocks < policy.min_pairs:
        return _strict_json_copy(
            {
                "status": "inconclusive",
                "candidate_status": "inconclusive",
                "budget_exhausted": True,
                "candidate_file": str(candidate_path),
                "candidate_sha256": candidate_snapshot["sha256"],
                "statistics": None,
                "kernel_evidence": {
                    "status": normalized_candidate.get("status"),
                    "statistics": normalized_candidate.get("statistics"),
                },
                "reason": "workload_budget_exhausted",
            },
            "decision",
        )
    else:
        selected_evaluator = (
            workload_evaluate.evaluate_pairs if evaluator is None else evaluator
        )
        evaluation_spec = workload_spec
        if policy.max_cases is not None and len(workload_spec.cases) > policy.max_cases:
            evaluation_spec = WorkloadSpec(
                kind=workload_spec.kind,
                source=workload_spec.source,
                objective=workload_spec.objective,
                cases=tuple(workload_spec.cases[: policy.max_cases]),
                source_hash=workload_spec.source_hash,
            )
        selected_workload_runner = (
            run_spec_once if workload_runner is None else workload_runner
        )
        if not callable(selected_workload_runner):
            raise ValueError("workload_runner must be callable")

        def frozen_runner(
            _evaluation_spec,
            *,
            candidate,
            role,
            case=None,
            timeout=None,
        ):
            return selected_workload_runner(
                workload_spec,
                candidate=candidate,
                role=role,
                case=case,
                timeout=timeout,
            )

        workload_result = selected_evaluator(
            evaluation_spec,
            baseline,
            str(candidate_path),
            blocks=blocks,
            retries=retries,
            seed=seed,
            confidence=_finite_real(confidence, "confidence"),
            runner=frozen_runner,
        )
        if input_hash is not None or iteration is not None:
            if not isinstance(workload_result, Mapping):
                raise ValueError("workload result must be a mapping")
            pairs = workload_result.get("pairs")
            if not isinstance(pairs, list):
                raise ValueError("workload result must contain raw pairs")
            if candidate_root is None or input_hash is None or iteration is None:
                raise ValueError("workload pair persistence binding is incomplete")
            candidate_digest = candidate_snapshot["sha256"]
            candidate_id = _candidate_identifier(
                normalized_candidate, candidate_digest[:16]
            )
            evidence_path = (
                iteration_root
                / "workload"
                / candidate_digest[:16]
                / "paired_samples.jsonl"
            )
            evidence = write_paired_samples(
                evidence_path,
                pairs,
                kind="workload",
                input_hash=input_hash,
                iteration=iteration,
                candidate_id=candidate_id,
                candidate_file=candidate_path,
                classifier_config={
                    "objective": _strict_json_copy(
                        workload_result.get("objective"), "workload objective"
                    ),
                    "objective_sha256": hashlib.sha256(
                        json.dumps(
                            workload_result.get("objective"),
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                            allow_nan=False,
                        ).encode("utf-8")
                    ).hexdigest(),
                    "confidence": workload_result.get("confidence", confidence),
                    "bootstrap_samples": workload_result.get(
                        "bootstrap_samples",
                        workload_evaluate.DEFAULT_BOOTSTRAP_SAMPLES,
                    ),
                    "seed": workload_result.get("seed", seed),
                },
            )
            workload_result = copy.deepcopy(dict(workload_result))
            workload_result.pop("pairs", None)
            workload_result["paired_samples"] = evidence
    terminal = build_terminal_decision(
        mode=mode,
        candidate=normalized_candidate,
        workload_result=workload_result,
        _snapshot=candidate_snapshot,
    )
    return terminal


def apply_decision(
    decision_payload,
    *,
    run_dir,
    iteration: int,
    state_path,
    kernel,
    bench,
    methods_json,
    retries: int = 0,
    attribution=None,
    sass_check=None,
    skip_validation: bool = False,
    hard_timeout: float | None = None,
    runner=None,
) -> dict:
    """Atomically publish a decision, then invoke state update with its path."""
    _positive_int(iteration, "iteration")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a non-negative integer")
    root = Path(run_dir).expanduser().resolve(strict=True)
    iter_dir = (root / f"iterv{iteration}").resolve(strict=True)
    if iter_dir.parent != root:
        raise ValueError("iteration directory escapes run root")
    decision_path = iter_dir / "decision.json"
    payload = _strict_json_copy(decision_payload, "decision")
    atomic_write_json(decision_path, payload)

    command = [
        sys.executable,
        str(SCRIPT_DIR / "state.py"),
        "update",
        "--state",
        str(state_path),
        "--iter",
        str(iteration),
        "--kernel",
        str(kernel),
        "--bench",
        str(bench),
        "--methods-json",
        str(methods_json),
        "--retries",
        str(retries),
        "--decision",
        str(decision_path),
    ]
    if attribution is not None and Path(attribution).is_file():
        command.extend(["--attribution", str(attribution)])
    if sass_check is not None and Path(sass_check).is_file():
        command.extend(["--sass-check", str(sass_check)])
    if skip_validation:
        command.append("--skip-validation")
    selected_runner = _run if runner is None else runner
    run_options = {}
    if hard_timeout is not None:
        run_options = {
            "capture_output": True,
            "hard_timeout": _finite_real(
                hard_timeout, "hard_timeout", minimum=0.0
            ),
        }
    completed = selected_runner(command, **run_options)
    return {
        "decision_path": str(decision_path),
        "command": command,
        "returncode": completed.returncode,
        "stdout": getattr(completed, "stdout", None),
        "stderr": getattr(completed, "stderr", None),
        "timed_out": bool(getattr(completed, "timed_out", False)),
    }


_SECRET_MARKERS = (
    "TOKEN", "SECRET", "PASSWORD", "COOKIE", "CREDENTIAL", "AUTH", "API_KEY"
)
_SENSITIVE_FLAGS = {
    "--token", "--secret", "--password", "--cookie", "--credential",
    "--authorization", "--api-key",
}


def _redacted_argv(cmd: Sequence[str]) -> list[str]:
    result = []
    redact_next = False
    for raw in cmd:
        value = str(raw)
        upper = value.upper()
        if redact_next:
            result.append("[REDACTED]")
            redact_next = False
            continue
        if value.lower() in _SENSITIVE_FLAGS:
            result.append(value)
            redact_next = True
            continue
        if any(marker in upper for marker in _SECRET_MARKERS):
            if "=" in value:
                result.append(value.split("=", 1)[0] + "=[REDACTED]")
            else:
                result.append("[REDACTED]")
            continue
        result.append(value)
    return result


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"[run] {' '.join(_redacted_argv(cmd))}", file=sys.stderr)
    hard_timeout = kw.pop("hard_timeout", None)
    if hard_timeout is None:
        return subprocess.run(cmd, text=True, **kw)
    timeout_seconds = _finite_real(
        hard_timeout, "hard_timeout", minimum=0.0
    )
    grace_seconds = _finite_real(
        kw.pop("term_grace", 2.0), "term_grace", minimum=0.0
    )
    if "timeout" in kw:
        raise ValueError("hard_timeout cannot be combined with timeout")
    capture_output = kw.pop("capture_output", False)
    if capture_output:
        if kw.get("stdout") is not None or kw.get("stderr") is not None:
            raise ValueError("capture_output conflicts with stdout or stderr")
        kw["stdout"] = subprocess.PIPE
        kw["stderr"] = subprocess.PIPE
    check = kw.pop("check", False)
    input_value = kw.pop("input", None)
    return run_budgeted_command(
        cmd,
        timeout_seconds=timeout_seconds,
        grace_seconds=grace_seconds,
        input_value=input_value,
        check=check,
        popen_options=kw,
    )


def _read(path: str | os.PathLike) -> dict:
    try:
        return json.loads(read_regular_bytes(path).decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError(f"JSON artifact is not UTF-8: {path}") from error


def _read_branch_results(path: str) -> dict:
    if not os.path.isfile(path):
        sys.exit(f"branch_results artifact missing: {path}")
    try:
        payload = _read(path)
    except (OSError, json.JSONDecodeError) as error:
        sys.exit(f"branch_results artifact malformed: {error}")
    if not isinstance(payload, dict):
        sys.exit("branch_results artifact must be a JSON object")
    status = payload.get("status")
    if type(status) is not str or status not in _BRANCH_RESULT_STATUSES:
        sys.exit(f"branch_results artifact status is invalid: {status!r}")
    return payload


def _selected_kernel(
    branch_payload: dict, *, iter_dir: str, decision_path: str
) -> str:
    """Resolve the champion solely from the authoritative branch artifact."""
    declared = branch_payload.get("selected_kernel")
    if not isinstance(declared, str) or not declared.strip():
        sys.exit("branch_results selected_kernel must be a non-empty path")

    candidate = Path(declared).expanduser()
    if candidate.is_symlink():
        sys.exit(f"selected_kernel must not be a symlink: {candidate}")
    try:
        candidate_stat = candidate.lstat()
    except OSError as error:
        sys.exit(f"selected_kernel is missing or unreadable: {error}")
    if not stat.S_ISREG(candidate_stat.st_mode):
        sys.exit(f"selected_kernel must be a regular file: {candidate}")

    iteration = Path(iter_dir).expanduser().resolve()
    resolved = candidate.resolve(strict=True)
    if resolved.parent != iteration or resolved.name not in {"kernel.cu", "kernel.py"}:
        sys.exit(
            "selected_kernel must be the current iteration kernel.cu or kernel.py"
        )

    try:
        decision = _read(decision_path)
    except (OSError, json.JSONDecodeError) as error:
        sys.exit(f"decision candidate_file cannot be validated: {error}")
    if not isinstance(decision, dict):
        sys.exit("decision candidate_file cannot be validated")
    decision_candidate = decision.get("candidate_file")
    if not isinstance(decision_candidate, str) or not decision_candidate.strip():
        sys.exit("decision candidate_file must be a non-empty path")
    if os.path.abspath(os.path.expanduser(decision_candidate)) != os.path.abspath(
        str(candidate)
    ):
        sys.exit("decision candidate_file does not match selected_kernel")
    return os.path.abspath(str(candidate))


def _policy_from_state(state: Mapping) -> BudgetPolicy:
    budget = state.get("budget")
    if not isinstance(budget, Mapping):
        raise ValueError("state budget must be a mapping")
    fields = {
        name: budget.get(name)
        for name in (
            "name",
            "soft_target_seconds",
            "max_seconds",
            "branches",
            "max_rounds",
            "min_pairs",
            "max_pairs",
            "outer_candidates",
            "max_cases",
            "sanitizer_mode",
            "reserve_seconds",
        )
    }
    if fields["soft_target_seconds"] is None and isinstance(
        fields["max_seconds"], int
    ):
        fields["soft_target_seconds"] = max(1, fields["max_seconds"] // 3)
    try:
        policy = BudgetPolicy(**fields)
    except TypeError as error:
        raise ValueError(f"state budget policy is incomplete: {error}") from error
    # Reuse budget.py's public validator through an exact custom resolution.
    return resolve_budget(
        "custom",
        max_seconds=policy.max_seconds,
        branches=policy.branches,
        max_rounds=policy.max_rounds,
        min_pairs=policy.min_pairs,
        max_pairs=policy.max_pairs,
        outer_candidates=policy.outer_candidates,
        max_cases=policy.max_cases,
        sanitizer_mode=policy.sanitizer_mode,
        reserve_seconds=policy.reserve_seconds,
        soft_target_seconds=policy.soft_target_seconds,
    )


def _publish_outer_candidate(
    candidate: Mapping, *, iter_dir: Path, _snapshot=None
) -> str:
    snapshot = _candidate_snapshot(candidate) if _snapshot is None else _snapshot
    source = Path(snapshot["path"])
    suffix = source.suffix
    if suffix not in {".cu", ".py"}:
        raise ValueError("outer candidate must be a .cu or .py kernel")
    destination = iter_dir / f"kernel{suffix}"
    if hashlib.sha256(snapshot["payload"]).hexdigest() != snapshot["sha256"]:
        raise ValueError("candidate snapshot payload does not match sha256")
    writes = {destination.name: snapshot["payload"]}
    removals = [
        f"kernel{stale_suffix}"
        for stale_suffix in {".cu", ".py"} - {suffix}
    ]
    bench_snapshot = snapshot.get("bench")
    if bench_snapshot is not None:
        if (
            hashlib.sha256(bench_snapshot["payload"]).hexdigest()
            != bench_snapshot["sha256"]
        ):
            raise ValueError("bench snapshot payload does not match sha256")
        writes["bench.json"] = bench_snapshot["payload"]
    else:
        removals.append("bench.json")
    published_hashes = publish_regular_bundle(iter_dir, writes, removals)
    if published_hashes[destination.name] != snapshot["sha256"]:
        raise ValueError("published candidate does not match captured snapshot")
    if (
        bench_snapshot is not None
        and published_hashes["bench.json"] != bench_snapshot["sha256"]
    ):
        raise ValueError("published bench does not match captured snapshot")
    return str(Path(os.path.abspath(destination)))


def _select_terminal_outer_result(results: list[tuple[dict, dict]]) -> tuple[dict, dict]:
    if not results:
        raise ValueError("outer loop did not produce a terminal decision")
    end_to_end = [
        pair for pair in results if pair[1].get("status") == "end_to_end_win"
    ]
    if end_to_end:
        def estimate(pair):
            statistics = pair[1].get("workload_statistics") or {}
            value = statistics.get("estimate_pct", float("-inf"))
            return _finite_real(value, "workload_statistics.estimate_pct")

        end_to_end.sort(key=estimate, reverse=True)
        return end_to_end[0]
    return results[0]


# ---------------------------------------------------------------------------
# setup  —  steps 0, 1, 2, and open-iter(1)
# ---------------------------------------------------------------------------

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
        value = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=nonfinite,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} must be valid strict JSON: {error}") from error
    return value


def _strict_json_object(text: str, field: str) -> dict:
    value = _strict_json_value(text, field)
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def validate_output_root(output_root, *, baseline) -> Path:
    """Resolve an existing non-symlink directory that may own new runs."""
    if output_root is None:
        root_arg = Path(baseline).expanduser().resolve(strict=True).parent
    else:
        root_arg = Path(output_root).expanduser()
        if root_arg.is_symlink():
            raise ValueError("output_root must not be a symlink")
    try:
        info = root_arg.lstat()
    except OSError as error:
        raise ValueError(f"output_root does not exist: {root_arg}") from error
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("output_root must be a directory")
    root = root_arg.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("output_root must resolve to a directory")
    return root


def _allocate_run_dir(output_root: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    nanos = time.time_ns() % 1_000_000_000
    for suffix in range(1000):
        name = f"run_{stamp}_{nanos:09d}"
        if suffix:
            name += f"_{suffix}"
        run_dir = output_root / name
        try:
            run_dir.mkdir(mode=0o755)
        except FileExistsError:
            continue
        resolved = run_dir.resolve(strict=True)
        if resolved.parent != output_root:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise ValueError("allocated run directory escaped output_root")
        return resolved
    raise ValueError("could not allocate a unique run directory")


def _frozen_input_hash(
    manifest: Mapping,
    *,
    workload,
    dims,
    backend: str,
    budget,
    confidence: float,
    min_effect_pct: float,
    ptr_size: int,
) -> str:
    input_digests = {
        key: value["sha256"]
        for key, value in sorted(manifest["inputs"].items())
    }
    frozen = {
        "inputs": input_digests,
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


def cmd_setup(args):
    started_at = time.time()
    policy = resolve_setup_policy(args)
    budget_clock = BudgetClock(policy, started_at=time.monotonic())
    confidence = _finite_real(args.confidence, "confidence")
    min_effect_pct = _finite_real(
        args.min_effect_pct, "min_effect_pct", minimum=0.0
    )
    dims = _strict_json_object(args.dims, "--dims")
    baseline = str(Path(args.baseline).expanduser().resolve(strict=True))
    ref = str(Path(args.ref).expanduser().resolve(strict=True))

    # Normalize exactly the caller-supplied workload form before preflight.
    workload_spec = normalize_workload(
        workload=args.workload,
        workload_cmd=args.workload_cmd,
        workload_manifest=args.workload_manifest,
        objective=args.objective,
    )
    report = preflight.run(
        baseline,
        ref,
        copy.deepcopy(dims),
        False,
        workload_spec,
    )
    if not report.get("ok"):
        errors = report.get("errors") or ["unknown preflight failure"]
        raise ValueError("preflight failed: " + "; ".join(map(str, errors)))

    output_root = validate_output_root(args.output_root, baseline=baseline)
    run_dir = _allocate_run_dir(output_root)
    try:
        env_path = run_dir / "env.json"
        environment_result = _run(
            [
                sys.executable,
                str(SCRIPT_DIR / "check_env.py"),
                "--out",
                str(env_path),
            ]
        )
        if environment_result.returncode != 0:
            raise ValueError(f"check_env failed rc={environment_result.returncode}")
        environment = _read(str(env_path)) if env_path.is_file() else {}
        if not isinstance(environment, Mapping):
            raise ValueError("env.json must contain a JSON object")

        budget_payload = _budget_payload(policy)
        workload_snapshot = _workload_snapshot(workload_spec)
        mode = "full" if workload_spec is not None else "kernel-only"
        store = ArtifactStore(run_dir)
        manifest = store.initialize(
            inputs={"baseline": baseline, "ref": ref},
            budget=budget_payload,
            environment=dict(environment),
        )
        manifest.update(
            {
                "mode": mode,
                "workload": workload_snapshot,
                "confidence": confidence,
                "min_effect_pct": min_effect_pct,
                "started_at": started_at,
                "dims": copy.deepcopy(dims),
                "backend": args.backend,
                "ptr_size": args.ptr_size,
            }
        )
        manifest["input_hash"] = _frozen_input_hash(
            manifest,
            workload=workload_snapshot,
            dims=dims,
            backend=args.backend,
            budget=budget_payload,
            confidence=confidence,
            min_effect_pct=min_effect_pct,
            ptr_size=args.ptr_size,
        )
        atomic_write_json(run_dir / "manifest.json", manifest)
        workload_file = run_dir / "workload" / "spec.json"
        atomic_write_json(workload_file, workload_snapshot)
        os.chmod(workload_file, 0o600)

        state_command = [
            sys.executable,
            str(SCRIPT_DIR / "state.py"),
            "init",
            "--baseline",
            baseline,
            "--ref",
            ref,
            "--dims",
            json.dumps(dims, sort_keys=True, allow_nan=False),
            "--env",
            str(env_path),
            "--ptr-size",
            str(args.ptr_size),
            "--ncu-num",
            str(args.ncu_num),
            "--output-root",
            str(output_root),
            "--budget-json",
            json.dumps(budget_payload, sort_keys=True, allow_nan=False),
            "--run-dir",
            str(run_dir),
            "--manifest",
            str(run_dir / "manifest.json"),
            "--mode",
            mode,
            "--workload-file",
            str(workload_file),
            "--started-at",
            str(started_at),
            "--backend",
            args.backend,
            "--confidence",
            str(confidence),
            "--min-effect-pct",
            str(min_effect_pct),
        ]
        initialized = _run(state_command, capture_output=True)
        if initialized.returncode != 0:
            raise ValueError(
                "state init failed rc="
                f"{initialized.returncode}: {(initialized.stderr or '').strip()}"
            )
        try:
            initialized_output = json.loads(initialized.stdout)
            state_path = Path(initialized_output["state"]).resolve(strict=True)
            state = _read(str(state_path))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"state init returned malformed output: {error}") from error
        state_manager.validate_state(state)
        if state_path != (run_dir / "state.json").resolve(strict=True):
            raise ValueError("state init returned a state outside the run directory")

        baseline_result = _run(
            [
                sys.executable,
                str(SCRIPT_DIR / "run_iteration.py"),
                "seed-baseline",
                "--state",
                str(state_path),
                "--benchmark",
                os.path.abspath(args.benchmark),
                "--warmup",
                str(args.warmup),
                "--repeat",
                str(args.repeat),
            ],
            capture_output=True,
            hard_timeout=budget_clock.execution_seconds_available(
                now=time.monotonic()
            ),
        )
        if getattr(baseline_result, "timed_out", False):
            raise ValueError("baseline seed exceeded the hard execution deadline")
        if baseline_result.returncode != 0:
            raise ValueError(
                "baseline seed failed rc="
                f"{baseline_result.returncode}: {(baseline_result.stderr or '').strip()}"
            )
        baseline_bench_path = run_dir / "baseline" / "bench.json"
        if baseline_bench_path.is_symlink():
            raise ValueError("baseline bench.json must not be a symlink")
        try:
            baseline_info = baseline_bench_path.lstat()
        except OSError as error:
            raise ValueError("baseline bench.json is missing") from error
        if (
            not stat.S_ISREG(baseline_info.st_mode)
            or baseline_bench_path.resolve(strict=True).parent != run_dir / "baseline"
        ):
            raise ValueError("baseline bench.json must be a run-local regular file")
        baseline_bench = _strict_json_value(
            baseline_bench_path.read_text("utf-8"), "baseline bench.json"
        )
        if not isinstance(baseline_bench, Mapping):
            raise ValueError("baseline bench.json must contain a JSON object")
        if baseline_bench.get("correctness", {}).get("passed") is not True:
            raise ValueError("baseline correctness.passed must be literal true")
        baseline_metric = _finite_real(
            baseline_bench.get("kernel", {}).get("average_ms"),
            "baseline kernel.average_ms",
        )
        if baseline_metric <= 0.0:
            raise ValueError("baseline kernel.average_ms must be positive")
        state = _read(str(state_path))
        state_manager.validate_state(state)
        state_metric = _finite_real(
            state.get("best_metric_ms"), "state best_metric_ms"
        )
        if state_metric != baseline_metric:
            raise ValueError("state best_metric_ms does not match baseline evidence")
        budget_now = time.monotonic()
        now = time.time()
        checkpoint = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "input_hash": manifest["input_hash"],
            "run_dir": str(run_dir),
            "iteration": 0,
            "stage": STAGES[0],
            "stage_index": 0,
            "status": "stage_complete",
            "candidate_id": None,
            "candidate_status": None,
            "stage_evidence": {
                "baseline": {
                    "status": "passed",
                    "bench_path": str(baseline_bench_path.resolve(strict=True)),
                    "bench_sha256": sha256_file(baseline_bench_path),
                    "metric_ms": baseline_metric,
                    "correctness_passed": True,
                }
            },
            "budget": _checkpoint_budget(budget_clock, budget_now),
            "updated_at": now,
            "checkpoint_written": True,
        }
        store.write_checkpoint(checkpoint)
    except BaseException:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "state": str(state_path),
                "manifest": str(run_dir / "manifest.json"),
                "checkpoint": str(run_dir / "checkpoint.json"),
                "env": str(env_path),
                "mode": mode,
                "input_hash": state["input_hash"],
                "next_stage": "candidate_correctness",
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# open-iter  —  profile + roofline for iteration N (if not done by setup)
# ---------------------------------------------------------------------------

def cmd_open_iter(args):
    state_path = os.path.join(args.run_dir, "state.json")
    if not os.path.isfile(state_path):
        sys.exit(f"state.json missing: {state_path}")

    state = _read(state_path)

    # Profile best_input for this iter
    rc = _run([
        sys.executable, str(SCRIPT_DIR / "profile_ncu.py"),
        "--state", state_path,
        "--iter", str(args.iter),
        "--which", "best_input",
        "--benchmark", os.path.abspath(args.benchmark),
    ]).returncode
    if rc != 0:
        print("[warn] ncu profiling failed or degraded", file=sys.stderr)

    # Roofline
    rc = _run([
        sys.executable, str(SCRIPT_DIR / "roofline.py"),
        "--state", state_path,
        "--iter", str(args.iter),
    ]).returncode

    # Check early stop
    iter_dir = os.path.join(args.run_dir, f"iterv{args.iter}")
    roofline_path = os.path.join(iter_dir, "roofline.json")
    early_stop = False
    if os.path.isfile(roofline_path):
        roofline = _read(roofline_path)
        early_stop = roofline.get("near_peak", False)

    # Create branch dirs
    num_branches = state.get("branches", 4)
    branches_dir = os.path.join(iter_dir, "branches")
    for b in range(1, num_branches + 1):
        os.makedirs(os.path.join(branches_dir, f"b{b}"), exist_ok=True)

    print(json.dumps({
        "iter": args.iter,
        "early_stop": early_stop,
        "branches_dir": branches_dir,
        "num_branches": num_branches,
        "next_step": (
            f"The agent should read iterv{args.iter}/roofline.json and ncu_top.json, "
            f"write {num_branches} branch kernels under iterv{args.iter}/branches/b{{1..{num_branches}}}/kernel.<ext>, "
            f"plus iterv{args.iter}/methods.json and iterv{args.iter}/analysis.md. "
            f"Then run: orchestrate.py close-iter --run-dir {args.run_dir} --iter {args.iter}"
        ) if not early_stop else "Near roofline — consider stopping.",
    }, indent=2))


# ---------------------------------------------------------------------------
# close-iter  —  branch explore → ncu champion → ablate → sass → update
# ---------------------------------------------------------------------------

_STAGE_ESTIMATES_SECONDS = {
    "candidate_correctness": 300.0,
    "candidate_profile": 180.0,
    "ablation": 180.0,
    "candidate_sanitizer": 120.0,
    "workload_paired": 120.0,
    "decision": 120.0,
}


def _budget_stop_output(args, checkpoint: Mapping) -> None:
    print(
        json.dumps(
            {
                "iter": args.iter,
                "status": "budget_exhausted",
                "stage": checkpoint["stage"],
                "candidate_status": "inconclusive",
                "checkpoint": str(Path(args.run_dir) / "checkpoint.json"),
            },
            indent=2,
        )
    )


def _hard_timeout_seconds(clock: BudgetClock) -> float:
    return clock.execution_seconds_available(now=time.monotonic())


def _persist_hard_timeout(
    checkpoint: Mapping,
    stage: str,
    *,
    store: ArtifactStore,
    clock: BudgetClock,
    candidate_id=None,
) -> dict:
    now = time.monotonic()
    stopped = transition_checkpoint(
        checkpoint,
        stage,
        status="budget_exhausted",
        candidate_id=candidate_id,
        candidate_status="inconclusive",
        budget=_checkpoint_budget(clock, now),
        evidence={
            "status": "inconclusive",
            "reason": "hard execution deadline expired",
        },
        updated_at=now,
    )
    return _persist_checkpoint(
        store, stopped, input_hash=stopped["input_hash"]
    )


def _inconclusive_decision(source: Mapping, *, reason: str) -> dict:
    decision = _strict_json_copy(source, "source decision")
    result = {
        "status": "inconclusive",
        "candidate_status": "inconclusive",
        "budget_exhausted": True,
        "reason": reason,
        "statistics": None,
        "kernel_evidence": decision.get("kernel_evidence") or {
            "status": decision.get("status"),
            "statistics": decision.get("statistics"),
        },
    }
    for field in ("candidate_file", "candidate_sha256"):
        if decision.get(field) is not None:
            result[field] = decision[field]
    workload_evidence = decision.get("workload_evidence")
    if workload_evidence is None and decision.get("workload_statistics") is not None:
        workload_evidence = {
            "status": decision.get("status"),
            "statistics": decision.get("workload_statistics"),
        }
    if workload_evidence is not None:
        result["workload_evidence"] = workload_evidence
    return _strict_json_copy(result, "inconclusive decision")


def _persist_decision_budget_exhausted(
    checkpoint: Mapping,
    source_decision: Mapping,
    *,
    iteration_dir: Path,
    store: ArtifactStore,
    clock: BudgetClock,
    reason: str,
) -> tuple[dict, dict]:
    decision = _inconclusive_decision(source_decision, reason=reason)
    decision_path = iteration_dir / "decision.json"
    atomic_write_json(decision_path, decision)
    now = time.monotonic()
    stopped = transition_checkpoint(
        checkpoint,
        "decision",
        status="budget_exhausted",
        candidate_id=checkpoint.get("candidate_id"),
        candidate_status="inconclusive",
        budget=_checkpoint_budget(clock, now),
        evidence={
            "status": "inconclusive",
            "reason": reason,
            "decision_path": str(decision_path.resolve(strict=True)),
            "decision_sha256": sha256_file(decision_path),
        },
        updated_at=now,
    )
    persisted = _persist_checkpoint(
        store, stopped, input_hash=stopped["input_hash"]
    )
    return decision, persisted


def _candidate_checkpoint_id(candidate: Mapping | None) -> str | None:
    if not isinstance(candidate, Mapping):
        return None
    for key in ("branch_index", "id"):
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    path = candidate.get("candidate_file") or candidate.get("kernel")
    if isinstance(path, str) and path.strip():
        candidate_path = Path(path).expanduser()
        if candidate_path.is_file() and not candidate_path.is_symlink():
            return sha256_file(candidate_path)[:16]
    return None


def _safe_iteration_file(iter_dir: Path, name: str) -> Path:
    path = iter_dir / name
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError(f"{name} is missing from the current iteration") from error
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{name} must be a regular file")
    if path.resolve(strict=True).parent != iter_dir:
        raise ValueError(f"{name} escapes the current iteration")
    return path


def _load_iteration_json(iter_dir: Path, name: str) -> dict:
    path = _safe_iteration_file(iter_dir, name)
    try:
        payload = _read(str(path))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is malformed: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must contain a JSON object")
    return _strict_json_copy(payload, name)


def _load_lifecycle_branch_results(iter_dir: Path) -> dict:
    payload = _load_iteration_json(iter_dir, "branch_results.json")
    status = payload.get("status")
    if status not in _BRANCH_RESULT_STATUSES:
        raise ValueError(f"branch_results.json status is invalid: {status!r}")
    return payload


def _select_lifecycle_candidate(
    branch_payload: Mapping, *, mode: str, policy: BudgetPolicy
) -> tuple[dict, list[dict]]:
    champion = branch_payload.get("champion")
    if mode == "full":
        candidates = select_outer_candidates(
            branch_payload.get("shortlist", []), policy.outer_candidates
        )
        if not candidates:
            raise ValueError("full mode shortlist contains no confirmed kernel win")
        return candidates[0], candidates
    if mode == "kernel-only":
        if not isinstance(champion, Mapping):
            raise ValueError("branch_results champion must be a mapping")
        selected = dict(champion)
        selected["candidate_file"] = branch_payload.get("selected_kernel")
        return selected, [selected]
    raise ValueError("state mode must be full or kernel-only")


def _validate_sanitizer_report(
    report,
    *,
    candidate: Mapping,
    state: Mapping,
    mode: str | None = None,
    expected_method_ids: Sequence[str] | None = None,
    expected_tools: Sequence[str] | None = None,
    expected_command: Sequence[str] | None = None,
) -> dict:
    if not isinstance(report, Mapping):
        raise ValueError("sanitizer report must contain a JSON object")
    clean = _strict_json_copy(report, "sanitizer report")
    status = clean.get("status")
    if status not in {
        "passed", "failed", "unavailable", "not_applicable", "timed_out"
    }:
        raise ValueError("sanitizer report status is invalid")
    expected_passed = status in {"passed", "not_applicable"}
    if clean.get("passed") is not expected_passed:
        raise ValueError("sanitizer report passed conflicts with status")
    expected_coverage = {
        "passed": "complete",
        "failed": clean.get("coverage"),
        "unavailable": "degraded",
        "not_applicable": "not_applicable",
        "timed_out": "incomplete",
    }[status]
    if status == "failed":
        if clean.get("coverage") not in {"complete", "degraded"}:
            raise ValueError("failed sanitizer report coverage is invalid")
    elif clean.get("coverage") != expected_coverage:
        raise ValueError("sanitizer report coverage conflicts with status")
    candidate_snapshot = _candidate_snapshot(candidate)
    candidate_file = candidate_snapshot["path"]
    if clean.get("candidate_file") != candidate_file:
        raise ValueError("sanitizer report candidate_file drifted")
    if clean.get("candidate_sha256") != candidate_snapshot["sha256"]:
        raise ValueError("sanitizer report candidate_sha256 drifted")
    if clean.get("input_hash") != state.get("input_hash"):
        raise ValueError("sanitizer report input_hash drifted")
    if mode is not None and clean.get("mode") != mode:
        raise ValueError("sanitizer report mode drifted")
    if expected_method_ids is not None and clean.get("method_ids") != list(
        expected_method_ids
    ):
        raise ValueError("sanitizer report method_ids drifted")
    if expected_tools is not None and clean.get("selected_tools") != list(
        expected_tools
    ):
        raise ValueError("sanitizer report selected_tools drifted")
    report_tools = clean.get("selected_tools")
    if not isinstance(report_tools, list):
        raise ValueError("sanitizer report selected_tools is malformed")
    if expected_command is None and report_tools:
        first_command = clean.get("results", [{}])[0].get("command")
        if not isinstance(first_command, list) or len(first_command) < 5:
            raise ValueError("sanitizer report command is malformed")
        report_command = first_command[5:]
    else:
        report_command = expected_command
    sanitizer_engine.validate_result(
        clean,
        selected_tools=report_tools,
        command=report_command,
    )
    return clean


def _aggregate_sanitizer_results(
    *, state: Mapping, mode: str, candidates, reports
) -> dict:
    if mode not in {"targeted", "full"}:
        raise ValueError("sanitizer mode must be targeted or full")
    candidate_list = list(candidates)
    report_list = list(reports)
    if not candidate_list or len(candidate_list) != len(report_list):
        raise ValueError("sanitizer candidates and reports must be non-empty and aligned")
    records = []
    for candidate, report in zip(candidate_list, report_list):
        clean = _validate_sanitizer_report(
            report, candidate=candidate, state=state
        )
        record = {
            "candidate_id": _candidate_checkpoint_id(candidate),
            "candidate_file": clean["candidate_file"],
            "candidate_sha256": clean["candidate_sha256"],
            "status": clean["status"],
            "candidate_status": (
                "rejected_correctness" if clean["status"] == "failed" else "eligible"
            ),
            "passed": clean["passed"],
            "coverage": clean["coverage"],
        }
        if clean.get("artifact") is not None:
            artifact = Path(clean["artifact"]).expanduser()
            if artifact.is_symlink() or not artifact.is_file():
                raise ValueError("sanitizer raw artifact must be a regular file")
            resolved_artifact = artifact.resolve(strict=True)
            record["artifact"] = str(resolved_artifact)
            record["artifact_sha256"] = sha256_file(resolved_artifact)
        records.append(record)

    statuses = {record["status"] for record in records}
    failed_count = sum(record["status"] == "failed" for record in records)
    eligible_count = len(records) - failed_count
    if failed_count and eligible_count:
        status = "partial_rejection"
    elif failed_count:
        status = "rejected_correctness"
    elif "unavailable" in statuses:
        status = "unavailable"
    elif statuses == {"not_applicable"}:
        status = "not_applicable"
    else:
        status = "passed"
    if any(record["coverage"] == "degraded" for record in records):
        coverage = "degraded"
    elif all(record["coverage"] == "not_applicable" for record in records):
        coverage = "not_applicable"
    else:
        coverage = "complete"
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "input_hash": state.get("input_hash"),
        "mode": mode,
        "status": status,
        "passed": status in {"passed", "not_applicable", "partial_rejection"},
        "coverage": coverage,
        "candidates": records,
    }


def _sanitizer_candidate_outcomes(
    aggregate: Mapping, candidates, *, mode: str
) -> tuple[list[dict], list[dict]]:
    if mode not in {"kernel-only", "full"}:
        raise ValueError("state mode must be full or kernel-only")
    records = aggregate.get("candidates")
    if not isinstance(records, list):
        raise ValueError("sanitizer aggregate candidates are malformed")
    by_identity = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("sanitizer aggregate candidate is malformed")
        identity = (record.get("candidate_file"), record.get("candidate_sha256"))
        if not all(isinstance(value, str) for value in identity) or identity in by_identity:
            raise ValueError("sanitizer aggregate candidate identity is invalid")
        by_identity[identity] = record
    eligible = []
    rejected = []
    for candidate in candidates:
        candidate_snapshot = _candidate_snapshot(candidate)
        candidate_digest = candidate_snapshot["sha256"]
        candidate_file = candidate_snapshot["path"]
        record = by_identity.get((candidate_file, candidate_digest))
        if record is None:
            raise ValueError("sanitizer aggregate is missing a candidate")
        if record.get("status") == "failed":
            failed_candidate = dict(candidate)
            failed_candidate["status"] = "rejected_correctness"
            rejected.append(
                build_terminal_decision(
                    mode=mode,
                    candidate=failed_candidate,
                    _snapshot=candidate_snapshot,
                )
            )
        elif record.get("status") in {
            "passed",
            "unavailable",
            "not_applicable",
        }:
            eligible.append(dict(candidate))
        else:
            raise ValueError("sanitizer aggregate candidate status is invalid")
    return eligible, rejected


def _run_sanitizer_gate(
    *,
    state: Mapping,
    policy: BudgetPolicy,
    candidates,
    iter_dir,
    methods_json,
    benchmark,
    hard_timeout,
) -> dict:
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("sanitizer gate requires at least one candidate")
    iteration_dir = Path(iter_dir).expanduser().resolve(strict=True)
    methods_source = Path(methods_json).expanduser()
    if methods_source.is_symlink():
        raise ValueError("methods.json must be a non-symlink regular file")
    methods_path = methods_source.resolve(strict=True)
    if not methods_path.is_file():
        raise ValueError("methods.json must be a non-symlink regular file")
    methods_payload = _read(str(methods_path))
    if not isinstance(methods_payload, Mapping) or not isinstance(
        methods_payload.get("methods"), list
    ):
        raise ValueError("methods.json must contain a top-level methods list")
    method_ids = []
    for index, method in enumerate(methods_payload["methods"]):
        if not isinstance(method, Mapping):
            raise ValueError(f"methods[{index}] must be a mapping")
        method_id = method.get("id")
        if type(method_id) is not str or not method_id.strip():
            raise ValueError(f"methods[{index}].id must be a non-empty string")
        method_ids.append(method_id.strip())
    methods_sha256 = sha256_file(methods_path)
    sanitizer_policy_path = SCRIPT_DIR.parent / "references" / "sanitizer_policy.json"
    sanitizer_policy = sanitizer_engine.load_policy(sanitizer_policy_path)
    policy_sha256 = sha256_file(sanitizer_policy_path)
    selected_tools = sanitizer_engine.select_tools(
        method_ids, mode=policy.sanitizer_mode, policy=sanitizer_policy
    )
    sanitizer_dir = iteration_dir / "sanitizer"
    if sanitizer_dir.is_symlink():
        raise ValueError("sanitizer artifact directory must not be a symlink")
    sanitizer_dir.mkdir(mode=0o700, exist_ok=True)
    if sanitizer_dir.resolve(strict=True).parent != iteration_dir:
        raise ValueError("sanitizer artifact directory escapes the iteration")

    dims = state.get("dims", {})
    if not isinstance(dims, Mapping):
        raise ValueError("state dims must be a mapping")
    ptr_size = state.get("ptr_size", 0)
    if isinstance(ptr_size, bool) or not isinstance(ptr_size, int) or ptr_size < 0:
        raise ValueError("state ptr_size must be a non-negative integer")
    timeout_provider = hard_timeout if callable(hard_timeout) else lambda: hard_timeout
    reports = []
    for candidate in candidate_list:
        candidate_snapshot = _candidate_snapshot(candidate)
        candidate_file = candidate_snapshot["path"]
        try:
            Path(candidate_file).relative_to(iteration_dir)
        except ValueError as error:
            raise ValueError("sanitizer candidate escapes the iteration") from error
        digest = candidate_snapshot["sha256"]
        benchmark_command = [
            sys.executable,
            str(benchmark),
            candidate_file,
            "--profile-only",
            "--warmup",
            "0",
            "--repeat",
            "1",
        ]
        if ptr_size:
            benchmark_command.extend(["--ptr-size", str(ptr_size)])
        benchmark_command.extend(
            f"--{key}={value}" for key, value in sorted(dims.items())
        )
        identity_digest = hashlib.sha256(
            f"{candidate_file}\0{digest}".encode("utf-8")
        ).hexdigest()
        output = sanitizer_dir / f"{identity_digest}.json"
        unbound_output = sanitizer_dir / f"{identity_digest}.unbound.json"
        if output.is_symlink():
            raise ValueError("sanitizer candidate artifact must not be a symlink")
        if output.is_file():
            try:
                report = _read(str(output))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"sanitizer candidate artifact is malformed: {error}") from error
            report["artifact"] = str(output.resolve(strict=True))
            validated = _validate_sanitizer_report(
                report,
                candidate=candidate,
                state=state,
                mode=policy.sanitizer_mode,
                expected_method_ids=method_ids,
                expected_tools=selected_tools,
                expected_command=benchmark_command,
            )
            has_methods_binding = "methods_sha256" in report
            has_policy_binding = "policy_sha256" in report
            if not has_methods_binding or not has_policy_binding:
                raise ValueError(
                    "sanitizer candidate artifact parent binding is missing or incomplete"
                )
            if report.get("methods_sha256") != methods_sha256:
                raise ValueError("sanitizer candidate artifact methods_sha256 drifted")
            if report.get("policy_sha256") != policy_sha256:
                raise ValueError("sanitizer candidate artifact policy_sha256 drifted")
            if validated["status"] == "timed_out":
                raise ValueError("final sanitizer artifact must not be timed_out")
            if unbound_output.is_symlink():
                raise ValueError("sanitizer unbound artifact must not be a symlink")
            if unbound_output.is_file():
                unbound_output.unlink()
            elif unbound_output.exists():
                raise ValueError("sanitizer unbound artifact must be a regular file")
            reports.append(validated)
            continue
        if output.exists():
            raise ValueError("sanitizer candidate artifact must be a regular file")

        if unbound_output.is_symlink():
            raise ValueError("sanitizer unbound artifact must not be a symlink")
        if unbound_output.is_file():
            try:
                unbound_report = _read(str(unbound_output))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"sanitizer unbound artifact is malformed: {error}"
                ) from error
            if (
                "methods_sha256" in unbound_report
                or "policy_sha256" in unbound_report
            ):
                raise ValueError(
                    "sanitizer unbound artifact must not contain parent binding"
                )
            _validate_sanitizer_report(
                unbound_report,
                candidate=candidate,
                state=state,
                mode=policy.sanitizer_mode,
                expected_method_ids=method_ids,
                expected_tools=selected_tools,
                expected_command=benchmark_command,
            )
            unbound_output.unlink()
        elif unbound_output.exists():
            raise ValueError("sanitizer unbound artifact must be a regular file")

        if not selected_tools:
            report = sanitizer_engine.run_tools(
                executable=None,
                tools=[],
                command=benchmark_command,
            )
            report.update(
                {
                    "mode": policy.sanitizer_mode,
                    "method_ids": method_ids,
                    "selected_tools": [],
                    "methods_sha256": methods_sha256,
                    "policy_sha256": policy_sha256,
                }
            )
            report = sanitizer_engine.bind_candidate(
                report,
                candidate_file=candidate_file,
                input_hash=state["input_hash"],
            )
            if sha256_file(candidate_file) != digest:
                raise ValueError("sanitizer candidate changed during execution")
            atomic_write_json(output, report)
            report["artifact"] = str(output.resolve(strict=True))
            reports.append(report)
            continue
        command = [
            sys.executable,
            str(SCRIPT_DIR / "sanitize.py"),
            "--mode",
            policy.sanitizer_mode,
            "--policy",
            str(SCRIPT_DIR.parent / "references" / "sanitizer_policy.json"),
            "--methods-json",
            str(methods_path),
            "--candidate-file",
            candidate_file,
            "--input-hash",
            state["input_hash"],
            "--out",
            str(unbound_output),
            "--",
            *benchmark_command,
        ]
        completed = _run(
            command,
            capture_output=True,
            hard_timeout=_finite_real(
                timeout_provider(), "sanitizer hard_timeout", minimum=0.0
            ),
        )
        if getattr(completed, "timed_out", False):
            return {
                "timed_out": True,
                "candidate_id": _candidate_checkpoint_id(candidate),
            }
        if completed.returncode not in {
            0, sanitizer_engine.ERROR_EXITCODE, 124
        }:
            diagnostic = (completed.stderr or completed.stdout or "").strip()
            raise SystemExit(
                f"sanitize failed rc={completed.returncode}"
                + (f": {diagnostic}" if diagnostic else "")
            )
        if not unbound_output.is_file() or unbound_output.is_symlink():
            raise ValueError("sanitize did not write a safe unbound artifact")
        try:
            report = _read(str(unbound_output))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"sanitizer unbound artifact is malformed: {error}") from error
        if "methods_sha256" in report or "policy_sha256" in report:
            raise ValueError(
                "sanitizer unbound artifact must not contain parent binding"
            )
        if sha256_file(candidate_file) != digest:
            raise ValueError("sanitizer candidate changed during execution")
        if sha256_file(methods_path) != methods_sha256:
            raise ValueError("methods.json changed during sanitizer execution")
        if sha256_file(sanitizer_policy_path) != policy_sha256:
            raise ValueError("sanitizer policy changed during execution")
        validated = _validate_sanitizer_report(
            report,
            candidate=candidate,
            state=state,
            mode=policy.sanitizer_mode,
            expected_method_ids=method_ids,
            expected_tools=selected_tools,
            expected_command=benchmark_command,
        )
        if completed.returncode == 124:
            if validated["status"] != "timed_out":
                raise ValueError("sanitize rc=124 requires a timed_out report")
            return {
                "timed_out": True,
                "candidate_id": _candidate_checkpoint_id(candidate),
            }
        if validated["status"] == "timed_out":
            raise ValueError("timed_out sanitizer report requires rc=124")
        bound_report = dict(validated)
        bound_report.update(
            {
                "methods_sha256": methods_sha256,
                "policy_sha256": policy_sha256,
            }
        )
        atomic_write_json(output, bound_report)
        unbound_output.unlink()
        bound_report["artifact"] = str(output.resolve(strict=True))
        reports.append(bound_report)

    aggregate = _aggregate_sanitizer_results(
        state=state,
        mode=policy.sanitizer_mode,
        candidates=candidate_list,
        reports=reports,
    )
    aggregate.update(
        {
            "methods_sha256": methods_sha256,
            "policy_sha256": policy_sha256,
            "method_ids": method_ids,
            "selected_tools": selected_tools,
        }
    )
    atomic_write_json(iteration_dir / "sanitizer.json", aggregate)
    return aggregate


def _load_sanitizer_aggregate(
    iter_dir: Path,
    *,
    state: Mapping,
    policy: BudgetPolicy,
    methods_json,
    candidates,
) -> dict:
    aggregate = _load_iteration_json(iter_dir, "sanitizer.json")
    if aggregate.get("schema_version") != CURRENT_SCHEMA_VERSION:
        raise ValueError("sanitizer.json schema_version is invalid")
    if aggregate.get("input_hash") != state.get("input_hash"):
        raise ValueError("sanitizer.json input_hash drifted")
    if aggregate.get("mode") != policy.sanitizer_mode:
        raise ValueError("sanitizer.json mode drifted")
    methods_path = Path(methods_json).expanduser()
    if methods_path.is_symlink() or not methods_path.is_file():
        raise ValueError("methods.json must be a non-symlink regular file")
    methods_path = methods_path.resolve(strict=True)
    methods_payload = _read(str(methods_path))
    if not isinstance(methods_payload, Mapping) or not isinstance(
        methods_payload.get("methods"), list
    ):
        raise ValueError("methods.json must contain a top-level methods list")
    method_ids = []
    for index, method in enumerate(methods_payload["methods"]):
        if not isinstance(method, Mapping):
            raise ValueError(f"methods[{index}] must be a mapping")
        method_id = method.get("id")
        if type(method_id) is not str or not method_id.strip():
            raise ValueError(f"methods[{index}].id must be a non-empty string")
        method_ids.append(method_id.strip())
    policy_path = SCRIPT_DIR.parent / "references" / "sanitizer_policy.json"
    sanitizer_policy = sanitizer_engine.load_policy(policy_path)
    selected_tools = sanitizer_engine.select_tools(
        method_ids, mode=policy.sanitizer_mode, policy=sanitizer_policy
    )
    if aggregate.get("methods_sha256") != sha256_file(methods_path):
        raise ValueError("sanitizer.json methods_sha256 drifted")
    if aggregate.get("policy_sha256") != sha256_file(policy_path):
        raise ValueError("sanitizer.json policy_sha256 drifted")
    if aggregate.get("method_ids") != method_ids:
        raise ValueError("sanitizer.json method_ids drifted")
    if aggregate.get("selected_tools") != selected_tools:
        raise ValueError("sanitizer.json selected_tools drifted")
    if aggregate.get("coverage") not in {
        "complete",
        "degraded",
        "not_applicable",
    }:
        raise ValueError("sanitizer.json coverage is invalid")
    records = aggregate.get("candidates")
    candidate_list = list(candidates)
    if not isinstance(records, list) or len(records) != len(candidate_list):
        raise ValueError("sanitizer.json candidates drifted")
    for record, candidate in zip(records, candidate_list):
        if not isinstance(record, Mapping):
            raise ValueError("sanitizer.json candidate is malformed")
        candidate_snapshot = _candidate_snapshot(candidate)
        candidate_file = candidate_snapshot["path"]
        candidate_sha256 = candidate_snapshot["sha256"]
        status = record.get("status")
        artifact_value = record.get("artifact")
        artifact_digest = record.get("artifact_sha256")
        if not isinstance(artifact_value, str) or not isinstance(
            artifact_digest, str
        ):
            raise ValueError("sanitizer.json artifact binding is missing")
        artifact_path = Path(artifact_value).expanduser()
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise ValueError("sanitizer.json raw artifact is missing or unsafe")
        resolved_artifact = artifact_path.resolve(strict=True)
        sanitizer_dir = (iter_dir / "sanitizer").resolve(strict=True)
        if (
            resolved_artifact.parent != sanitizer_dir
            or str(resolved_artifact) != artifact_value
            or sha256_file(resolved_artifact) != artifact_digest
        ):
            raise ValueError("sanitizer.json raw artifact binding drifted")
        try:
            raw_report = _read(str(resolved_artifact))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"sanitizer raw artifact is malformed: {error}"
            ) from error
        if raw_report.get("methods_sha256") != aggregate.get("methods_sha256"):
            raise ValueError("sanitizer raw artifact methods_sha256 drifted")
        if raw_report.get("policy_sha256") != aggregate.get("policy_sha256"):
            raise ValueError("sanitizer raw artifact policy_sha256 drifted")
        raw_report["artifact"] = artifact_value
        validated_report = _validate_sanitizer_report(
            raw_report,
            candidate=candidate,
            state=state,
            mode=policy.sanitizer_mode,
            expected_method_ids=method_ids,
            expected_tools=selected_tools,
        )
        if (
            record.get("candidate_id") != _candidate_checkpoint_id(candidate)
            or record.get("candidate_file") != candidate_file
            or record.get("candidate_sha256") != candidate_sha256
            or status not in {"passed", "failed", "unavailable", "not_applicable"}
            or record.get("candidate_status")
            != ("rejected_correctness" if status == "failed" else "eligible")
            or record.get("passed") != (status in {"passed", "not_applicable"})
            or status != validated_report.get("status")
            or record.get("coverage") != validated_report.get("coverage")
            or record.get("passed") != validated_report.get("passed")
        ):
            raise ValueError("sanitizer.json candidates drifted")
    statuses = {record["status"] for record in records}
    failed_count = sum(record["status"] == "failed" for record in records)
    if failed_count and failed_count < len(records):
        derived_status = "partial_rejection"
    elif failed_count:
        derived_status = "rejected_correctness"
    elif "unavailable" in statuses:
        derived_status = "unavailable"
    elif statuses == {"not_applicable"}:
        derived_status = "not_applicable"
    else:
        derived_status = "passed"
    if aggregate.get("status") != derived_status or aggregate.get("passed") != (
        derived_status in {"passed", "not_applicable", "partial_rejection"}
    ):
        raise ValueError("sanitizer.json derived status drifted")
    if any(record.get("coverage") == "degraded" for record in records):
        derived_coverage = "degraded"
    elif all(record.get("coverage") == "not_applicable" for record in records):
        derived_coverage = "not_applicable"
    else:
        derived_coverage = "complete"
    if aggregate.get("coverage") != derived_coverage:
        raise ValueError("sanitizer.json derived coverage drifted")
    return aggregate


def _write_selection_artifact(
    iter_dir: Path,
    *,
    state: Mapping,
    candidate: Mapping,
    kernel: str,
    candidate_id: str | None,
) -> dict:
    normalized = _strict_json_copy(candidate, "selected candidate")
    normalized["candidate_file"] = kernel
    payload = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "input_hash": state["input_hash"],
        "mode": state.get("mode", "kernel-only"),
        "candidate_id": candidate_id,
        "candidate_file": kernel,
        "candidate_sha256": sha256_file(kernel),
        "candidate": normalized,
    }
    atomic_write_json(iter_dir / "selected_candidate.json", payload)
    return payload


def _load_selection_artifact(iter_dir: Path, *, state: Mapping) -> dict:
    payload = _load_iteration_json(iter_dir, "selected_candidate.json")
    if payload.get("schema_version") != CURRENT_SCHEMA_VERSION:
        raise ValueError("selected_candidate.json schema_version is invalid")
    if payload.get("input_hash") != state.get("input_hash"):
        raise ValueError("selected_candidate.json input_hash drifted")
    if payload.get("mode") != state.get("mode", "kernel-only"):
        raise ValueError("selected_candidate.json mode drifted")
    candidate_id = payload.get("candidate_id")
    if candidate_id is not None and (
        type(candidate_id) is not str or not candidate_id.strip()
    ):
        raise ValueError("selected_candidate.json candidate_id is invalid")
    candidate = payload.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("selected_candidate.json candidate is malformed")
    candidate_file = payload.get("candidate_file")
    digest = payload.get("candidate_sha256")
    if not isinstance(candidate_file, str) or not isinstance(digest, str):
        raise ValueError("selected_candidate.json candidate binding is malformed")
    kernel = Path(candidate_file).expanduser()
    if kernel.is_symlink() or not kernel.is_file():
        raise ValueError("selected_candidate.json candidate file drifted")
    resolved = kernel.resolve(strict=True)
    if resolved.parent != iter_dir or resolved.name not in {"kernel.py", "kernel.cu"}:
        raise ValueError("selected_candidate.json candidate escapes the iteration")
    if sha256_file(resolved) != digest.lower():
        raise ValueError("selected_candidate.json candidate sha256 drifted")
    if candidate.get("candidate_file") != str(resolved):
        raise ValueError("selected_candidate.json candidate path conflicts")
    payload["candidate_file"] = str(resolved)
    return payload


_CANDIDATE_PROFILE_ARTIFACTS = (
    "kernel.ncu-rep",
    "kernel.ncu.log",
    "ncu_top.json",
    "candidate_profile_binding.json",
)


def _invalidate_candidate_profile(
    iter_dir: Path,
    *,
    state: Mapping,
    old_candidate_id: str | None,
    new_candidate_id: str | None,
) -> None:
    removed = []
    for name in _CANDIDATE_PROFILE_ARTIFACTS:
        path = iter_dir / name
        if path.is_symlink():
            raise ValueError(f"{name} must not be a symlink")
        if path.is_file():
            path.unlink()
            removed.append(name)
        elif path.exists():
            raise ValueError(f"{name} must be a regular file")
    atomic_write_json(
        iter_dir / "candidate_profile_invalidation.json",
        {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "input_hash": state["input_hash"],
            "old_candidate_id": old_candidate_id,
            "new_candidate_id": new_candidate_id,
            "removed_artifacts": removed,
        },
    )


def _write_candidate_profile_binding(
    iter_dir: Path,
    *,
    state: Mapping,
    candidate_id: str | None,
    kernel: str,
    returncode: int,
) -> dict:
    kernel_path = Path(kernel).expanduser()
    if kernel_path.is_symlink() or not kernel_path.is_file():
        raise ValueError("profiled candidate must be a non-symlink regular file")
    resolved = kernel_path.resolve(strict=True)
    if resolved.parent != iter_dir or resolved.name not in {"kernel.py", "kernel.cu"}:
        raise ValueError("profiled candidate escapes the iteration")
    payload = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "input_hash": state["input_hash"],
        "candidate_id": candidate_id,
        "candidate_file": str(resolved),
        "candidate_sha256": sha256_file(resolved),
        "returncode": returncode,
        "ncu_top": None,
        "ncu_top_sha256": None,
    }
    top_path = iter_dir / "ncu_top.json"
    if top_path.is_symlink():
        raise ValueError("ncu_top.json must not be a symlink")
    if not top_path.is_file():
        raise ValueError("profile_ncu did not write ncu_top.json")
    top = _load_iteration_json(iter_dir, "ncu_top.json")
    profiled_file = top.get("profiled_file")
    if not isinstance(profiled_file, str):
        raise ValueError("ncu_top.json profiled_file is missing")
    profiled_path = Path(profiled_file).expanduser()
    if profiled_path.is_symlink() or profiled_path.resolve(strict=True) != resolved:
        raise ValueError("ncu_top.json is bound to a different candidate")
    payload["ncu_top"] = str(top_path.resolve(strict=True))
    payload["ncu_top_sha256"] = sha256_file(top_path)
    atomic_write_json(iter_dir / "candidate_profile_binding.json", payload)
    return payload


def _profile_binding_matches(
    iter_dir: Path,
    *,
    state: Mapping,
    candidate_id: str | None,
    kernel: str,
) -> bool:
    path = iter_dir / "candidate_profile_binding.json"
    if path.is_symlink() or not path.is_file():
        return False
    try:
        binding = _load_iteration_json(iter_dir, "candidate_profile_binding.json")
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    kernel_path = Path(kernel).expanduser()
    if kernel_path.is_symlink() or not kernel_path.is_file():
        return False
    resolved_path = kernel_path.resolve(strict=True)
    resolved = str(resolved_path)
    if not (
        binding.get("schema_version") == CURRENT_SCHEMA_VERSION
        and binding.get("input_hash") == state.get("input_hash")
        and binding.get("candidate_id") == candidate_id
        and binding.get("candidate_file") == resolved
        and binding.get("candidate_sha256") == sha256_file(resolved_path)
    ):
        return False
    top_value = binding.get("ncu_top")
    top_digest = binding.get("ncu_top_sha256")
    if not isinstance(top_value, str) or not isinstance(top_digest, str):
        return False
    top_path = Path(top_value).expanduser()
    if top_path.is_symlink() or not top_path.is_file():
        return False
    try:
        resolved_top = top_path.resolve(strict=True)
    except OSError:
        return False
    if (
        resolved_top.parent != iter_dir
        or resolved_top.name != "ncu_top.json"
        or str(resolved_top) != top_value
        or sha256_file(resolved_top) != top_digest
    ):
        return False
    try:
        top = _load_iteration_json(iter_dir, "ncu_top.json")
        profiled_path = Path(top.get("profiled_file", "")).expanduser()
        if profiled_path.is_symlink() or profiled_path.resolve(strict=True) != resolved_path:
            return False
    except (OSError, ValueError, TypeError):
        return False
    return True


def _run_candidate_profile(
    *,
    state_path: str,
    iteration: int,
    iter_dir: Path,
    state: Mapping,
    benchmark: str,
    candidate_id: str | None,
    kernel: str,
    hard_timeout: float,
):
    result = _run(
        [
            sys.executable,
            str(SCRIPT_DIR / "profile_ncu.py"),
            "--state",
            state_path,
            "--iter",
            str(iteration),
            "--which",
            "kernel",
            "--benchmark",
            os.path.abspath(benchmark),
            "--promote-if-best",
        ],
        hard_timeout=hard_timeout,
    )
    if not getattr(result, "timed_out", False):
        _write_candidate_profile_binding(
            iter_dir,
            state=state,
            candidate_id=candidate_id,
            kernel=kernel,
            returncode=result.returncode,
        )
    return result


def _write_workload_result_artifact(
    iter_dir: Path,
    *,
    state: Mapping,
    terminal_decision: Mapping,
    candidate_id: str | None,
) -> dict:
    decision = _strict_json_copy(terminal_decision, "terminal decision")
    payload = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "input_hash": state["input_hash"],
        "mode": state.get("mode", "kernel-only"),
        "candidate_id": candidate_id,
        "candidate_file": decision.get("candidate_file"),
        "candidate_sha256": decision.get("candidate_sha256"),
        "decision": decision,
    }
    atomic_write_json(iter_dir / "workload_result.json", payload)
    return payload


def _load_workload_result_artifact(iter_dir: Path, *, state: Mapping) -> dict:
    payload = _load_iteration_json(iter_dir, "workload_result.json")
    if payload.get("schema_version") != CURRENT_SCHEMA_VERSION:
        raise ValueError("workload_result.json schema_version is invalid")
    if payload.get("input_hash") != state.get("input_hash"):
        raise ValueError("workload_result.json input_hash drifted")
    if payload.get("mode") != state.get("mode", "kernel-only"):
        raise ValueError("workload_result.json mode drifted")
    decision = payload.get("decision")
    if not isinstance(decision, Mapping):
        raise ValueError("workload_result.json decision is malformed")
    candidate_file = payload.get("candidate_file")
    digest = payload.get("candidate_sha256")
    if (
        decision.get("candidate_file") != candidate_file
        or decision.get("candidate_sha256") != digest
    ):
        raise ValueError("workload_result.json candidate binding conflicts")
    if not isinstance(candidate_file, str) or not isinstance(digest, str):
        raise ValueError("workload_result.json candidate binding is malformed")
    kernel = Path(candidate_file).expanduser()
    if kernel.is_symlink() or not kernel.is_file():
        raise ValueError("workload_result.json candidate file drifted")
    resolved = kernel.resolve(strict=True)
    if resolved.parent != iter_dir or resolved.name not in {"kernel.py", "kernel.cu"}:
        raise ValueError("workload_result.json candidate escapes the iteration")
    if sha256_file(resolved) != digest.lower():
        raise ValueError("workload_result.json candidate sha256 drifted")
    payload["candidate_file"] = str(resolved)
    payload["decision"] = _strict_json_copy(decision, "terminal decision")
    return payload


def _decision_already_applied(state: Mapping, iteration: int, payload: Mapping) -> bool:
    binding = state.get("candidates", {}).get(f"iter-{iteration}")
    if not isinstance(binding, Mapping):
        return False
    if (
        binding.get("candidate_file") != payload.get("candidate_file")
        or binding.get("candidate_sha256") != payload.get("candidate_sha256")
        or binding.get("status") != payload.get("status")
    ):
        return False
    expected_decision = str(
        (Path(state["run_dir"]) / f"iterv{iteration}" / "decision.json").resolve()
    )
    try:
        authoritative = _strict_json_copy(
            _read(expected_decision), "authoritative decision"
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if authoritative != _strict_json_copy(payload, "expected decision"):
        return False
    return any(
        isinstance(item, Mapping)
        and item.get("iter") == iteration
        and item.get("decision_json") == expected_decision
        and item.get("status") == payload.get("status")
        for item in state.get("history", [])
    )


def _decision_record_already_applied(
    state: Mapping, iteration: int, payload: Mapping
) -> bool:
    decision_path = (
        Path(state["run_dir"]) / f"iterv{iteration}" / "decision.json"
    ).resolve()
    try:
        if _strict_json_copy(
            _read(str(decision_path)), "authoritative decision"
        ) != _strict_json_copy(payload, "expected decision"):
            return False
        digest = sha256_file(decision_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return any(
        isinstance(item, Mapping)
        and item.get("event") == "decision_record"
        and item.get("iter") == iteration
        and item.get("status") == payload.get("status")
        and item.get("decision_json") == str(decision_path)
        and item.get("decision_sha256") == digest
        for item in state.get("history", [])
    )


def _restore_state_after_producer_timeout(
    state_path: str, snapshot: Mapping
) -> None:
    state_manager.validate_state(dict(snapshot))
    atomic_write_json(state_path, _strict_json_copy(snapshot, "state snapshot"))


def _closed_lifecycle_output(
    args, state_path: str, state: Mapping, *, next_iter: int | None = None
) -> None:
    print(
        json.dumps(
            {
                "iter": args.iter,
                "status": "closed",
                "best_ms": state.get("best_metric_ms"),
                "next_iter": next_iter,
                "early_stop": False,
                "state": state_path,
            },
            indent=2,
        )
    )


def _cmd_close_iter_lifecycle(args, state, state_path, iter_dir, methods_json):
    run_dir = Path(args.run_dir).expanduser().resolve(strict=True)
    iteration_dir = Path(iter_dir).expanduser().resolve(strict=True)
    if iteration_dir.parent != run_dir:
        raise ValueError("iteration directory escapes the run root")
    manifest, manifest_hash = _load_and_verify_manifest(run_dir)
    if state.get("input_hash") != manifest_hash:
        raise ValueError("manifest/state frozen input_hash mismatch")
    if manifest.get("mode") != state.get("mode", "kernel-only"):
        raise ValueError("manifest/state mode mismatch")
    _verify_state_candidates(state)

    policy = _policy_from_state(state)
    store = ArtifactStore(run_dir)
    checkpoint = _validate_checkpoint(
        store.load_checkpoint(expected_input_hash=state["input_hash"]),
        input_hash=state["input_hash"],
    )
    clock = BudgetClock(
        policy,
        started_at=time.monotonic(),
        elapsed_seconds=_finite_real(
            checkpoint["budget"]["elapsed_seconds"],
            "checkpoint budget.elapsed_seconds",
            minimum=0.0,
        ),
    )
    mode = state.get("mode", "kernel-only")
    attribution_path = str(iteration_dir / "attribution.json")
    sass_check_path = str(iteration_dir / "sass_check.json")

    initial = resume(
        checkpoint,
        input_hash=state["input_hash"],
        max_rounds=policy.max_rounds,
    )
    if initial["next_stage"] != "complete" and initial["next_iteration"] != args.iter:
        if (
            checkpoint["stage"] == "decision"
            and checkpoint["status"] == "stage_complete"
            and checkpoint["iteration"] == args.iter
        ):
            _closed_lifecycle_output(
                args,
                state_path,
                _read(state_path),
                next_iter=initial["next_iteration"],
            )
            return
        raise ValueError(
            "close iteration does not match checkpoint next iteration "
            f"{initial['next_iteration']}"
        )

    while True:
        restored = resume(
            checkpoint,
            input_hash=state["input_hash"],
            max_rounds=policy.max_rounds,
        )
        next_stage = restored["next_stage"]
        if next_stage == "complete":
            _closed_lifecycle_output(args, state_path, _read(state_path))
            return
        if restored["next_iteration"] != args.iter:
            _closed_lifecycle_output(
                args,
                state_path,
                _read(state_path),
                next_iter=restored["next_iteration"],
            )
            return
        if next_stage == "baseline":
            raise ValueError("baseline stage must complete before close-iter")

        if next_stage == "candidate_correctness":
            admitted, checkpoint = _admit_checkpoint_stage(
                checkpoint,
                next_stage,
                store=store,
                clock=clock,
                estimated_seconds=_STAGE_ESTIMATES_SECONDS[next_stage],
                candidate_id=None,
            )
            if not admitted:
                _budget_stop_output(args, checkpoint)
                return
            branch_result = _run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "branch_explore.py"),
                    "--state",
                    state_path,
                    "--iter",
                    str(args.iter),
                    "--benchmark",
                    os.path.abspath(args.benchmark),
                    "--warmup",
                    str(args.warmup),
                    "--repeat",
                    str(args.repeat),
                ],
                capture_output=True,
                hard_timeout=_hard_timeout_seconds(clock),
            )
            sys.stderr.write(branch_result.stderr or "")
            if getattr(branch_result, "timed_out", False):
                checkpoint = _persist_hard_timeout(
                    checkpoint,
                    next_stage,
                    store=store,
                    clock=clock,
                )
                _budget_stop_output(args, checkpoint)
                return
            if branch_result.returncode == 2:
                print(
                    json.dumps(
                        {
                            "iter": args.iter,
                            "status": "all_branches_failed",
                            "guidance": "The agent should fix the kernels and retry close-iter.",
                        },
                        indent=2,
                    )
                )
                raise SystemExit(2)
            if branch_result.returncode != 0:
                raise SystemExit(
                    f"branch_explore failed rc={branch_result.returncode}"
                )
            branch_payload = _load_lifecycle_branch_results(iteration_dir)
            checkpoint = _complete_checkpoint_stage(
                checkpoint,
                next_stage,
                {"status": "passed"},
                store=store,
                clock=clock,
                candidate_id=_candidate_checkpoint_id(
                    branch_payload.get("champion")
                ),
            )
            continue

        if next_stage == "candidate_paired":
            branch_payload = _load_lifecycle_branch_results(iteration_dir)
            checkpoint = _complete_checkpoint_stage(
                checkpoint,
                next_stage,
                {
                    "status": (
                        "completed"
                        if branch_payload["status"] == "no_confirmed_kernel_win"
                        else "passed"
                    ),
                    "completed_comparisons": branch_payload.get(
                        "completed_comparisons", 0
                    ),
                },
                store=store,
                clock=clock,
                candidate_id=_candidate_checkpoint_id(
                    branch_payload.get("champion")
                ),
            )
            continue

        if next_stage == "candidate_profile":
            branch_payload = _load_lifecycle_branch_results(iteration_dir)
            if branch_payload["status"] == "no_confirmed_kernel_win":
                checkpoint = _complete_checkpoint_stage(
                    checkpoint,
                    next_stage,
                    {"status": "not_applicable"},
                    store=store,
                    clock=clock,
                )
                continue
            selected, _candidates = _select_lifecycle_candidate(
                branch_payload, mode=mode, policy=policy
            )
            candidate_id = _candidate_checkpoint_id(selected)
            kernel = _publish_outer_candidate(selected, iter_dir=iteration_dir)
            bench = _load_iteration_json(iteration_dir, "bench.json")
            if not bool(bench.get("correctness", {}).get("passed", False)):
                raise SystemExit("champion failed correctness validation")
            _write_selection_artifact(
                iteration_dir,
                state=state,
                candidate=selected,
                kernel=kernel,
                candidate_id=candidate_id,
            )
            admitted, checkpoint = _admit_checkpoint_stage(
                checkpoint,
                next_stage,
                store=store,
                clock=clock,
                estimated_seconds=_STAGE_ESTIMATES_SECONDS[next_stage],
                candidate_id=candidate_id,
            )
            if not admitted:
                _budget_stop_output(args, checkpoint)
                return
            _invalidate_candidate_profile(
                iteration_dir,
                state=state,
                old_candidate_id=None,
                new_candidate_id=candidate_id,
            )
            profile_result = _run_candidate_profile(
                state_path=state_path,
                iteration=args.iter,
                iter_dir=iteration_dir,
                state=state,
                benchmark=args.benchmark,
                candidate_id=candidate_id,
                kernel=kernel,
                hard_timeout=_hard_timeout_seconds(clock),
            )
            if getattr(profile_result, "timed_out", False):
                checkpoint = _persist_hard_timeout(
                    checkpoint,
                    next_stage,
                    store=store,
                    clock=clock,
                    candidate_id=candidate_id,
                )
                _budget_stop_output(args, checkpoint)
                return
            profile_rc = profile_result.returncode
            checkpoint = _complete_checkpoint_stage(
                checkpoint,
                next_stage,
                {
                    "status": "passed" if profile_rc == 0 else "deferred",
                    "returncode": profile_rc,
                },
                store=store,
                clock=clock,
                candidate_id=candidate_id,
            )
            continue

        if next_stage == "candidate_sanitizer":
            branch_payload = _load_lifecycle_branch_results(iteration_dir)
            if branch_payload["status"] == "no_confirmed_kernel_win":
                checkpoint = _complete_checkpoint_stage(
                    checkpoint,
                    next_stage,
                    {
                        "status": "deferred",
                        "reason": "no candidate entered the sanitizer gate",
                    },
                    store=store,
                    clock=clock,
                )
                continue
            selection = _load_selection_artifact(iteration_dir, state=state)
            candidate_id = selection.get("candidate_id")
            _selected, sanitizer_candidates = _select_lifecycle_candidate(
                branch_payload, mode=mode, policy=policy
            )
            admitted, checkpoint = _admit_checkpoint_stage(
                checkpoint,
                next_stage,
                store=store,
                clock=clock,
                estimated_seconds=_STAGE_ESTIMATES_SECONDS[next_stage],
                candidate_id=candidate_id,
            )
            if not admitted:
                _budget_stop_output(args, checkpoint)
                return
            sanitizer_result = _run_sanitizer_gate(
                state=state,
                policy=policy,
                candidates=sanitizer_candidates,
                iter_dir=iteration_dir,
                methods_json=_safe_iteration_file(iteration_dir, "methods.json"),
                benchmark=os.path.abspath(args.benchmark),
                hard_timeout=lambda: _hard_timeout_seconds(clock),
            )
            if sanitizer_result.get("timed_out"):
                checkpoint = _persist_hard_timeout(
                    checkpoint,
                    next_stage,
                    store=store,
                    clock=clock,
                    candidate_id=sanitizer_result.get("candidate_id"),
                )
                _budget_stop_output(args, checkpoint)
                return
            eligible_candidates, rejected_candidates = (
                _sanitizer_candidate_outcomes(
                    sanitizer_result, sanitizer_candidates, mode=mode
                )
            )
            if sanitizer_result["coverage"] == "degraded":
                current_state = _read(state_path)
                current_state["sanitizer_coverage"] = "degraded"
                current_state["sanitizer_coverage_degraded"] = True
                atomic_write_json(state_path, current_state)
                state = current_state

            if eligible_candidates:
                selected_candidate = eligible_candidates[0]
                previous_candidate_id = candidate_id
                candidate_id = _candidate_checkpoint_id(selected_candidate)
                selected_snapshot = _candidate_snapshot(selected_candidate)
                selected_digest = selected_snapshot["sha256"]
                selection_changed = (
                    previous_candidate_id != candidate_id
                    or selection.get("candidate_sha256") != selected_digest
                )
                kernel = _publish_outer_candidate(
                    selected_candidate,
                    iter_dir=iteration_dir,
                    _snapshot=selected_snapshot,
                )
                _write_selection_artifact(
                    iteration_dir,
                    state=state,
                    candidate=selected_candidate,
                    kernel=kernel,
                    candidate_id=candidate_id,
                )
                needs_profile = selection_changed or not _profile_binding_matches(
                    iteration_dir,
                    state=state,
                    candidate_id=candidate_id,
                    kernel=kernel,
                )
                if needs_profile:
                    _invalidate_candidate_profile(
                        iteration_dir,
                        state=state,
                        old_candidate_id=previous_candidate_id,
                        new_candidate_id=candidate_id,
                    )
                    reprofiling_result = _run_candidate_profile(
                        state_path=state_path,
                        iteration=args.iter,
                        iter_dir=iteration_dir,
                        state=state,
                        benchmark=args.benchmark,
                        candidate_id=candidate_id,
                        kernel=kernel,
                        hard_timeout=_hard_timeout_seconds(clock),
                    )
                    if getattr(reprofiling_result, "timed_out", False):
                        checkpoint = _persist_hard_timeout(
                            checkpoint,
                            next_stage,
                            store=store,
                            clock=clock,
                            candidate_id=candidate_id,
                        )
                        _budget_stop_output(args, checkpoint)
                        return
                    reprofile_returncode = reprofiling_result.returncode
                else:
                    reprofile_returncode = None
            else:
                selection_changed = False
                reprofile_returncode = None
            ablation_dir = iteration_dir / "ablations"
            if eligible_candidates and ablation_dir.is_dir():
                if not clock.can_start(
                    now=time.monotonic(),
                    estimated_seconds=_STAGE_ESTIMATES_SECONDS["ablation"],
                ):
                    denied = schedule_next(
                        checkpoint,
                        clock,
                        _STAGE_ESTIMATES_SECONDS["ablation"],
                        now=time.monotonic(),
                        store=store,
                        candidate_id=candidate_id,
                    )
                    _budget_stop_output(args, denied)
                    return
                ablation_result = _run(
                    [
                        sys.executable,
                        str(SCRIPT_DIR / "ablate.py"),
                        "--state",
                        state_path,
                        "--iter",
                        str(args.iter),
                        "--benchmark",
                        os.path.abspath(args.benchmark),
                    ],
                    hard_timeout=_hard_timeout_seconds(clock),
                )
                if getattr(ablation_result, "timed_out", False):
                    checkpoint = _persist_hard_timeout(
                        checkpoint,
                        next_stage,
                        store=store,
                        clock=clock,
                        candidate_id=candidate_id,
                    )
                    _budget_stop_output(args, checkpoint)
                    return
            if eligible_candidates and not clock.can_start(
                now=time.monotonic(),
                estimated_seconds=_STAGE_ESTIMATES_SECONDS[next_stage],
            ):
                denied = schedule_next(
                    checkpoint,
                    clock,
                    _STAGE_ESTIMATES_SECONDS[next_stage],
                    now=time.monotonic(),
                    store=store,
                    candidate_id=candidate_id,
                )
                _budget_stop_output(args, denied)
                return
            if eligible_candidates:
                sass_result = _run(
                    [
                        sys.executable,
                        str(SCRIPT_DIR / "sass_check.py"),
                        "--state",
                        state_path,
                        "--iter",
                        str(args.iter),
                    ],
                    hard_timeout=_hard_timeout_seconds(clock),
                )
                if getattr(sass_result, "timed_out", False):
                    checkpoint = _persist_hard_timeout(
                        checkpoint,
                        next_stage,
                        store=store,
                        clock=clock,
                        candidate_id=candidate_id,
                    )
                    _budget_stop_output(args, checkpoint)
                    return
                sass_status = (
                    "passed" if sass_result.returncode == 0 else "failed"
                )
            else:
                sass_status = "not_applicable"
            checkpoint = _complete_checkpoint_stage(
                checkpoint,
                next_stage,
                {
                    "status": (
                        "deferred"
                        if sanitizer_result["status"] == "not_applicable"
                        else sanitizer_result["status"]
                    ),
                    "coverage": sanitizer_result["coverage"],
                    "artifact": str(iteration_dir / "sanitizer.json"),
                    "eligible_candidates": len(eligible_candidates),
                    "rejected_candidates": len(rejected_candidates),
                    "sass_status": sass_status,
                    "selection_changed": selection_changed,
                    "reprofile_returncode": reprofile_returncode,
                    "reason": (
                        "no selected method matched sanitizer policy"
                        if sanitizer_result["status"] == "not_applicable"
                        else None
                    ),
                },
                store=store,
                clock=clock,
                candidate_id=candidate_id,
                candidate_status=(
                    "rejected_correctness" if not eligible_candidates else None
                ),
            )
            continue

        if next_stage == "workload_paired":
            branch_payload = _load_lifecycle_branch_results(iteration_dir)
            if branch_payload["status"] == "no_confirmed_kernel_win":
                checkpoint = _complete_checkpoint_stage(
                    checkpoint,
                    next_stage,
                    {"status": "not_applicable"},
                    store=store,
                    clock=clock,
                )
                continue
            _preliminary, outer_candidates = _select_lifecycle_candidate(
                branch_payload, mode=mode, policy=policy
            )
            sanitizer_result = _load_sanitizer_aggregate(
                iteration_dir,
                state=state,
                policy=policy,
                methods_json=_safe_iteration_file(iteration_dir, "methods.json"),
                candidates=outer_candidates,
            )
            eligible_candidates, rejected_candidates = (
                _sanitizer_candidate_outcomes(
                    sanitizer_result, outer_candidates, mode=mode
                )
            )
            if eligible_candidates:
                selected_candidate = eligible_candidates[0]
            else:
                selected_candidate = outer_candidates[0]
            candidate_id = _candidate_checkpoint_id(selected_candidate)
            kernel = _publish_outer_candidate(
                selected_candidate, iter_dir=iteration_dir
            )
            if eligible_candidates and not _profile_binding_matches(
                iteration_dir,
                state=state,
                candidate_id=candidate_id,
                kernel=kernel,
            ):
                raise ValueError(
                    "candidate profile evidence is missing, unsafe, or drifted"
                )
            if mode == "full" and eligible_candidates:
                pair_estimate = _finite_real(
                    state.get("estimated_workload_pair_seconds", 0.0),
                    "estimated_workload_pair_seconds",
                    minimum=0.0,
                )
                workload_estimate = max(
                    _STAGE_ESTIMATES_SECONDS[next_stage],
                    policy.min_pairs * pair_estimate,
                )
            else:
                pair_estimate = 0.0
                workload_estimate = 0.0
            admitted, checkpoint = _admit_checkpoint_stage(
                checkpoint,
                next_stage,
                store=store,
                clock=clock,
                estimated_seconds=workload_estimate,
                candidate_id=candidate_id,
            )
            if not admitted:
                _budget_stop_output(args, checkpoint)
                return
            if not eligible_candidates:
                terminal_decision = rejected_candidates[0]
                workload_evidence = {
                    "status": "rejected_correctness",
                    "reason": "all finalists failed the sanitizer gate",
                }
            elif mode == "kernel-only":
                terminal_decision = evaluate_outer_candidate(
                    selected_candidate,
                    mode=mode,
                    workload_spec=None,
                    baseline=state.get("best_file"),
                    policy=policy,
                    confidence=state.get("confidence", 0.95),
                    candidate_root=iteration_dir,
                    input_hash=state["input_hash"],
                    iteration=args.iter,
                )
                workload_evidence = {"status": "not_applicable"}
            else:
                workload_spec = _workload_from_snapshot(state.get("workload"))
                evaluated = [
                    (
                        candidate,
                        evaluate_outer_candidate(
                            candidate,
                            mode=mode,
                            workload_spec=workload_spec,
                            baseline=state.get("best_file"),
                            policy=policy,
                            confidence=state.get("confidence", 0.95),
                            estimated_seconds_per_pair=pair_estimate,
                            budget_clock=clock,
                            now=time.monotonic(),
                            candidate_root=iteration_dir,
                            retries=args.retries,
                            seed=state.get("seed", 0),
                            input_hash=state["input_hash"],
                            iteration=args.iter,
                        ),
                    )
                    for candidate in eligible_candidates
                ]
                selected_candidate, terminal_decision = (
                    _select_terminal_outer_result(evaluated)
                )
                candidate_id = _candidate_checkpoint_id(selected_candidate)
                kernel = _publish_outer_candidate(
                    selected_candidate, iter_dir=iteration_dir
                )
                workload_evidence = {"status": "evaluated"}
            terminal_decision["candidate_file"] = kernel
            terminal_decision["candidate_sha256"] = sha256_file(kernel)
            terminal_decision = _strict_json_copy(
                terminal_decision, "terminal decision"
            )
            _write_workload_result_artifact(
                iteration_dir,
                state=state,
                terminal_decision=terminal_decision,
                candidate_id=candidate_id,
            )
            checkpoint = _complete_checkpoint_stage(
                checkpoint,
                next_stage,
                workload_evidence,
                store=store,
                clock=clock,
                candidate_id=candidate_id,
            )
            continue

        if next_stage == "decision":
            branch_payload = _load_lifecycle_branch_results(iteration_dir)
            if branch_payload["status"] == "no_confirmed_kernel_win":
                source_decision = _load_iteration_json(
                    iteration_dir, "decision.json"
                )
                if source_decision.get("status") not in {
                    "no_confirmed_kernel_win",
                    "inconclusive",
                }:
                    raise ValueError(
                        "decision.json conflicts with no-win branch results"
                    )
                candidate_id = None
                workload_result = None
            else:
                workload_result = _load_workload_result_artifact(
                    iteration_dir, state=state
                )
                source_decision = workload_result["decision"]
                candidate_id = workload_result.get("candidate_id")

            admitted, checkpoint = _admit_checkpoint_stage(
                checkpoint,
                next_stage,
                store=store,
                clock=clock,
                estimated_seconds=_STAGE_ESTIMATES_SECONDS[next_stage],
                candidate_id=candidate_id,
            )
            if not admitted:
                _decision, checkpoint = _persist_decision_budget_exhausted(
                    checkpoint,
                    source_decision,
                    iteration_dir=iteration_dir,
                    store=store,
                    clock=clock,
                    reason="decision_budget_exhausted",
                )
                _budget_stop_output(args, checkpoint)
                return

            if branch_payload["status"] == "no_confirmed_kernel_win":
                current_state = _read(state_path)
                if not _decision_record_already_applied(
                    current_state, args.iter, source_decision
                ):
                    state_before_producer = copy.deepcopy(current_state)
                    recorded = _run(
                        [
                            sys.executable,
                            str(SCRIPT_DIR / "state.py"),
                            "record-decision",
                            "--state",
                            state_path,
                            "--iter",
                            str(args.iter),
                            "--decision",
                            str(iteration_dir / "decision.json"),
                        ],
                        capture_output=True,
                        hard_timeout=_hard_timeout_seconds(clock),
                    )
                    if getattr(recorded, "timed_out", False):
                        _restore_state_after_producer_timeout(
                            state_path, state_before_producer
                        )
                        _decision, checkpoint = (
                            _persist_decision_budget_exhausted(
                                checkpoint,
                                source_decision,
                                iteration_dir=iteration_dir,
                                store=store,
                                clock=clock,
                                reason="decision_producer_timeout",
                            )
                        )
                        _budget_stop_output(args, checkpoint)
                        return
                    if recorded.returncode != 0:
                        raise SystemExit("state decision record failed")
                checkpoint = _complete_checkpoint_stage(
                    checkpoint,
                    next_stage,
                    {"status": "no_confirmed_kernel_win"},
                    store=store,
                    clock=clock,
                    candidate_status="no_confirmed_kernel_win",
                )
                state_manager.persist_checkpoint_snapshot(
                    state_path,
                    checkpoint,
                    Path(run_dir) / "checkpoint.json",
                )
                continue
            terminal_decision = source_decision
            kernel = workload_result["candidate_file"]
            checkpoint = transition_checkpoint(
                checkpoint,
                next_stage,
                status="in_progress",
                candidate_id=workload_result.get("candidate_id"),
                candidate_status=terminal_decision["status"],
                updated_at=time.monotonic(),
            )
            checkpoint = _persist_checkpoint(
                store, checkpoint, input_hash=state["input_hash"]
            )
            current_state = _read(state_path)
            if not _decision_already_applied(
                current_state, args.iter, terminal_decision
            ):
                state_before_producer = copy.deepcopy(current_state)
                applied = apply_decision(
                    terminal_decision,
                    run_dir=run_dir,
                    iteration=args.iter,
                    state_path=state_path,
                    kernel=kernel,
                    bench=_safe_iteration_file(iteration_dir, "bench.json"),
                    methods_json=_safe_iteration_file(
                        iteration_dir, Path(methods_json).name
                    ),
                    retries=args.retries,
                    attribution=attribution_path,
                    sass_check=sass_check_path,
                    skip_validation=True,
                    hard_timeout=_hard_timeout_seconds(clock),
                )
                if applied.get("timed_out"):
                    _restore_state_after_producer_timeout(
                        state_path, state_before_producer
                    )
                    _decision, checkpoint = _persist_decision_budget_exhausted(
                        checkpoint,
                        terminal_decision,
                        iteration_dir=iteration_dir,
                        store=store,
                        clock=clock,
                        reason="decision_producer_timeout",
                    )
                    _budget_stop_output(args, checkpoint)
                    return
                if applied["returncode"] != 0:
                    diagnostic = (
                        applied.get("stderr") or applied.get("stdout") or ""
                    ).strip()
                    raise SystemExit(
                        "state update failed"
                        + (f": {diagnostic}" if diagnostic else "")
                    )
            checkpoint = _complete_checkpoint_stage(
                checkpoint,
                next_stage,
                {"status": terminal_decision["status"]},
                store=store,
                clock=clock,
                candidate_id=workload_result.get("candidate_id"),
                candidate_status=terminal_decision["status"],
            )
            state_manager.persist_checkpoint_snapshot(
                state_path,
                checkpoint,
                Path(run_dir) / "checkpoint.json",
            )
            continue

        raise ValueError(f"unsupported lifecycle stage: {next_stage}")

def cmd_close_iter(args):
    state_path = os.path.join(args.run_dir, "state.json")
    if not os.path.isfile(state_path):
        sys.exit(f"state.json missing: {state_path}")

    state = _read(state_path)
    iter_dir = os.path.join(args.run_dir, f"iterv{args.iter}")
    methods_json = os.path.join(iter_dir, "methods.json")

    checkpoint_path = Path(args.run_dir) / "checkpoint.json"
    if (
        state.get("schema_version") == CURRENT_SCHEMA_VERSION
        and isinstance(state.get("budget"), Mapping)
        and isinstance(state.get("input_hash"), str)
        and checkpoint_path.is_file()
        and not checkpoint_path.is_symlink()
    ):
        return _cmd_close_iter_lifecycle(
            args, state, state_path, iter_dir, methods_json
        )
    if not os.path.isfile(methods_json):
        sys.exit(f"methods.json missing at {methods_json}")

    # Step 3e: Branch explore — compile + benchmark all branches
    branch_result = _run([
        sys.executable, str(SCRIPT_DIR / "branch_explore.py"),
        "--state", state_path,
        "--iter", str(args.iter),
        "--benchmark", os.path.abspath(args.benchmark),
        "--warmup", str(args.warmup),
        "--repeat", str(args.repeat),
    ], capture_output=True)
    sys.stderr.write(branch_result.stderr or "")

    if branch_result.returncode == 2:
        # All branches failed
        print(json.dumps({
            "iter": args.iter,
            "status": "all_branches_failed",
            "guidance": "The agent should fix the kernels and retry close-iter.",
        }, indent=2))
        sys.exit(2)
    if branch_result.returncode != 0:
        sys.exit(f"branch_explore failed rc={branch_result.returncode}")

    branch_results_path = os.path.join(iter_dir, "branch_results.json")
    branch_payload = _read_branch_results(branch_results_path)
    if branch_payload.get("status") == "no_confirmed_kernel_win":
        decision_path = os.path.join(iter_dir, "decision.json")
        print(json.dumps({
            "iter": args.iter,
            "status": "no_confirmed_kernel_win",
            "decision": decision_path,
            "state": state_path,
            "next_step": (
                "No candidate was promoted. Continue with another iteration "
                "or finalize the run with the current best."
            ),
        }, indent=2))
        return

    decision_path = os.path.join(iter_dir, "decision.json")
    terminal_decision = None
    budgeted_run = (
        state.get("schema_version") == CURRENT_SCHEMA_VERSION
        and isinstance(state.get("budget"), Mapping)
        and isinstance(state.get("input_hash"), str)
    )
    if budgeted_run:
        mode = state.get("mode", "kernel-only")
        policy = _policy_from_state(state)
        if mode == "full":
            candidates = select_outer_candidates(
                branch_payload.get("shortlist", []), policy.outer_candidates
            )
            if not candidates:
                raise ValueError("full mode shortlist contains no confirmed kernel win")
            workload_spec = _workload_from_snapshot(state.get("workload"))
            current_time = time.time()
            budget_clock = BudgetClock(
                policy, started_at=state.get("started_at", current_time)
            )
            checkpoint_path = Path(args.run_dir) / "checkpoint.json"
            checkpoint_state = None
            checkpoint_store = None
            if checkpoint_path.is_file() and not checkpoint_path.is_symlink():
                checkpoint_state = _validate_checkpoint(
                    _read(str(checkpoint_path)), input_hash=state["input_hash"]
                )
                checkpoint_store = ArtifactStore(args.run_dir)
            evaluated = []
            for candidate in candidates:
                iteration_now = time.time()
                elapsed = max(0.0, iteration_now - budget_clock.started_at)
                remaining = max(0.0, policy.max_seconds - elapsed)
                pair_estimate = state.get("estimated_workload_pair_seconds", 0.0)
                if checkpoint_state is not None:
                    admission = schedule_next(
                        checkpoint_state,
                        budget_clock,
                        policy.min_pairs
                        * _finite_real(
                            pair_estimate,
                            "estimated_workload_pair_seconds",
                            minimum=0.0,
                        ),
                        now=iteration_now,
                        store=checkpoint_store,
                        candidate_id=candidate.get("id", candidate.get("branch_index")),
                    )
                    if admission["status"] == "budget_exhausted":
                        checkpoint_state = admission
                terminal = evaluate_outer_candidate(
                    candidate,
                    mode="full",
                    workload_spec=workload_spec,
                    baseline=state.get("best_file"),
                    policy=policy,
                    confidence=state.get("confidence", 0.95),
                    remaining_seconds=remaining,
                    estimated_seconds_per_pair=pair_estimate,
                    budget_clock=budget_clock,
                    now=iteration_now,
                    candidate_root=iter_dir,
                    retries=args.retries,
                    seed=state.get("seed", 0),
                    input_hash=state["input_hash"],
                    iteration=args.iter,
                )
                evaluated.append((candidate, terminal))
            selected_candidate, terminal_decision = _select_terminal_outer_result(
                evaluated
            )
        elif mode == "kernel-only":
            champion = branch_payload.get("champion")
            if not isinstance(champion, Mapping):
                raise ValueError("branch_results champion must be a mapping")
            selected_candidate = dict(champion)
            selected_candidate["candidate_file"] = branch_payload.get(
                "selected_kernel"
            )
            terminal_decision = evaluate_outer_candidate(
                selected_candidate,
                mode="kernel-only",
                workload_spec=None,
                baseline=state.get("best_file"),
                policy=policy,
                confidence=state.get("confidence", 0.95),
                candidate_root=iter_dir,
                input_hash=state["input_hash"],
                iteration=args.iter,
            )
        else:
            raise ValueError("state mode must be full or kernel-only")

        kernel = _publish_outer_candidate(
            selected_candidate, iter_dir=Path(iter_dir).resolve()
        )
        terminal_decision["candidate_file"] = kernel
        terminal_decision["candidate_sha256"] = sha256_file(kernel)
        terminal_decision = _strict_json_copy(terminal_decision, "decision")
    else:
        # Preserve the legacy close-iter contract for existing v2 runs.
        kernel = _selected_kernel(
            branch_payload, iter_dir=iter_dir, decision_path=decision_path
        )

    bench_json = os.path.join(iter_dir, "bench.json")
    if not os.path.isfile(bench_json):
        sys.exit(f"bench.json missing for champion")

    bench = _read(bench_json)
    passed = bool(bench.get("correctness", {}).get("passed", False))

    if not passed:
        print(json.dumps({
            "iter": args.iter,
            "status": "validation_failed",
            "bench_json": bench_json,
            "guidance": "The agent should fix the kernel and re-run close-iter.",
        }, indent=2))
        sys.exit(2)

    # Step 3g: Profile champion with ncu (MANDATORY full report)
    rc = _run([
        sys.executable, str(SCRIPT_DIR / "profile_ncu.py"),
        "--state", state_path,
        "--iter", str(args.iter),
        "--which", "kernel",
        "--benchmark", os.path.abspath(args.benchmark),
        "--promote-if-best",
    ]).returncode
    if rc != 0:
        print("[warn] ncu profiling of champion failed", file=sys.stderr)

    # Step 3h: Ablation attribution (optional — runs if ablation kernels exist)
    attribution_path = os.path.join(iter_dir, "attribution.json")
    ablation_dir = os.path.join(iter_dir, "ablations")
    if os.path.isdir(ablation_dir):
        _run([
            sys.executable, str(SCRIPT_DIR / "ablate.py"),
            "--state", state_path,
            "--iter", str(args.iter),
            "--benchmark", os.path.abspath(args.benchmark),
        ])

    # Step 3i: SASS verification
    sass_check_path = os.path.join(iter_dir, "sass_check.json")
    _run([
        sys.executable, str(SCRIPT_DIR / "sass_check.py"),
        "--state", state_path,
        "--iter", str(args.iter),
    ])

    # Step 3j: persist the terminal decision before the explicit state update.
    if terminal_decision is None:
        terminal_decision = _read(decision_path)
    applied = apply_decision(
        terminal_decision,
        run_dir=args.run_dir,
        iteration=args.iter,
        state_path=state_path,
        kernel=kernel,
        bench=bench_json,
        methods_json=methods_json,
        retries=args.retries,
        attribution=attribution_path,
        sass_check=sass_check_path,
    )
    if applied["returncode"] != 0:
        sys.exit("state update failed")

    # Open next iteration if needed
    state = _read(state_path)
    next_iter = args.iter + 1
    if next_iter <= state["iterations_total"]:
        # Profile best_input for next iter + roofline
        _run([
            sys.executable, str(SCRIPT_DIR / "profile_ncu.py"),
            "--state", state_path,
            "--iter", str(next_iter),
            "--which", "best_input",
            "--benchmark", os.path.abspath(args.benchmark),
        ])
        _run([
            sys.executable, str(SCRIPT_DIR / "roofline.py"),
            "--state", state_path,
            "--iter", str(next_iter),
        ])

        # Check early stop
        roofline_path = os.path.join(state["run_dir"], f"iterv{next_iter}", "roofline.json")
        early_stop = False
        if os.path.isfile(roofline_path):
            roofline = _read(roofline_path)
            early_stop = roofline.get("near_peak", False)
    else:
        early_stop = False

    print(json.dumps({
        "iter": args.iter,
        "status": "closed",
        "best_ms": state.get("best_metric_ms"),
        "next_iter": next_iter if next_iter <= state["iterations_total"] else None,
        "early_stop": early_stop,
        "state": state_path,
    }, indent=2))


# ---------------------------------------------------------------------------
# finalize  —  step 4
# ---------------------------------------------------------------------------

def cmd_finalize(args):
    state_path = os.path.join(args.run_dir, "state.json")
    summary_path = os.path.join(args.run_dir, "summary.md")
    checkpoint_path = Path(args.run_dir) / "checkpoint.json"
    complete_checkpoint = None
    if checkpoint_path.is_symlink():
        raise ValueError("checkpoint.json must not be a symlink")
    if checkpoint_path.is_file():
        current = _validate_checkpoint(_read(str(checkpoint_path)))
        if current["stage"] == "complete":
            complete_checkpoint = current
        elif current["stage"] == "decision" and current["status"] == "stage_complete":
            complete_checkpoint = transition_checkpoint(
                current,
                "complete",
                status="complete",
                updated_at=time.time(),
            )
        else:
            raise ValueError(
                "cannot finalize before the decision stage checkpoint is complete"
            )
    if complete_checkpoint is not None:
        ArtifactStore(args.run_dir).write_checkpoint(complete_checkpoint)
        state_manager.persist_checkpoint_snapshot(
            state_path, complete_checkpoint, checkpoint_path
        )
    rc = _run([
        sys.executable, str(SCRIPT_DIR / "summarize.py"),
        "--state", state_path,
        "--out", summary_path,
    ]).returncode
    if rc != 0:
        sys.exit("summarize failed")
    print(json.dumps({"summary": summary_path}, indent=2))


def _safe_run_file(run_dir: Path, name: str) -> Path:
    path = run_dir / name
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError(f"{name} is missing: {path}") from error
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{name} must be a regular file")
    if path.resolve(strict=True).parent != run_dir:
        raise ValueError(f"{name} escapes the run directory")
    return path


def _verify_state_candidates(state: Mapping) -> None:
    candidates = state.get("candidates", {})
    if not isinstance(candidates, Mapping):
        raise ValueError("state candidates must be a mapping")
    for candidate_id, record in candidates.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"state candidate {candidate_id!r} is malformed")
        path = record.get("candidate_file") or record.get("path")
        digest = record.get("candidate_sha256") or record.get("sha256")
        if path is None and digest is None:
            continue
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError(f"state candidate {candidate_id!r} binding is malformed")
        candidate = Path(path).expanduser()
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"state candidate {candidate_id!r} file drifted")
        if sha256_file(candidate) != digest.lower():
            raise ValueError(f"state candidate {candidate_id!r} file drifted")


def _load_and_verify_manifest(run_dir: Path) -> tuple[dict, str]:
    manifest_path = _safe_run_file(run_dir, "manifest.json")
    try:
        manifest = _read(str(manifest_path))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"run manifest is malformed: {error}") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must contain a JSON object")
    if manifest.get("schema_version") != CURRENT_SCHEMA_VERSION:
        raise ValueError("manifest schema_version is invalid")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping) or not {"baseline", "ref"}.issubset(inputs):
        raise ValueError("manifest inputs must contain baseline and ref")
    for name, record in inputs.items():
        if not isinstance(name, str) or not isinstance(record, Mapping):
            raise ValueError("manifest input record is malformed")
        path_value = record.get("path")
        declared_hash = record.get("sha256")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"manifest input {name} path is malformed")
        if (
            not isinstance(declared_hash, str)
            or len(declared_hash) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in declared_hash)
        ):
            raise ValueError(f"manifest input {name} sha256 is malformed")
        source = Path(path_value).expanduser()
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"manifest input {name} drifted or is unsafe")
        if sha256_file(source) != declared_hash.lower():
            raise ValueError(f"manifest input {name} drifted from frozen sha256")
        size = record.get("size_bytes")
        if size is not None and (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or source.stat().st_size != size
        ):
            raise ValueError(f"manifest input {name} size drifted")

    required = {
        "workload",
        "dims",
        "backend",
        "budget",
        "confidence",
        "min_effect_pct",
        "ptr_size",
        "input_hash",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"manifest missing frozen field: {missing[0]}")
    recomputed = _frozen_input_hash(
        manifest,
        workload=manifest["workload"],
        dims=manifest["dims"],
        backend=manifest["backend"],
        budget=manifest["budget"],
        confidence=manifest["confidence"],
        min_effect_pct=manifest["min_effect_pct"],
        ptr_size=manifest["ptr_size"],
    )
    if manifest.get("input_hash") != recomputed:
        raise ValueError("manifest frozen input_hash does not match current inputs")
    return dict(manifest), recomputed


def cmd_resume(args) -> None:
    run_arg = Path(args.run_dir).expanduser()
    if run_arg.is_symlink():
        raise ValueError("run directory must not be a symlink")
    try:
        run_dir = run_arg.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"run directory does not exist: {run_arg}") from error
    if not run_dir.is_dir():
        raise ValueError("run directory must be a directory")
    checkpoint_path = _safe_run_file(run_dir, "checkpoint.json")
    state_path = _safe_run_file(run_dir, "state.json")
    try:
        checkpoint = _read(str(checkpoint_path))
        state = _read(str(state_path))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"run checkpoint/state is malformed: {error}") from error
    state_manager.validate_state(state)
    declared_run_dir = state.get("run_dir")
    if not isinstance(declared_run_dir, str) or Path(declared_run_dir).resolve() != run_dir:
        raise ValueError("state run_dir does not match --run-dir")
    if checkpoint.get("run_dir") is not None and Path(checkpoint["run_dir"]).resolve() != run_dir:
        raise ValueError("checkpoint run_dir does not match --run-dir")
    if checkpoint.get("input_hash") != state.get("input_hash"):
        raise ValueError("checkpoint/state frozen input_hash mismatch")
    _verify_state_candidates(state)
    manifest, manifest_hash = _load_and_verify_manifest(run_dir)
    if state.get("input_hash") != manifest_hash:
        raise ValueError("manifest/state frozen input_hash mismatch")
    if state.get("workload") != manifest.get("workload"):
        raise ValueError("manifest/state frozen workload mismatch")
    workload_spec = _workload_from_snapshot(manifest.get("workload"))
    if workload_spec is not None:
        verify_frozen_spec(workload_spec)
    policy = _policy_from_state(state)
    restored = resume(
        checkpoint, input_hash=manifest_hash, max_rounds=policy.max_rounds
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "status": restored["status"],
                "stage": restored["stage"],
                "next_stage": restored["next_stage"],
                "next_iteration": restored["next_iteration"],
                "input_hash": restored["input_hash"],
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _removed_noise_option(_text: str):
    raise argparse.ArgumentTypeError(
        "--noise-threshold-pct was removed; migrate to --min-effect-pct"
    )


def _removed_env_option(_text: str):
    raise argparse.ArgumentTypeError(
        "--env-out was removed; environment evidence is always stored in run_dir/env.json"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Budget-aware CUDA kernel optimization orchestrator"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    _default_bench = str(SCRIPT_DIR / "benchmark.py")

    ps = sub.add_parser("setup", help="preflight and initialize a frozen run")
    ps.add_argument("--baseline", required=True)
    ps.add_argument("--ref", required=True)
    ps.add_argument("--benchmark", default=_default_bench)
    ps.add_argument("--ncu-num", type=int, default=5)
    ps.add_argument(
        "--budget",
        choices=("quick", "balanced", "thorough", "custom"),
        default="balanced",
    )
    ps.add_argument("--max-seconds", type=_cli_positive_int, default=None)
    ps.add_argument("--max-rounds", type=_cli_positive_int, default=None)
    ps.add_argument("--iterations", type=_cli_positive_int, default=None)
    ps.add_argument("--branches", type=_cli_positive_int, default=None)
    ps.add_argument("--min-pairs", type=_cli_positive_int, default=None)
    ps.add_argument("--max-pairs", type=_cli_positive_int, default=None)
    ps.add_argument("--outer-candidates", type=_cli_positive_int, default=None)
    ps.add_argument("--confidence", type=float, default=0.95)
    ps.add_argument("--min-effect-pct", type=float, default=0.5)
    ps.add_argument("--output-root", default=None)
    ps.add_argument("--dims", required=True, help="JSON dict of name->int")
    ps.add_argument("--backend", choices=("auto", "cuda", "cutlass", "triton"), default="auto")
    ps.add_argument("--workload", default=None)
    ps.add_argument("--workload-cmd", default=None)
    ps.add_argument("--workload-manifest", default=None)
    ps.add_argument("--objective", default=None)
    ps.add_argument(
        "--noise-threshold-pct",
        type=_removed_noise_option,
        default=None,
        help=argparse.SUPPRESS,
    )
    ps.add_argument(
        "--env-out",
        type=_removed_env_option,
        default=None,
        help=argparse.SUPPRESS,
    )
    ps.add_argument("--ptr-size", type=int, default=0)
    ps.add_argument("--warmup", type=int, default=10)
    ps.add_argument("--repeat", type=int, default=20)
    ps.set_defaults(func=cmd_setup)

    pr = sub.add_parser("resume", help="validate and resume a frozen run")
    pr.add_argument("--run-dir", required=True)
    pr.set_defaults(func=cmd_resume)

    po = sub.add_parser("open-iter")
    po.add_argument("--run-dir", required=True)
    po.add_argument("--iter", type=int, required=True)
    po.add_argument("--benchmark", default=_default_bench)
    po.set_defaults(func=cmd_open_iter)

    pc = sub.add_parser("close-iter")
    pc.add_argument("--run-dir", required=True)
    pc.add_argument("--iter", type=int, required=True)
    pc.add_argument("--benchmark", default=_default_bench)
    pc.add_argument("--warmup", type=int, default=10)
    pc.add_argument("--repeat", type=int, default=20)
    pc.add_argument("--retries", type=int, default=0)
    pc.set_defaults(func=cmd_close_iter)

    pf = sub.add_parser("finalize")
    pf.add_argument("--run-dir", required=True)
    pf.set_defaults(func=cmd_finalize)

    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
