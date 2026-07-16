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
    sha256_file,
)
from budget import BudgetClock, BudgetPolicy, resolve_budget  # noqa: E402
import decision as decision_engine  # noqa: E402
import preflight  # noqa: E402
import state as state_manager  # noqa: E402
from workload_adapter import (  # noqa: E402
    WorkloadSpec,
    normalize_workload,
    run_spec_once,
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
    for field in (
        "max_seconds",
        "max_rounds",
        "branches",
        "min_pairs",
        "max_pairs",
        "outer_candidates",
    ):
        value = getattr(args, field, None)
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
    if target_index < current_index:
        raise ValueError("checkpoint stage order cannot move backward")
    if target_index > current_index + 1:
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
    result = copy.deepcopy(current)
    result.update(
        {
            "stage": stage,
            "stage_index": target_index,
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


def resume(checkpoint, *, input_hash: str) -> dict:
    """Validate and detach resumable state without replaying completed work."""
    current = _validate_checkpoint(checkpoint, input_hash=input_hash)
    result = copy.deepcopy(current)
    if current["stage"] == "complete":
        result["status"] = "complete"
        result["next_stage"] = "complete"
        return result
    if current["status"] == "stage_complete":
        result["next_stage"] = STAGES[current["stage_index"] + 1]
    else:
        result["next_stage"] = current["stage"]
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
    current_time = time.time() if now is None else _finite_real(now, "now")
    elapsed = max(0.0, current_time - clock.started_at)
    remaining = clock.remaining_seconds(now=current_time)
    if isinstance(state, Mapping) and "stage" not in state:
        checkpoint = {
            "schema_version": state.get("schema_version", CURRENT_SCHEMA_VERSION),
            "input_hash": state.get("input_hash"),
            "run_dir": state.get("run_dir"),
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
        "elapsed_seconds": max(0.0, now - clock.started_at),
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
    current_time = time.time() if now is None else _finite_real(now, "now")
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
    current_time = time.time() if now is None else _finite_real(now, "now")
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
    current_time = time.time() if now is None else _finite_real(now, "now")
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


def _candidate_file(candidate: Mapping) -> str:
    value = candidate.get("candidate_file") or candidate.get("kernel")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("candidate must provide candidate_file or kernel")
    path = Path(value).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError("candidate artifact must be a non-symlink regular file")
    return str(path.resolve(strict=True))


def build_terminal_decision(
    *,
    mode: str,
    candidate: Mapping,
    workload_result=None,
    decide_fn=None,
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
    candidate_file = _candidate_file(candidate)
    result["candidate_file"] = candidate_file
    result["candidate_sha256"] = sha256_file(candidate_file)
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
) -> dict:
    """Evaluate one confirmed inner winner through the applicable outer loop."""
    candidate_path = Path(_candidate_file(candidate))
    if candidate_path.suffix not in {".cu", ".py"}:
        raise ValueError("outer candidate must be a .cu or .py kernel")
    if candidate_root is not None:
        iteration_root = Path(candidate_root).expanduser().resolve(strict=True)
        try:
            candidate_path.relative_to(iteration_root)
        except ValueError as error:
            raise ValueError("outer candidate escapes the current iteration") from error
    normalized_candidate = dict(candidate)
    normalized_candidate["candidate_file"] = str(candidate_path)
    if mode == "kernel-only":
        return build_terminal_decision(mode=mode, candidate=normalized_candidate)
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
        current_time = time.time() if now is None else _finite_real(now, "now")
        if not budget_clock.can_start(
            now=current_time,
            estimated_seconds=policy.min_pairs * pair_estimate,
        ):
            budget_exhausted = True
        if pair_estimate > 0.0:
            execution_remaining = max(
                0.0,
                budget_clock.started_at
                + budget_clock.policy.max_seconds
                - budget_clock.policy.reserve_seconds
                - current_time,
            )
            blocks = min(blocks, int(execution_remaining // pair_estimate))
    if remaining_seconds is not None:
        remaining = _finite_real(
            remaining_seconds, "remaining_seconds", minimum=0.0
        )
        if pair_estimate > 0.0:
            blocks = min(blocks, int(remaining // pair_estimate))
    if budget_exhausted or blocks < policy.min_pairs:
        workload_result = {
            "status": "workload_failed",
            "reason": "budget_exhausted",
            "candidate_status": "inconclusive",
        }
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
    terminal = build_terminal_decision(
        mode=mode,
        candidate=normalized_candidate,
        workload_result=workload_result,
    )
    if budget_exhausted or blocks < policy.min_pairs:
        terminal["candidate_status"] = "inconclusive"
        terminal["budget_exhausted"] = True
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
    completed = selected_runner(command)
    return {
        "decision_path": str(decision_path),
        "command": command,
        "returncode": completed.returncode,
        "stdout": getattr(completed, "stdout", None),
        "stderr": getattr(completed, "stderr", None),
    }


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"[run] {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, text=True, **kw)


def _read(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    )


def _publish_outer_candidate(candidate: Mapping, *, iter_dir: Path) -> str:
    source = Path(_candidate_file(candidate))
    suffix = source.suffix
    if suffix not in {".cu", ".py"}:
        raise ValueError("outer candidate must be a .cu or .py kernel")
    destination = iter_dir / f"kernel{suffix}"
    if source != destination.resolve(strict=False):
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=str(iter_dir)
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copy2(source, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
    for stale_suffix in {".cu", ".py"} - {suffix}:
        stale = iter_dir / f"kernel{stale_suffix}"
        if stale.is_symlink() or stale.is_file():
            stale.unlink()
        elif stale.exists():
            raise ValueError("stale iteration kernel is not a regular file")
    source_bench = source.parent / "bench.json"
    destination_bench = iter_dir / "bench.json"
    if source_bench.is_file() and source_bench.resolve() != destination_bench.resolve():
        shutil.copy2(source_bench, destination_bench)
    return str(destination.resolve(strict=True))


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

def _strict_json_object(text: str, field: str) -> dict:
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
            "--workload-json",
            json.dumps(workload_snapshot, sort_keys=True, allow_nan=False),
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
        now = time.time()
        checkpoint = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "input_hash": manifest["input_hash"],
            "run_dir": str(run_dir),
            "stage": STAGES[0],
            "stage_index": 0,
            "status": "stage_complete",
            "candidate_id": None,
            "candidate_status": None,
            "stage_evidence": {"baseline": {"status": "passed"}},
            "budget": {
                "elapsed_seconds": max(0.0, now - started_at),
                "remaining_seconds": max(
                    0.0, policy.max_seconds - (now - started_at)
                ),
            },
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


def _cmd_close_iter_lifecycle(args, state, state_path, iter_dir, methods_json):
    policy = _policy_from_state(state)
    started_at = _finite_real(
        state.get("started_at", time.time()), "state started_at"
    )
    clock = BudgetClock(policy, started_at=started_at)
    store = ArtifactStore(args.run_dir)
    checkpoint = _validate_checkpoint(
        store.load_checkpoint(expected_input_hash=state["input_hash"]),
        input_hash=state["input_hash"],
    )

    admitted, checkpoint = _admit_checkpoint_stage(
        checkpoint,
        "candidate_correctness",
        store=store,
        clock=clock,
        estimated_seconds=_STAGE_ESTIMATES_SECONDS["candidate_correctness"],
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
    )
    sys.stderr.write(branch_result.stderr or "")
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
        raise SystemExit(f"branch_explore failed rc={branch_result.returncode}")

    branch_results_path = os.path.join(iter_dir, "branch_results.json")
    branch_payload = _read_branch_results(branch_results_path)
    champion = branch_payload.get("champion")
    candidate_id = _candidate_checkpoint_id(champion)
    checkpoint = _complete_checkpoint_stage(
        checkpoint,
        "candidate_correctness",
        {"status": "passed"},
        store=store,
        clock=clock,
        candidate_id=candidate_id,
    )
    paired_status = (
        "completed"
        if branch_payload.get("status") == "no_confirmed_kernel_win"
        else "passed"
    )
    checkpoint = _complete_checkpoint_stage(
        checkpoint,
        "candidate_paired",
        {
            "status": paired_status,
            "completed_comparisons": branch_payload.get(
                "completed_comparisons", 0
            ),
        },
        store=store,
        clock=clock,
        candidate_id=candidate_id,
    )

    decision_path = os.path.join(iter_dir, "decision.json")
    if branch_payload.get("status") == "no_confirmed_kernel_win":
        checkpoint = _complete_checkpoint_stage(
            checkpoint,
            "candidate_profile",
            {"status": "not_applicable"},
            store=store,
            clock=clock,
        )
        checkpoint = _complete_checkpoint_stage(
            checkpoint,
            "candidate_sanitizer",
            {
                "status": "deferred",
                "reason": "sanitizer stage is not implemented yet",
            },
            store=store,
            clock=clock,
        )
        checkpoint = _complete_checkpoint_stage(
            checkpoint,
            "workload_paired",
            {"status": "not_applicable"},
            store=store,
            clock=clock,
        )
        _complete_checkpoint_stage(
            checkpoint,
            "decision",
            {"status": "no_confirmed_kernel_win"},
            store=store,
            clock=clock,
            candidate_status="no_confirmed_kernel_win",
        )
        print(
            json.dumps(
                {
                    "iter": args.iter,
                    "status": "no_confirmed_kernel_win",
                    "decision": decision_path,
                    "state": state_path,
                    "next_step": (
                        "No candidate was promoted. Continue with another iteration "
                        "or finalize the run with the current best."
                    ),
                },
                indent=2,
            )
        )
        return

    mode = state.get("mode", "kernel-only")
    if mode == "full":
        candidates = select_outer_candidates(
            branch_payload.get("shortlist", []), policy.outer_candidates
        )
        if not candidates:
            raise ValueError("full mode shortlist contains no confirmed kernel win")
        selected_candidate = candidates[0]
    elif mode == "kernel-only":
        if not isinstance(champion, Mapping):
            raise ValueError("branch_results champion must be a mapping")
        selected_candidate = dict(champion)
        selected_candidate["candidate_file"] = branch_payload.get(
            "selected_kernel"
        )
        candidates = [selected_candidate]
    else:
        raise ValueError("state mode must be full or kernel-only")

    candidate_id = _candidate_checkpoint_id(selected_candidate)
    kernel = _publish_outer_candidate(
        selected_candidate, iter_dir=Path(iter_dir).resolve()
    )
    bench_json = os.path.join(iter_dir, "bench.json")
    if not os.path.isfile(bench_json):
        raise SystemExit("bench.json missing for champion")
    bench = _read(bench_json)
    if not bool(bench.get("correctness", {}).get("passed", False)):
        print(
            json.dumps(
                {
                    "iter": args.iter,
                    "status": "validation_failed",
                    "bench_json": bench_json,
                    "guidance": "The agent should fix the kernel and re-run close-iter.",
                },
                indent=2,
            )
        )
        raise SystemExit(2)

    admitted, checkpoint = _admit_checkpoint_stage(
        checkpoint,
        "candidate_profile",
        store=store,
        clock=clock,
        estimated_seconds=_STAGE_ESTIMATES_SECONDS["candidate_profile"],
        candidate_id=candidate_id,
    )
    if not admitted:
        _budget_stop_output(args, checkpoint)
        return
    profile_rc = _run(
        [
            sys.executable,
            str(SCRIPT_DIR / "profile_ncu.py"),
            "--state",
            state_path,
            "--iter",
            str(args.iter),
            "--which",
            "kernel",
            "--benchmark",
            os.path.abspath(args.benchmark),
            "--promote-if-best",
        ]
    ).returncode
    checkpoint = _complete_checkpoint_stage(
        checkpoint,
        "candidate_profile",
        {"status": "passed" if profile_rc == 0 else "deferred", "returncode": profile_rc},
        store=store,
        clock=clock,
        candidate_id=candidate_id,
    )

    admitted, checkpoint = _admit_checkpoint_stage(
        checkpoint,
        "candidate_sanitizer",
        store=store,
        clock=clock,
        estimated_seconds=_STAGE_ESTIMATES_SECONDS["candidate_sanitizer"],
        candidate_id=candidate_id,
    )
    if not admitted:
        _budget_stop_output(args, checkpoint)
        return
    attribution_path = os.path.join(iter_dir, "attribution.json")
    ablation_dir = os.path.join(iter_dir, "ablations")
    if os.path.isdir(ablation_dir):
        if not clock.can_start(
            now=time.time(),
            estimated_seconds=_STAGE_ESTIMATES_SECONDS["ablation"],
        ):
            denied = schedule_next(
                checkpoint,
                clock,
                _STAGE_ESTIMATES_SECONDS["ablation"],
                now=time.time(),
                store=store,
                candidate_id=candidate_id,
            )
            _budget_stop_output(args, denied)
            return
        _run(
            [
                sys.executable,
                str(SCRIPT_DIR / "ablate.py"),
                "--state",
                state_path,
                "--iter",
                str(args.iter),
                "--benchmark",
                os.path.abspath(args.benchmark),
            ]
        )

    if not clock.can_start(
        now=time.time(),
        estimated_seconds=_STAGE_ESTIMATES_SECONDS["candidate_sanitizer"],
    ):
        denied = schedule_next(
            checkpoint,
            clock,
            _STAGE_ESTIMATES_SECONDS["candidate_sanitizer"],
            now=time.time(),
            store=store,
            candidate_id=candidate_id,
        )
        _budget_stop_output(args, denied)
        return
    sass_check_path = os.path.join(iter_dir, "sass_check.json")
    sass_rc = _run(
        [
            sys.executable,
            str(SCRIPT_DIR / "sass_check.py"),
            "--state",
            state_path,
            "--iter",
            str(args.iter),
        ]
    ).returncode
    checkpoint = _complete_checkpoint_stage(
        checkpoint,
        "candidate_sanitizer",
        {
            "status": "deferred",
            "reason": "sanitizer stage is not implemented yet",
            "sass_status": "passed" if sass_rc == 0 else "failed",
        },
        store=store,
        clock=clock,
        candidate_id=candidate_id,
    )

    if mode == "kernel-only":
        terminal_decision = evaluate_outer_candidate(
            selected_candidate,
            mode="kernel-only",
            workload_spec=None,
            baseline=state.get("best_file"),
            policy=policy,
            confidence=state.get("confidence", 0.95),
            candidate_root=iter_dir,
        )
        checkpoint = _complete_checkpoint_stage(
            checkpoint,
            "workload_paired",
            {"status": "not_applicable"},
            store=store,
            clock=clock,
            candidate_id=candidate_id,
        )
    else:
        pair_estimate = _finite_real(
            state.get("estimated_workload_pair_seconds", 0.0),
            "estimated_workload_pair_seconds",
            minimum=0.0,
        )
        workload_estimate = max(
            _STAGE_ESTIMATES_SECONDS["workload_paired"],
            policy.min_pairs * pair_estimate,
        )
        admitted, checkpoint = _admit_checkpoint_stage(
            checkpoint,
            "workload_paired",
            store=store,
            clock=clock,
            estimated_seconds=workload_estimate,
            candidate_id=candidate_id,
        )
        if not admitted:
            _budget_stop_output(args, checkpoint)
            return
        workload_spec = _workload_from_snapshot(state.get("workload"))
        evaluated = []
        for candidate in candidates:
            evaluated.append(
                (
                    candidate,
                    evaluate_outer_candidate(
                        candidate,
                        mode="full",
                        workload_spec=workload_spec,
                        baseline=state.get("best_file"),
                        policy=policy,
                        confidence=state.get("confidence", 0.95),
                        estimated_seconds_per_pair=pair_estimate,
                        budget_clock=clock,
                        now=time.time(),
                        candidate_root=iter_dir,
                        retries=args.retries,
                        seed=state.get("seed", 0),
                    ),
                )
            )
        selected_candidate, terminal_decision = _select_terminal_outer_result(
            evaluated
        )
        candidate_id = _candidate_checkpoint_id(selected_candidate)
        kernel = _publish_outer_candidate(
            selected_candidate, iter_dir=Path(iter_dir).resolve()
        )
        checkpoint = _complete_checkpoint_stage(
            checkpoint,
            "workload_paired",
            {"status": "evaluated"},
            store=store,
            clock=clock,
            candidate_id=candidate_id,
        )

    terminal_decision["candidate_file"] = kernel
    terminal_decision["candidate_sha256"] = sha256_file(kernel)
    terminal_decision = _strict_json_copy(terminal_decision, "decision")
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
        skip_validation=True,
    )
    if applied["returncode"] != 0:
        diagnostic = (applied.get("stderr") or applied.get("stdout") or "").strip()
        raise SystemExit(
            "state update failed" + (f": {diagnostic}" if diagnostic else "")
        )
    checkpoint = _complete_checkpoint_stage(
        checkpoint,
        "decision",
        {"status": terminal_decision["status"]},
        store=store,
        clock=clock,
        candidate_id=candidate_id,
        candidate_status=terminal_decision["status"],
    )
    updated_state = _read(state_path)
    print(
        json.dumps(
            {
                "iter": args.iter,
                "status": "closed",
                "best_ms": updated_state.get("best_metric_ms"),
                "next_iter": None,
                "early_stop": False,
                "state": state_path,
            },
            indent=2,
        )
    )

def cmd_close_iter(args):
    state_path = os.path.join(args.run_dir, "state.json")
    if not os.path.isfile(state_path):
        sys.exit(f"state.json missing: {state_path}")

    state = _read(state_path)
    iter_dir = os.path.join(args.run_dir, f"iterv{args.iter}")
    methods_json = os.path.join(iter_dir, "methods.json")
    if not os.path.isfile(methods_json):
        sys.exit(f"methods.json missing at {methods_json}")

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
    rc = _run([
        sys.executable, str(SCRIPT_DIR / "summarize.py"),
        "--state", state_path,
        "--out", summary_path,
    ]).returncode
    if rc != 0:
        sys.exit("summarize failed")
    if complete_checkpoint is not None:
        ArtifactStore(args.run_dir).write_checkpoint(complete_checkpoint)
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
    _manifest, manifest_hash = _load_and_verify_manifest(run_dir)
    if state.get("input_hash") != manifest_hash:
        raise ValueError("manifest/state frozen input_hash mismatch")
    restored = resume(checkpoint, input_hash=manifest_hash)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "status": restored["status"],
                "stage": restored["stage"],
                "next_stage": restored["next_stage"],
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
