#!/usr/bin/env python3
"""Budget presets and deadline admission for optimizer work."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class BudgetPolicy:
    name: str
    soft_target_seconds: int
    max_seconds: int
    branches: int
    max_rounds: int
    min_pairs: int
    max_pairs: int
    outer_candidates: int
    max_cases: int | None
    sanitizer_mode: str
    reserve_seconds: int = 300


PRESETS = {
    "quick": BudgetPolicy("quick", 900, 2700, 4, 2, 20, 50, 1, 3, "targeted"),
    "balanced": BudgetPolicy("balanced", 3600, 10800, 8, 4, 20, 100, 2, 10, "targeted"),
    "thorough": BudgetPolicy("thorough", 14400, 36000, 16, 8, 30, 200, 3, None, "full"),
}

_REQUIRED_POSITIVE_FIELDS = (
    "max_seconds",
    "branches",
    "max_rounds",
    "min_pairs",
    "max_pairs",
    "outer_candidates",
)
_OPTIONAL_OVERRIDE_FIELDS = {
    "max_cases",
    "sanitizer_mode",
    "reserve_seconds",
    "soft_target_seconds",
}
_VALID_SANITIZER_MODES = {"targeted", "full"}


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_time(value: object, parameter: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{parameter} must be a finite number")
    try:
        numeric = float(value)
    except OverflowError:
        raise ValueError(f"{parameter} must be a finite number") from None
    if not math.isfinite(numeric):
        raise ValueError(f"{parameter} must be a finite number")
    return numeric


def _validate_policy(policy: BudgetPolicy) -> None:
    for field in _REQUIRED_POSITIVE_FIELDS:
        if not _is_positive_int(getattr(policy, field)):
            raise ValueError(f"{field} must be a positive integer")
    if policy.min_pairs > policy.max_pairs:
        raise ValueError("min_pairs must be less than or equal to max_pairs")
    if policy.max_cases is not None and not _is_positive_int(policy.max_cases):
        raise ValueError("max_cases must be a positive integer or None")
    if policy.sanitizer_mode not in _VALID_SANITIZER_MODES:
        raise ValueError(
            "sanitizer_mode must be one of: "
            + ", ".join(sorted(_VALID_SANITIZER_MODES))
        )
    if (
        not isinstance(policy.reserve_seconds, int)
        or isinstance(policy.reserve_seconds, bool)
        or policy.reserve_seconds < 0
    ):
        raise ValueError("reserve_seconds must be a non-negative integer")
    if policy.reserve_seconds >= policy.max_seconds:
        raise ValueError("reserve_seconds must be less than max_seconds")
    if not _is_positive_int(policy.soft_target_seconds):
        raise ValueError("soft_target_seconds must be a positive integer")
    if policy.soft_target_seconds > policy.max_seconds:
        raise ValueError("soft_target_seconds must not exceed max_seconds")


def resolve_budget(name: str, **overrides: object) -> BudgetPolicy:
    """Return a validated copy of a preset with explicit overrides applied."""
    allowed = set(_REQUIRED_POSITIVE_FIELDS) | _OPTIONAL_OVERRIDE_FIELDS
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise TypeError(f"unknown budget override: {', '.join(unknown)}")

    if name == "custom":
        for field in _REQUIRED_POSITIVE_FIELDS:
            if field in overrides and not _is_positive_int(overrides[field]):
                raise ValueError(f"{field} must be a positive integer")
        missing = [field for field in _REQUIRED_POSITIVE_FIELDS if field not in overrides]
        if missing:
            raise ValueError(f"custom budget requires {', '.join(missing)}")
        policy = BudgetPolicy(
            name="custom",
            soft_target_seconds=overrides.get(
                "soft_target_seconds",
                max(1, int(overrides["max_seconds"]) // 3),
            ),
            max_seconds=overrides["max_seconds"],
            branches=overrides["branches"],
            max_rounds=overrides["max_rounds"],
            min_pairs=overrides["min_pairs"],
            max_pairs=overrides["max_pairs"],
            outer_candidates=overrides["outer_candidates"],
            max_cases=overrides.get("max_cases"),
            sanitizer_mode=overrides.get("sanitizer_mode", "targeted"),
            reserve_seconds=overrides.get("reserve_seconds", 300),
        )
    else:
        if name not in PRESETS:
            raise ValueError(f"unknown budget preset: {name}")
        policy = replace(PRESETS[name], **overrides)

    _validate_policy(policy)
    return policy


@dataclass(frozen=True)
class BudgetClock:
    policy: BudgetPolicy
    started_at: float
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        _validate_time(self.started_at, "started_at")
        elapsed = _validate_time(self.elapsed_seconds, "elapsed_seconds")
        if elapsed < 0.0:
            raise ValueError("elapsed_seconds must be a non-negative finite number")

    def elapsed(self, *, now: float) -> float:
        current = _validate_time(now, "now")
        return self.elapsed_seconds + max(0.0, current - self.started_at)

    def can_start(self, *, now: float, estimated_seconds: float) -> bool:
        estimate = max(
            0.0, _validate_time(estimated_seconds, "estimated_seconds")
        )
        return estimate <= self.execution_seconds_available(now=now)

    def execution_seconds_available(self, *, now: float) -> float:
        return max(
            0.0,
            self.policy.max_seconds
            - self.policy.reserve_seconds
            - self.elapsed(now=now),
        )

    def remaining_seconds(self, *, now: float) -> float:
        return max(0.0, self.policy.max_seconds - self.elapsed(now=now))

    def soft_remaining_seconds(self, *, now: float) -> float:
        return max(0.0, self.policy.soft_target_seconds - self.elapsed(now=now))


_CANDIDATE_STAGES = (
    "static_review",
    "build_correctness",
    "short_paired",
    "profiler",
    "formal_paired",
    "service",
)
_CLAIM_LAST_STAGE = {
    "kernel": "formal_paired",
    "workload": "formal_paired",
    "serving": "service",
}


def maintenance_budget_seconds(hard_ceiling_seconds: float) -> float:
    hard = _validate_time(hard_ceiling_seconds, "hard_ceiling_seconds")
    if hard <= 0.0:
        raise ValueError("hard_ceiling_seconds must be positive")
    return min(180.0, hard * 0.1)


def _positive_number(value: object, field: str) -> float:
    parsed = _validate_time(value, field)
    if parsed <= 0.0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _validate_gate_contract(value: Mapping) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("time gate contract must be a mapping")
    required = {"soft_target_seconds", "hard_ceiling_seconds", "minimum_effect"}
    if set(value) != required:
        raise ValueError("time gate contract fields are invalid")
    soft = _positive_number(value["soft_target_seconds"], "soft_target_seconds")
    hard = _positive_number(value["hard_ceiling_seconds"], "hard_ceiling_seconds")
    if soft > hard:
        raise ValueError("soft_target_seconds must not exceed hard_ceiling_seconds")
    thresholds = value["minimum_effect"]
    if not isinstance(thresholds, Mapping) or set(thresholds) != {
        "mechanism_us",
        "service_pct",
    }:
        raise ValueError("minimum_effect must define mechanism_us and service_pct")
    return {
        "soft_target_seconds": soft,
        "hard_ceiling_seconds": hard,
        "minimum_effect": {
            "mechanism_us": _positive_number(
                thresholds["mechanism_us"], "minimum_effect.mechanism_us"
            ),
            "service_pct": _positive_number(
                thresholds["service_pct"], "minimum_effect.service_pct"
            ),
        },
    }


def _validate_candidate(value: Mapping, contract: Mapping) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("candidate declaration must be a mapping")
    required = {
        "claim_layer",
        "cheapest_falsifier",
        "estimated_cost",
        "minimum_effect",
        "rejection_condition",
        "promotion_condition",
    }
    if not required.issubset(value):
        raise ValueError("candidate declaration is incomplete")
    claim = value["claim_layer"]
    if claim not in _CLAIM_LAST_STAGE:
        raise ValueError("claim_layer must be kernel, workload, or serving")
    if value["cheapest_falsifier"] not in _CANDIDATE_STAGES:
        raise ValueError("cheapest_falsifier is invalid")
    costs = value["estimated_cost"]
    if not isinstance(costs, Mapping) or set(costs) != set(_CANDIDATE_STAGES):
        raise ValueError("estimated_cost must cover every candidate stage")
    clean_costs = {
        stage: _positive_number(costs[stage], f"estimated_cost.{stage}")
        for stage in _CANDIDATE_STAGES
    }
    last_stage = _CLAIM_LAST_STAGE[claim]
    applicable = _CANDIDATE_STAGES[: _CANDIDATE_STAGES.index(last_stage) + 1]
    for earlier, later in zip(applicable, applicable[1:]):
        if clean_costs[later] < clean_costs[earlier]:
            raise ValueError(
                "estimated_cost must be nondecreasing in executable stage order"
            )
    cheapest = applicable[0]
    if value["cheapest_falsifier"] != cheapest:
        raise ValueError(
            "cheapest_falsifier must name the lowest-cost applicable stage "
            f"({cheapest})"
        )
    effect = value["minimum_effect"]
    if not isinstance(effect, Mapping) or set(effect) != {"metric", "value"}:
        raise ValueError("candidate minimum_effect fields are invalid")
    expected_metric = "mechanism_us" if claim == "kernel" else "service_pct"
    if effect["metric"] != expected_metric:
        raise ValueError("candidate minimum_effect metric does not match claim_layer")
    minimum = _positive_number(effect["value"], "candidate minimum_effect.value")
    if minimum < contract["minimum_effect"][expected_metric]:
        raise ValueError("candidate minimum_effect is below the project contract")
    for field in ("rejection_condition", "promotion_condition"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    return {
        **dict(value),
        "estimated_cost": clean_costs,
        "minimum_effect": {"metric": expected_metric, "value": minimum},
    }


def validate_candidate_declaration(value: Mapping, contract: Mapping) -> dict:
    """Validate the executable evidence declaration attached to a candidate."""
    return _validate_candidate(value, _validate_gate_contract(contract))


class CandidateGate:
    """Run the cheapest eligible evidence first and stop on a conclusive gate."""

    def __init__(
        self,
        contract: Mapping,
        candidate: Mapping,
        *,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.contract = _validate_gate_contract(contract)
        self.candidate = _validate_candidate(candidate, self.contract)
        if not callable(now):
            raise ValueError("now must be callable")
        self.now = now

    def _result(
        self,
        *,
        started_at: float,
        decision: str,
        stop_reason: str,
        completed: Sequence[str],
    ) -> dict:
        elapsed = max(0.0, float(self.now()) - started_at)
        last_stage = _CLAIM_LAST_STAGE[self.candidate["claim_layer"]]
        applicable = _CANDIDATE_STAGES[: _CANDIDATE_STAGES.index(last_stage) + 1]
        skipped = [stage for stage in applicable if stage not in completed]
        return {
            "decision": decision,
            "elapsed_seconds": float(elapsed),
            "stop_reason": stop_reason,
            "skipped_expensive_stages": skipped,
            "completed_stages": list(completed),
            "soft_target_exceeded": elapsed
            > self.contract["soft_target_seconds"],
        }

    def run(self, actions: Mapping[str, Callable[[], Mapping]]) -> dict:
        if not isinstance(actions, Mapping):
            raise ValueError("actions must be a mapping")
        started = float(self.now())
        completed: list[str] = []
        last_stage = _CLAIM_LAST_STAGE[self.candidate["claim_layer"]]
        applicable = _CANDIDATE_STAGES[: _CANDIDATE_STAGES.index(last_stage) + 1]
        threshold = self.candidate["minimum_effect"]["value"]
        for stage in applicable:
            action = actions.get(stage)
            if not callable(action):
                return self._result(
                    started_at=started,
                    decision="STOP",
                    stop_reason=f"missing_{stage}_action",
                    completed=completed,
                )
            elapsed = max(0.0, float(self.now()) - started)
            remaining = self.contract["hard_ceiling_seconds"] - elapsed
            if remaining <= 0.0 or self.candidate["estimated_cost"][stage] > remaining:
                return self._result(
                    started_at=started,
                    decision="STOP",
                    stop_reason="hard_ceiling_admission_failed",
                    completed=completed,
                )
            outcome = action()
            if not isinstance(outcome, Mapping):
                raise ValueError(f"{stage} result must be a mapping")
            completed.append(stage)
            status = outcome.get("status")
            if status not in {"passed", "not_applicable"}:
                reason = {
                    "static_review": "static_falsified",
                    "build_correctness": "correctness_failed",
                    "short_paired": "short_pair_failed",
                    "profiler": "profiler_failed",
                    "formal_paired": "formal_pair_failed",
                    "service": "service_failed",
                }[stage]
                return self._result(
                    started_at=started,
                    decision="STOP",
                    stop_reason=reason,
                    completed=completed,
                )
            if stage == "short_paired":
                upper = outcome.get("upper_bound")
                if isinstance(upper, bool) or not isinstance(upper, (int, float)):
                    return self._result(
                        started_at=started,
                        decision="STOP",
                        stop_reason="short_pair_missing_upper_bound",
                        completed=completed,
                    )
                if float(upper) < threshold:
                    return self._result(
                        started_at=started,
                        decision="STOP",
                        stop_reason="effect_upper_bound_below_minimum",
                        completed=completed,
                    )
            if stage in {"formal_paired", "service"}:
                lower = outcome.get("lower_bound")
                if isinstance(lower, bool) or not isinstance(lower, (int, float)):
                    return self._result(
                        started_at=started,
                        decision="STOP",
                        stop_reason=f"{stage}_missing_lower_bound",
                        completed=completed,
                    )
                if float(lower) < threshold:
                    return self._result(
                        started_at=started,
                        decision="STOP",
                        stop_reason="effect_not_confirmed",
                        completed=completed,
                    )
        return self._result(
            started_at=started,
            decision="PROMOTE",
            stop_reason="promotion_condition_satisfied",
            completed=completed,
        )


def _terminate_process_group(
    process: subprocess.Popen, *, grace_seconds: float
) -> tuple[str | None, str | None]:
    group = process.pid

    def group_exists() -> bool:
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not group_exists():
            break
        process.poll()
        time.sleep(0.01)
    if group_exists():
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return process.communicate()


def run_budgeted_command(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    grace_seconds: float = 0.2,
    input_value: str | None = None,
    check: bool = False,
    popen_options: Mapping | None = None,
    heartbeat_interval_seconds: float | None = None,
    event_sink: Callable[[Mapping], None] | None = None,
) -> subprocess.CompletedProcess:
    timeout = _positive_number(timeout_seconds, "timeout_seconds")
    grace = _positive_number(grace_seconds, "grace_seconds")
    if event_sink is not None and not callable(event_sink):
        raise ValueError("event_sink must be callable")
    heartbeat_interval = None
    if heartbeat_interval_seconds is not None:
        heartbeat_interval = _positive_number(
            heartbeat_interval_seconds, "heartbeat_interval_seconds"
        )
        if event_sink is None:
            raise ValueError("heartbeat_interval_seconds requires event_sink")
    options = dict(popen_options or {})
    options.setdefault("text", True)
    options.setdefault("stdout", subprocess.PIPE)
    options.setdefault("stderr", subprocess.PIPE)
    options["start_new_session"] = True
    process = subprocess.Popen(list(command), **options)
    timed_out = False
    started = time.monotonic()
    pending_input = input_value
    try:
        while True:
            elapsed = max(0.0, time.monotonic() - started)
            remaining = timeout - elapsed
            if remaining <= 0.0:
                timed_out = True
                stdout, stderr = _terminate_process_group(
                    process, grace_seconds=grace
                )
                break
            wait_seconds = remaining
            if heartbeat_interval is not None:
                wait_seconds = min(wait_seconds, heartbeat_interval)
            try:
                stdout, stderr = process.communicate(
                    input=pending_input, timeout=wait_seconds
                )
                break
            except subprocess.TimeoutExpired:
                pending_input = None
                elapsed = max(0.0, time.monotonic() - started)
                if elapsed >= timeout:
                    timed_out = True
                    stdout, stderr = _terminate_process_group(
                        process, grace_seconds=grace
                    )
                    break
                if event_sink is not None:
                    event_sink(
                        {
                            "event": "heartbeat",
                            "elapsed_seconds": float(elapsed),
                            "remaining_seconds": float(max(0.0, timeout - elapsed)),
                        }
                    )
    except BaseException:
        _terminate_process_group(process, grace_seconds=grace)
        raise
    elapsed = max(0.0, time.monotonic() - started)
    stop_reason = (
        "hard_deadline_exceeded"
        if timed_out
        else "completed"
        if process.returncode == 0
        else "command_failed"
    )
    result = subprocess.CompletedProcess(
        list(command), 124 if timed_out else process.returncode, stdout, stderr
    )
    result.timed_out = timed_out
    result.elapsed_seconds = float(elapsed)
    result.stop_reason = stop_reason
    if event_sink is not None:
        event_sink(
            {
                "event": "terminal",
                "elapsed_seconds": float(elapsed),
                "stop_reason": stop_reason,
                "returncode": result.returncode,
            }
        )
    if check and result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode,
            list(command),
            output=stdout,
            stderr=stderr,
        )
    return result


def run_maintenance_command(
    command: Sequence[str],
    *,
    hard_ceiling_seconds: float,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Run repair to its real deadline; the former 10% cap is advisory only."""
    hard = _positive_number(hard_ceiling_seconds, "hard_ceiling_seconds")
    soft = maintenance_budget_seconds(hard)
    result = run_budgeted_command(
        command,
        timeout_seconds=hard,
        **kwargs,
    )
    result.soft_limit_seconds = float(soft)
    result.soft_limit_exceeded = result.elapsed_seconds > soft
    return result
