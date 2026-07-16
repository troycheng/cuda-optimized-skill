#!/usr/bin/env python3
"""Render an answer-first, independently recomputable optimization summary."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import stat
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import quote


_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import decision as decision_engine  # noqa: E402


_TERMINAL_STATUSES = {
    "rejected_compile",
    "rejected_correctness",
    "rejected_constraint",
    "confirmed_loss",
    "inconclusive",
    "kernel_only_win",
    "end_to_end_win",
    "pareto_frontier",
}
_STATUS_ALIASES = {
    "confirmed_win": "kernel_only_win",
    "invalid": "inconclusive",
    "no_confirmed_kernel_win": "inconclusive",
    "workload_failed": "inconclusive",
}


def _reject_symlink_components(path, *, field: str, include_leaf: bool) -> Path:
    candidate = Path(path).expanduser().absolute()
    limit = candidate if include_leaf else candidate.parent
    current = Path(candidate.anchor)
    for part in limit.parts[1:]:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"{field} parent path must not contain a symlink: {current}")
    return candidate


def _read(path: str) -> dict:
    source = _reject_symlink_components(path, field="state", include_leaf=True)
    if source.is_symlink():
        raise ValueError(f"state path must not be a symlink: {source}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError(f"state path could not be opened safely: {source}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"state path must be a regular file: {source}")
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            payload = json.load(stream)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(payload, dict):
        raise ValueError("state must contain a JSON object")
    return payload


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_text(path: str, text: str) -> None:
    target = _reject_symlink_components(path, field="output", include_leaf=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(target, field="output", include_leaf=True)
    if target.is_symlink():
        raise ValueError(f"summary output must not be a symlink: {target}")
    if target.exists() and not target.is_file():
        raise ValueError(f"summary output must be a regular file: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        # A late symlink is safe to replace, but refusing it keeps the output
        # contract explicit and avoids silently changing caller-owned paths.
        if target.is_symlink():
            raise ValueError(f"summary output became a symlink: {target}")
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _mapping(value) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _list(value) -> list:
    return list(value) if isinstance(value, (list, tuple)) else []


def _safe_text(value, *, missing: str = "not recorded") -> str:
    """Return hostile values as one inert Markdown text line."""
    if value is None:
        return missing
    if isinstance(value, (dict, list, tuple)):
        try:
            value = json.dumps(
                value, sort_keys=True, ensure_ascii=False, allow_nan=False
            )
        except (TypeError, ValueError):
            return missing
    text = " ".join(str(value).split())
    if not text:
        return missing
    escaped = html.escape(text, quote=True)
    escaped = escaped.replace("\\", "\\\\")
    for marker in ("`", "!", "[", "]", "(", ")", "*", "#", "|"):
        escaped = escaped.replace(marker, "\\" + marker)
    return escaped


def _finite(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _status(value) -> str | None:
    if type(value) is not str:
        return None
    if value in _TERMINAL_STATUSES:
        return value
    return _STATUS_ALIASES.get(value)


def _latest_history(state: Mapping) -> Mapping:
    history = _list(state.get("history"))
    for item in reversed(history):
        if isinstance(item, Mapping):
            return item
    return {}


def _latest_iteration(state: Mapping) -> int | None:
    iterations = []
    for item in _list(state.get("history")):
        if not isinstance(item, Mapping):
            continue
        iteration = item.get("iter")
        if isinstance(iteration, int) and not isinstance(iteration, bool) and iteration > 0:
            iterations.append(iteration)
    return max(iterations) if iterations else None


def _latest_terminal_decision(state: Mapping) -> Mapping:
    decision = state.get("terminal_decision")
    if isinstance(decision, Mapping):
        return decision
    latest = _latest_history(state)
    if not latest:
        return {}
    # Legacy state remains readable, but cannot acquire stronger evidence than
    # it actually persisted.
    return {
        "legacy": True,
        "iteration": latest.get("iter"),
        "status": latest.get("status"),
        "mode": state.get("mode"),
        "statistics": latest.get("statistics"),
        "correctness": {"status": (
            "passed" if latest.get("validation_passed") is True else "not recorded"
        )},
        "decision_json": latest.get("decision_json"),
    }


def _statistics(state: Mapping, *, workload: bool) -> Mapping:
    decision = _latest_terminal_decision(state)
    decision_field = "workload_statistics" if workload else "statistics"
    nested = decision.get(decision_field)
    if isinstance(nested, Mapping):
        return nested
    return {}


def _validated_statistics(
    statistics: Mapping,
    field: str,
    *,
    expected_direction: str | None = None,
    expected_min_effect: float | None = None,
    required_status: str | None = None,
) -> Mapping | None:
    try:
        return decision_engine.validate_paired_statistics(
            statistics,
            field,
            expected_direction=expected_direction,
            expected_min_effect=expected_min_effect,
            required_status=required_status,
        )
    except ValueError:
        return None


def _valid_workload_snapshot(value) -> bool:
    if not isinstance(value, Mapping):
        return False
    if not isinstance(value.get("kind"), str) or not value["kind"].strip():
        return False
    if not isinstance(value.get("source_hash"), str) or not value[
        "source_hash"
    ].strip():
        return False
    objective = _mapping(value.get("objective"))
    primary = _mapping(objective.get("primary_metric"))
    return bool(
        isinstance(primary.get("name"), str)
        and primary["name"].strip()
        and isinstance(primary.get("direction"), str)
        and primary["direction"] in {"lower", "higher"}
    )


def _valid_paired_sample_metadata(
    decision: Mapping, kind: str, statistics: Mapping
) -> bool:
    evidence = _mapping(decision.get(f"{kind}_paired_samples"))
    required = {
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
    }
    if not required.issubset(evidence):
        return False
    valid = _integer(statistics.get("valid_pairs"))
    invalid = _integer(statistics.get("invalid_pairs"))
    pairs = _integer(evidence.get("pairs"))
    return bool(
        evidence.get("schema_version") == 2
        and evidence.get("kind") == kind
        and type(evidence.get("path")) is str
        and evidence["path"].strip()
        and type(evidence.get("sha256")) is str
        and len(evidence["sha256"]) == 64
        and pairs is not None
        and valid is not None
        and invalid is not None
        and pairs == valid + invalid
        and evidence.get("input_hash") == decision.get("input_hash")
        and evidence.get("iteration") == decision.get("iteration")
        and type(evidence.get("candidate_id")) is str
        and evidence["candidate_id"].strip()
        and evidence.get("candidate_id") == decision.get("candidate_id")
        and type(evidence.get("candidate_file")) is str
        and evidence["candidate_file"].strip()
        and evidence.get("candidate_sha256") == decision.get("candidate_sha256")
    )


def _terminal_result(state) -> tuple[str, list[str]]:
    """Resolve terminal status conservatively and return visible warnings."""
    if not isinstance(state, Mapping):
        return "inconclusive", ["state is not a mapping"]

    warnings = []
    decision = _latest_terminal_decision(state)
    claimed = _status(decision.get("status"))
    if claimed is None:
        return "inconclusive", ["latest terminal decision is missing or invalid"]
    if decision.get("legacy") is not True:
        if decision.get("input_hash") != state.get("input_hash"):
            return "inconclusive", ["terminal decision input binding is invalid"]
        iteration = decision.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration <= 0:
            return "inconclusive", ["terminal decision iteration binding is invalid"]
        if decision.get("mode") != state.get("mode"):
            return "inconclusive", ["terminal decision mode binding is invalid"]

    kernel_statistics = _statistics(state, workload=False)
    workload_statistics = _statistics(state, workload=True)
    state_min_effect = _finite(state.get("min_effect_pct"))
    kernel_evidence = _validated_statistics(
        kernel_statistics,
        "summary.kernel",
        expected_direction="lower",
        expected_min_effect=state_min_effect,
        required_status=(
            "confirmed_win"
            if claimed in {"kernel_only_win", "end_to_end_win"}
            else claimed if claimed in {"confirmed_loss", "inconclusive"} else None
        ),
    )

    if claimed == "end_to_end_win":
        mode = state.get("mode")
        workload = state.get("workload")
        if workload is None and kernel_evidence is not None:
            return "kernel_only_win", [
                "end_to_end_win was reduced because no user workload was supplied"
            ]
        if mode != "full":
            if kernel_evidence is not None:
                warnings.append(
                    "end_to_end_win was reduced because mode is not full"
                )
                return "kernel_only_win", warnings
            warnings.append("contradictory terminal evidence: mode is not full")
            return "inconclusive", warnings
        if not _valid_workload_snapshot(workload):
            warnings.append("contradictory terminal evidence: workload is malformed")
            return "inconclusive", warnings
        objective = _objective(state)
        primary = _mapping(objective.get("primary_metric"))
        workload_evidence = _validated_statistics(
            workload_statistics,
            "summary.workload",
            expected_direction=primary.get("direction"),
            expected_min_effect=_finite(objective.get("min_effect_pct")),
            required_status="confirmed_win",
        )
        if kernel_evidence is None or workload_evidence is None:
            warnings.append(
                "contradictory terminal evidence: end_to_end_win lacks confirmed "
                "kernel and workload statistics"
            )
            return "inconclusive", warnings
        if not _valid_paired_sample_metadata(
            decision, "kernel", kernel_evidence
        ) or not _valid_paired_sample_metadata(
            decision, "workload", workload_evidence
        ):
            return "inconclusive", [
                "terminal win lacks complete bound raw paired sample evidence"
            ]
        try:
            recomputed = decision_engine.decide(
                mode="full",
                kernel={"status": "confirmed_win", "statistics": kernel_evidence},
                workload={
                    "status": "evaluated",
                    "objective": dict(objective),
                    "primary": workload_evidence,
                    "constraints": list(decision.get("constraints") or []),
                },
            )
        except (TypeError, ValueError):
            return "inconclusive", ["terminal workload evidence is invalid"]
        if recomputed.get("status") != "end_to_end_win":
            return "inconclusive", ["terminal decision contradicts recomputed evidence"]

    if claimed == "kernel_only_win" and kernel_evidence is None:
        warnings.append(
            "contradictory terminal evidence: kernel_only_win lacks confirmed "
            "kernel statistics"
        )
        return "inconclusive", warnings
    if claimed == "kernel_only_win" and not _valid_paired_sample_metadata(
        decision, "kernel", kernel_evidence
    ):
        return "inconclusive", [
            "terminal win lacks complete bound raw paired sample evidence"
        ]
    if claimed in {"confirmed_loss", "inconclusive"} and kernel_evidence is None:
        return "inconclusive", ["terminal kernel statistics are invalid"]
    return claimed, warnings


def _fmt_number(value, suffix: str = "") -> str:
    number = _finite(value)
    return f"{number:.3f}{suffix}" if number is not None else "not recorded"


def _stats_lines(statistics: Mapping, *, prefix: str) -> list[str]:
    if not statistics:
        return [f"- {prefix} statistics: not recorded"]
    estimate = _finite(statistics.get("estimate_pct"))
    ci_low = _finite(statistics.get("ci_low_pct"))
    ci_high = _finite(statistics.get("ci_high_pct"))
    valid = _integer(statistics.get("valid_pairs"))
    invalid = _integer(statistics.get("invalid_pairs"))
    lines = [f"- {prefix} status: {_safe_text(statistics.get('status'))}"]
    lines.append(
        f"- estimate: {estimate:.3f}%" if estimate is not None else "- estimate: not recorded"
    )
    lines.append(
        f"- CI: [{ci_low:.3f}%, {ci_high:.3f}%]"
        if ci_low is not None and ci_high is not None
        else "- CI: not recorded"
    )
    lines.append(
        f"- pairs: {valid} valid / {invalid} invalid"
        if valid is not None and invalid is not None
        else "- pairs: not recorded"
    )
    lines.append(f"- statistic: {_safe_text(statistics.get('statistic'))}")
    lines.append(f"- confidence: {_fmt_number(statistics.get('confidence'))}")
    return lines


def _correctness(state: Mapping) -> str:
    explicit = _mapping(_latest_terminal_decision(state).get("correctness")).get(
        "status"
    )
    if type(explicit) is str:
        return _safe_text(explicit)
    validation = _latest_history(state).get("validation_passed")
    if type(validation) is bool:
        return "passed" if validation else "failed"
    return "not recorded"


def _sass(state: Mapping) -> str:
    terminal = _mapping(_latest_terminal_decision(state).get("sass"))
    if type(terminal.get("status")) is str:
        return _safe_text(terminal["status"])
    for field in ("sass_verification", "sass_check"):
        status = _mapping(state.get(field)).get("status")
        if type(status) is str:
            return _safe_text(status)
    coverage = state.get("sass_coverage")
    return _safe_text(coverage) if type(coverage) is str else "not recorded"


def _objective(state: Mapping) -> Mapping:
    return _mapping(_mapping(state.get("workload")).get("objective"))


def _coverage(state: Mapping) -> tuple[list[str], list[str]]:
    """Return coverage bullets and prominent degradation warnings."""
    lines = []
    warnings = []
    ncu = _mapping(_mapping(state.get("env")).get("ncu"))
    counter_error = ncu.get("counter_access_error")
    if type(counter_error) is str and counter_error.strip():
        profiler = f"degraded ({_safe_text(counter_error)})"
        warnings.append(f"profiler coverage degraded: {_safe_text(counter_error)}")
    elif ncu.get("can_read_counters") is True:
        profiler = "available"
    elif ncu.get("can_read_counters") is False:
        profiler = "unavailable"
        warnings.append("profiler coverage: unavailable")
    else:
        profiler = "not recorded"
    lines.append(f"- profiler coverage: {profiler}")

    terminal = _latest_terminal_decision(state)
    sanitizer_evidence = _mapping(terminal.get("sanitizer"))
    sanitizer = sanitizer_evidence.get("coverage")
    if type(sanitizer) is not str:
        sanitizer = "degraded" if state.get("sanitizer_coverage_degraded") is True else "not recorded"
    sanitizer_text = _safe_text(sanitizer)
    lines.append(f"- sanitizer coverage: {sanitizer_text}")
    if sanitizer in {"unavailable", "degraded"}:
        warnings.append(f"sanitizer coverage: {sanitizer_text}")

    compiler = _mapping(terminal.get("compiler_evidence"))
    compiler_status = compiler.get("status")
    compiler_text = (
        _safe_text(compiler_status)
        if type(compiler_status) is str
        else "not recorded"
    )
    lines.append(f"- compiler coverage: {compiler_text}")
    stages = compiler.get("stages")
    if isinstance(stages, (list, tuple)) and stages:
        clean_stages = ", ".join(_safe_text(stage) for stage in stages)
        lines.append(f"- compiler stages: {clean_stages}")
    elif isinstance(compiler.get("manifest"), Mapping):
        available = [
            name
            for name, evidence in compiler["manifest"].items()
            if isinstance(evidence, Mapping) and evidence.get("status") == "available"
        ]
        lines.append(
            "- compiler stages: "
            + (", ".join(_safe_text(name) for name in available) or "not recorded")
        )
    else:
        lines.append("- compiler stages: not recorded")
    return lines, warnings


def _artifact_link(label: str, path) -> str | None:
    if type(path) is not str or not path.strip():
        return None
    one_line = " ".join(path.split())
    encoded = quote(one_line, safe="/._-~")
    return f"[{_safe_text(label)}](<{encoded}>)"


def _join_artifact(run_dir, name: str) -> str | None:
    if type(run_dir) is not str or not run_dir.strip():
        return None
    return os.path.join(" ".join(run_dir.split()), name)


def _candidate_lines(state: Mapping) -> list[str]:
    lines = [f"- best: {_safe_text(state.get('best_file'))}"]
    lines.append(f"- best kernel time: {_fmt_number(state.get('best_metric_ms'), ' ms')}")

    frontier = [item for item in _list(state.get("frontier")) if isinstance(item, Mapping)]
    if frontier:
        lines.append(f"- frontier: {len(frontier)} candidate(s)")
        for item in frontier:
            candidate = item.get("candidate_file") or item.get("kernel") or item.get("id")
            lines.append(
                f"  - {_safe_text(candidate)}: {_safe_text(item.get('status'))}"
            )
    else:
        lines.append("- frontier: none recorded")

    rejected = [
        item
        for item in _list(state.get("rejected_candidates"))
        if isinstance(item, Mapping)
    ]
    inconclusive = [
        item
        for item in _list(state.get("inconclusive_candidates"))
        if isinstance(item, Mapping)
    ]
    candidates = state.get("candidates")
    if isinstance(candidates, Mapping):
        for candidate_id, item in candidates.items():
            if not isinstance(item, Mapping):
                continue
            entry = dict(item)
            entry.setdefault("id", candidate_id)
            status = entry.get("status")
            if type(status) is str and status.startswith("rejected_"):
                rejected.append(entry)
            elif status in {
                "inconclusive",
                "no_confirmed_kernel_win",
                "workload_failed",
            }:
                inconclusive.append(entry)
    for item in _list(state.get("history")):
        if not isinstance(item, Mapping):
            continue
        status = item.get("status")
        if type(status) is str and status.startswith("rejected_"):
            rejected.append(item)
        elif status in {"inconclusive", "no_confirmed_kernel_win", "workload_failed"}:
            inconclusive.append(item)

    def unique(items: list[Mapping]) -> list[Mapping]:
        result = []
        seen = set()
        for item in items:
            identity = (
                item.get("candidate_file") or item.get("id") or item.get("iter"),
                item.get("status"),
            )
            if identity not in seen:
                seen.add(identity)
                result.append(item)
        return result

    rejected = unique(rejected)
    inconclusive = unique(inconclusive)

    lines.append(f"- rejected: {len(rejected)}")
    for item in rejected:
        identity = item.get("candidate_file") or item.get("id") or item.get("iter")
        lines.append(f"  - {_safe_text(identity)}: {_safe_text(item.get('status'))}")
    lines.append(f"- inconclusive: {len(inconclusive)}")
    for item in inconclusive:
        identity = item.get("candidate_file") or item.get("id") or item.get("iter")
        lines.append(f"  - {_safe_text(identity)}: {_safe_text(item.get('status'))}")
    return lines


def render_text(state) -> str:
    """Purely render *state*; this function performs no filesystem I/O."""
    source = state if isinstance(state, Mapping) else {}
    result, terminal_warnings = _terminal_result(state)
    coverage_lines, coverage_warnings = _coverage(source)
    warnings = terminal_warnings + coverage_warnings
    budget = _mapping(source.get("budget"))

    lines = [f"# Result: {result}", ""]
    lines.append(
        f"- budget preset: {_safe_text(budget.get('name') or budget.get('preset'))}"
    )
    max_seconds = _finite(budget.get("max_seconds"))
    lines.append(
        f"- budget limit: {max_seconds:.3f} seconds"
        if max_seconds is not None
        else "- budget limit: not recorded"
    )
    lines.append(f"- budget rounds: {_safe_text(budget.get('max_rounds'))}")
    lines.append(f"- mode: {_safe_text(source.get('mode'))}")
    if warnings:
        lines.append("")
        for warning in warnings:
            lines.append(f"> **WARNING: {_safe_text(warning)}**")
    lines.append("")

    lines.extend(["## Frozen inputs and environment", ""])
    lines.append(f"- input hash: {_safe_text(source.get('input_hash'))}")
    lines.append(
        f"- baseline: {_safe_text(source.get('baseline_file_original') or source.get('baseline_file'))}"
    )
    lines.append(f"- reference: {_safe_text(source.get('ref_file'))}")
    lines.append(f"- backend: {_safe_text(source.get('backend'))}")
    lines.append(f"- dimensions: {_safe_text(source.get('dims'))}")
    workload = source.get("workload")
    workload_map = _mapping(workload)
    lines.append(f"- workload kind: {_safe_text(workload_map.get('kind'))}")
    lines.append(f"- workload source hash: {_safe_text(workload_map.get('source_hash'))}")
    env = _mapping(source.get("env"))
    gpu_list = _list(env.get("gpus"))
    gpu = _mapping(gpu_list[0]) if gpu_list else _mapping(env.get("gpu"))
    lines.append(f"- GPU: {_safe_text(gpu.get('name') or env.get('gpu'))}")
    lines.append(
        f"- architecture: {_safe_text(gpu.get('sm_arch') or gpu.get('compute_capability'))}"
    )
    lines.append(f"- nvcc: {_safe_text(_mapping(env.get('nvcc')).get('version'))}")
    lines.append("")

    lines.extend(["## Kernel evidence", ""])
    lines.extend(_stats_lines(_statistics(source, workload=False), prefix="kernel"))
    lines.append(f"- correctness: {_correctness(source)}")
    lines.append(f"- SASS: {_sass(source)}")
    lines.append("")

    lines.extend(["## Real workload evidence", ""])
    if workload is None:
        lines.append("- No user workload was supplied; no end-to-end win is claimed.")
        lines.append("- primary KPI: not recorded")
        lines.append("- workload statistics: not recorded")
    elif not _valid_workload_snapshot(workload):
        lines.append("- workload input is malformed; no end-to-end win is claimed.")
        lines.append("- primary KPI: not recorded")
        lines.append("- workload statistics: not recorded")
    else:
        objective = _objective(source)
        primary = _mapping(objective.get("primary_metric"))
        if primary.get("name") is None or primary.get("direction") is None:
            lines.append("- primary KPI: not recorded")
        else:
            lines.append(
                f"- primary KPI: {_safe_text(primary.get('name'))} "
                f"({_safe_text(primary.get('direction'))})"
            )
        lines.extend(
            _stats_lines(_statistics(source, workload=True), prefix="workload")
        )
        constraints = _list(objective.get("constraints"))
        if not constraints:
            lines.append("- constraints: none declared")
        for constraint in constraints:
            if not isinstance(constraint, Mapping):
                lines.append("- constraint: malformed")
                continue
            cap = _finite(constraint.get("max_regression_pct"))
            cap_text = f"{cap:.3f}%" if cap is not None else "not recorded"
            lines.append(
                f"- constraint: {_safe_text(constraint.get('name'))} "
                f"<= {cap_text} regression"
            )
        decision = _latest_terminal_decision(source)
        constraint_results = decision.get("constraints")
        if not isinstance(constraint_results, (list, tuple)):
            constraint_results = source.get("workload_constraint_results")
        for constraint in _list(constraint_results):
            if not isinstance(constraint, Mapping):
                lines.append("- constraint result: malformed")
                continue
            estimate = _finite(constraint.get("estimate_pct"))
            ci_low = _finite(constraint.get("ci_low_pct"))
            ci_high = _finite(constraint.get("ci_high_pct"))
            estimate_text = (
                f"{estimate:.3f}%" if estimate is not None else "not recorded"
            )
            ci_text = (
                f"[{ci_low:.3f}%, {ci_high:.3f}%]"
                if ci_low is not None and ci_high is not None
                else "not recorded"
            )
            lines.append(
                f"- constraint result: {_safe_text(constraint.get('name'))}: "
                f"{_safe_text(constraint.get('status'))}; estimate "
                f"{estimate_text}, CI {ci_text}"
            )
    lines.append("")

    lines.extend(["## Evidence coverage", ""])
    lines.extend(coverage_lines)
    lines.append("")

    lines.extend(["## Candidate outcomes", ""])
    lines.extend(_candidate_lines(source))
    lines.append("")

    lines.extend(["## Historical best evidence", ""])
    historical_kernel = _mapping(source.get("best_kernel_statistics"))
    historical_workload = _mapping(source.get("best_workload_statistics"))
    lines.extend(_stats_lines(historical_kernel, prefix="historical best kernel"))
    lines.extend(
        _stats_lines(historical_workload, prefix="historical best workload")
    )
    lines.append("")

    lines.extend(["## Raw artifacts and resume", ""])
    terminal = _latest_terminal_decision(source)
    run_dir = source.get("run_dir")
    kernel_samples = _mapping(terminal.get("kernel_paired_samples")).get("path")
    workload_samples = _mapping(terminal.get("workload_paired_samples")).get(
        "path"
    )
    artifact_entries = [
        (
            "state.json",
            _join_artifact(run_dir, "state.json"),
        ),
        (
            "manifest.json",
            _join_artifact(run_dir, "manifest.json"),
        ),
        (
            "kernel paired_samples.jsonl",
            kernel_samples,
        ),
        (
            "workload paired_samples.jsonl",
            workload_samples,
        ),
        (
            "decision.json",
            terminal.get("decision_json"),
        ),
        (
            "compiler manifest",
            _mapping(terminal.get("compiler_evidence")).get("manifest_path"),
        ),
    ]
    for label, path in artifact_entries:
        link = _artifact_link(label, path)
        if link is None:
            lines.append(f"- {label}: not recorded")
        else:
            lines.append(f"- {link}")

    resume = _mapping(terminal.get("resume"))
    lines.append(f"- resume status: {_safe_text(resume.get('status'))}")
    checkpoint = resume.get("checkpoint")
    checkpoint_link = _artifact_link("checkpoint.json", checkpoint)
    lines.append(
        f"- {checkpoint_link}"
        if checkpoint_link is not None
        else "- checkpoint.json: not recorded"
    )
    lines.append("")
    return "\n".join(lines)


def render(state_path: str, out_path: str) -> None:
    state = _read(state_path)
    text = render_text(state)
    _atomic_write_text(out_path, text)
    print(json.dumps({"summary": out_path}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    render(args.state, args.out)


if __name__ == "__main__":
    main()
