#!/usr/bin/env python3
"""Render an answer-first, independently recomputable optimization summary."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import quote


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


def _read(path: str) -> dict:
    source = Path(path).expanduser()
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
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
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
    # HTML is escaped and backticks cannot break into a Markdown code span.
    return html.escape(text, quote=True).replace("`", "&#96;")


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


def _statistics(state: Mapping, *, workload: bool) -> Mapping:
    field = "best_workload_statistics" if workload else "best_kernel_statistics"
    direct = state.get(field)
    if isinstance(direct, Mapping):
        return direct
    decision = _mapping(state.get("terminal_decision") or state.get("decision"))
    decision_field = "workload_statistics" if workload else "statistics"
    nested = decision.get(decision_field)
    if isinstance(nested, Mapping):
        return nested
    if not workload:
        latest = _latest_history(state).get("statistics")
        if isinstance(latest, Mapping):
            return latest
    return {}


def _statistics_confirmed(statistics: Mapping) -> bool:
    if statistics.get("status") != "confirmed_win":
        return False
    if not isinstance(statistics.get("statistic"), str) or not statistics[
        "statistic"
    ].strip():
        return False
    estimate = _finite(statistics.get("estimate_pct"))
    ci_low = _finite(statistics.get("ci_low_pct"))
    ci_high = _finite(statistics.get("ci_high_pct"))
    confidence = _finite(statistics.get("confidence"))
    valid_pairs = _integer(statistics.get("valid_pairs"))
    invalid_pairs = _integer(statistics.get("invalid_pairs"))
    return bool(
        estimate is not None
        and ci_low is not None
        and ci_high is not None
        and ci_low <= ci_high
        and confidence is not None
        and 0.0 < confidence < 1.0
        and valid_pairs is not None
        and valid_pairs > 0
        and invalid_pairs is not None
    )


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


def _terminal_result(state) -> tuple[str, list[str]]:
    """Resolve terminal status conservatively and return visible warnings."""
    if not isinstance(state, Mapping):
        return "inconclusive", ["state is not a mapping"]

    warnings = []
    sources = []
    invalid_explicit = False
    for field in ("terminal_result", "result"):
        if field not in state:
            continue
        normalized = _status(state.get(field))
        if normalized is None:
            invalid_explicit = True
            warnings.append(f"{field} has an invalid type or value")
        else:
            sources.append((field, normalized))

    top_status = _status(state.get("status"))
    if top_status is not None:
        sources.append(("status", top_status))
    latest_status = _status(_latest_history(state).get("status"))
    if latest_status is not None:
        sources.append(("history", latest_status))
    decision_status = _status(
        _mapping(state.get("terminal_decision") or state.get("decision")).get(
            "status"
        )
    )
    if decision_status is not None:
        sources.append(("decision", decision_status))

    distinct = {value for _, value in sources}
    if invalid_explicit or len(distinct) > 1:
        if len(distinct) > 1:
            detail = ", ".join(f"{name}={value}" for name, value in sources)
            warnings.append(f"terminal sources disagree ({detail})")
        return "inconclusive", warnings

    kernel_statistics = _statistics(state, workload=False)
    workload_statistics = _statistics(state, workload=True)
    claimed = next(iter(distinct), None)
    if claimed is None:
        if _statistics_confirmed(workload_statistics):
            claimed = "end_to_end_win"
        elif _statistics_confirmed(kernel_statistics):
            claimed = "kernel_only_win"
        else:
            return "inconclusive", warnings

    if claimed == "end_to_end_win":
        mode = state.get("mode")
        workload = state.get("workload")
        if workload is None and _statistics_confirmed(kernel_statistics):
            warnings.append(
                "end_to_end_win was reduced because no user workload was supplied"
            )
            return "kernel_only_win", warnings
        if mode != "full":
            if _statistics_confirmed(kernel_statistics):
                warnings.append(
                    "end_to_end_win was reduced because mode is not full"
                )
                return "kernel_only_win", warnings
            warnings.append("contradictory terminal evidence: mode is not full")
            return "inconclusive", warnings
        if not _valid_workload_snapshot(workload):
            warnings.append("contradictory terminal evidence: workload is malformed")
            return "inconclusive", warnings
        if not (
            _statistics_confirmed(kernel_statistics)
            and _statistics_confirmed(workload_statistics)
        ):
            warnings.append(
                "contradictory terminal evidence: end_to_end_win lacks confirmed "
                "kernel and workload statistics"
            )
            return "inconclusive", warnings

    if claimed == "kernel_only_win" and not _statistics_confirmed(kernel_statistics):
        warnings.append(
            "contradictory terminal evidence: kernel_only_win lacks confirmed "
            "kernel statistics"
        )
        return "inconclusive", warnings
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
    explicit = _mapping(state.get("correctness")).get("status")
    if type(explicit) is str:
        return _safe_text(explicit)
    validation = _latest_history(state).get("validation_passed")
    if type(validation) is bool:
        return "passed" if validation else "failed"
    return "not recorded"


def _sass(state: Mapping) -> str:
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

    sanitizer = state.get("sanitizer_coverage")
    if type(sanitizer) is not str:
        sanitizer = "degraded" if state.get("sanitizer_coverage_degraded") is True else "not recorded"
    sanitizer_text = _safe_text(sanitizer)
    lines.append(f"- sanitizer coverage: {sanitizer_text}")
    if sanitizer in {"unavailable", "degraded"}:
        warnings.append(f"sanitizer coverage: {sanitizer_text}")

    compiler = _mapping(state.get("compiler_evidence"))
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
    lines.append(f"- budget preset: {_safe_text(budget.get('preset'))}")
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
        decision = _mapping(
            source.get("terminal_decision") or source.get("decision")
        )
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

    lines.extend(["## Raw artifacts and resume", ""])
    raw = _mapping(source.get("raw_artifacts"))
    run_dir = source.get("run_dir")
    latest_iteration = _latest_iteration(source)
    conventional_kernel_samples = None
    conventional_workload_samples = None
    if latest_iteration is not None:
        iteration_dir = _join_artifact(run_dir, f"iterv{latest_iteration}")
        conventional_kernel_samples = _join_artifact(
            iteration_dir, "paired_samples.jsonl"
        )
        if _valid_workload_snapshot(workload):
            conventional_workload_samples = _join_artifact(
                iteration_dir, os.path.join("workload", "paired_samples.jsonl")
            )
    kernel_samples = raw.get("kernel_paired_samples") or raw.get("paired_samples")
    workload_samples = raw.get("workload_paired_samples")
    artifact_entries = [
        (
            "state.json",
            raw.get("state") or _join_artifact(run_dir, "state.json"),
            False,
        ),
        (
            "manifest.json",
            raw.get("manifest") or _join_artifact(run_dir, "manifest.json"),
            False,
        ),
        (
            "kernel paired_samples.jsonl",
            kernel_samples or conventional_kernel_samples,
            kernel_samples is None and conventional_kernel_samples is not None,
        ),
        (
            "workload paired_samples.jsonl",
            workload_samples or conventional_workload_samples,
            workload_samples is None and conventional_workload_samples is not None,
        ),
        (
            "decision.json",
            raw.get("decision") or _latest_history(source).get("decision_json"),
            False,
        ),
        (
            "compiler manifest",
            raw.get("compiler_manifest")
            or _mapping(source.get("compiler_evidence")).get("manifest_path"),
            False,
        ),
    ]
    for label, path, conventional in artifact_entries:
        link = _artifact_link(label, path)
        if link is None:
            lines.append(f"- {label}: not recorded")
        elif conventional:
            lines.append(
                f"- {link} — conventional location; existence not verified"
            )
        else:
            lines.append(f"- {link}")

    resume = _mapping(source.get("resume"))
    lines.append(f"- resume status: {_safe_text(resume.get('status'))}")
    checkpoint = resume.get("checkpoint") or _join_artifact(run_dir, "checkpoint.json")
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
