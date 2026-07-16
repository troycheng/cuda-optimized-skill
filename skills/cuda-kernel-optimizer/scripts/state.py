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
  "best_profiled_file": str | null,
  "best_profiled_metric_ms": float | null,
  "best_profiled_ncu_rep": str | null,
  "env": {...},
  "iterations_total": int,
  "ncu_num": int,
  "branches": int,
  "noise_threshold_pct": float,
  "ptr_size": int,
  "dims": dict,
  "selected_methods":   [ {id, name, axis, iter} ],
  "effective_methods":  [ {id, name, axis, iter, attribution_ms} ],
  "ineffective_methods":[ {id, name, axis, iter} ],
  "implementation_failed_methods": [ {id, name, axis, iter, note} ],
  "unverified_methods": [ {id, name, axis, iter, note} ],
  "strategy_memory": {path, scope_key, constraints},
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
import shutil
import subprocess
import sys
from pathlib import Path

from common_options import BENCHMARK_DEFAULTS, NCU_DEFAULTS
from strategy_memory import DEFAULT_PATH as DEFAULT_STRATEGY_MEMORY
from strategy_memory import load_constraints, record as record_strategy, scope_key


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _read(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _sha256(path: str) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kernel_metric(bench: dict) -> tuple[float | None, str]:
    kernel = bench.get("kernel") or {}
    if kernel.get("median_ms") is not None:
        return float(kernel["median_ms"]), "median_ms"
    if kernel.get("average_ms") is not None:
        return float(kernel["average_ms"]), "average_ms"
    return None, "unavailable"


def _update_manifest(state: dict) -> None:
    path = os.path.join(state["run_dir"], "run_manifest.json")
    manifest = _read(path) if os.path.isfile(path) else {}
    manifest["winner_state"] = {
        "benchmark_winner": {
            "file": state.get("best_file"),
            "metric_ms": state.get("best_metric_ms"),
            "metric_name": state.get("metric_name"),
        },
        "fully_profiled_winner": {
            "file": state.get("best_profiled_file"),
            "metric_ms": state.get("best_profiled_metric_ms"),
            "ncu_rep": state.get("best_profiled_ncu_rep"),
        },
    }
    manifest["history_count"] = len(state.get("history", []))
    _write(path, manifest)


def _write_preflight_markdown(preflight_json: str, markdown_path: str) -> None:
    """Render a compact, durable human-readable companion to preflight.json."""
    report = _read(preflight_json)
    baseline = report.get("baseline") or {}
    reference = report.get("ref") or {}
    lines = [
        "# CUDA Kernel Optimizer Preflight",
        "",
        f"- Status: `{'passed' if report.get('ok') else 'failed'}`",
        f"- Baseline: `{baseline.get('path', 'unknown')}`",
        f"- Backend: `{baseline.get('backend', 'unknown')}`",
        f"- Reference: `{reference.get('path', 'unknown')}`",
        "",
    ]
    for title, key in (("Errors", "errors"), ("Warnings", "warnings")):
        items = report.get(key) or []
        lines += [f"## {title}", ""]
        lines += [f"- {item}" for item in items] if items else ["- None"]
        lines.append("")
    Path(markdown_path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> None:
    baseline = os.path.abspath(args.baseline)
    ref = os.path.abspath(args.ref)
    if not os.path.isfile(baseline):
        sys.exit(f"baseline not found: {baseline}")
    if not os.path.isfile(ref):
        sys.exit(f"ref not found: {ref}")

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.dirname(baseline), f"run_{ts}")
    os.makedirs(run_dir, exist_ok=False)

    baseline_copy_dir = os.path.join(run_dir, "baseline")
    os.makedirs(baseline_copy_dir, exist_ok=True)
    baseline_copy = os.path.join(baseline_copy_dir, os.path.basename(baseline))
    shutil.copy2(baseline, baseline_copy)

    env = {}
    if args.env and os.path.isfile(args.env):
        env = _read(args.env)

    try:
        dims = json.loads(args.dims) if args.dims else {}
    except json.JSONDecodeError as e:
        sys.exit(f"--dims must be valid JSON: {e}")

    try:
        benchmark_options = dict(BENCHMARK_DEFAULTS)
        benchmark_options.update(json.loads(args.benchmark_options or "{}"))
        ncu_options = dict(NCU_DEFAULTS)
        ncu_options.update(json.loads(args.ncu_options or "{}"))
    except json.JSONDecodeError as e:
        sys.exit(f"benchmark/ncu options must be valid JSON: {e}")

    env_snapshot = os.path.join(run_dir, "env.json")
    if args.env and os.path.isfile(args.env):
        shutil.copy2(args.env, env_snapshot)
    preflight_snapshot = os.path.join(run_dir, "preflight.json")
    if args.preflight and os.path.isfile(args.preflight):
        shutil.copy2(args.preflight, preflight_snapshot)
    preflight_markdown = os.path.join(run_dir, "preflight.md")
    if os.path.isfile(preflight_snapshot):
        _write_preflight_markdown(preflight_snapshot, preflight_markdown)

    strategy_path = os.path.abspath(os.path.expanduser(args.strategy_memory))
    workload_scope = scope_key(
        benchmark_options.get("backend", "auto"),
        baseline,
        ref,
        dims,
        benchmark_options.get("arch", ""),
    )
    prior_constraints = load_constraints(strategy_path, workload_scope)

    state = {
        "run_dir": run_dir,
        "baseline_file": baseline_copy,
        "baseline_file_original": baseline,
        "ref_file": ref,
        "best_file": baseline_copy,
        "best_metric_ms": None,
        "metric_name": "median_ms",
        "best_ncu_rep": None,
        "best_profiled_file": None,
        "best_profiled_metric_ms": None,
        "best_profiled_ncu_rep": None,
        "env": env,
        "env_path": env_snapshot if os.path.isfile(env_snapshot) else None,
        "preflight_path": preflight_snapshot if os.path.isfile(preflight_snapshot) else None,
        "preflight_markdown_path": (
            preflight_markdown if os.path.isfile(preflight_markdown) else None
        ),
        "benchmark_options": benchmark_options,
        "ncu_options": ncu_options,
        "iterations_total": int(args.iterations),
        "ncu_num": int(args.ncu_num),
        "branches": int(args.branches),
        "noise_threshold_pct": float(args.noise_threshold_pct),
        "ptr_size": int(args.ptr_size),
        "dims": dims,
        "selected_methods": [],
        "effective_methods": [],
        "ineffective_methods": [],
        "implementation_failed_methods": [],
        "unverified_methods": [],
        "history": [],
        "roofline_history": [],
        "frontier": [],
        "created_at": ts,
        "strategy_memory": {
            "path": strategy_path,
            "scope_key": workload_scope,
            "constraints": prior_constraints,
        },
    }
    state_path = os.path.join(run_dir, "state.json")
    _write(state_path, state)

    _write(os.path.join(run_dir, "run_manifest.json"), {
        "schema_version": 2,
        "created_at": ts,
        "run_dir": run_dir,
        "artifacts": {
            "baseline_original": baseline,
            "baseline_copy": baseline_copy,
            "baseline_sha256": _sha256(baseline),
            "reference": ref,
            "reference_sha256": _sha256(ref),
            "env": state.get("env_path"),
            "preflight": state.get("preflight_path"),
            "preflight_markdown": state.get("preflight_markdown_path"),
        },
        "dimensions": dims,
        "benchmark_options": benchmark_options,
        "ncu_options": ncu_options,
        "strategy_memory": state["strategy_memory"],
    })
    _update_manifest(state)

    for i in range(1, state["iterations_total"] + 1):
        os.makedirs(os.path.join(run_dir, f"iterv{i}"), exist_ok=True)

    print(json.dumps({"run_dir": run_dir, "state": state_path}, indent=2))


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
    state = _read(args.state)
    bench = _read(args.bench)
    methods = _read(args.methods_json)

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
    new_ms, metric_name = _kernel_metric(bench)
    ref_ms = None
    if bench.get("reference"):
        ref_ms = bench["reference"].get("median_ms")
        if ref_ms is None:
            ref_ms = bench["reference"].get("average_ms")

    # Load attribution and sass_check if provided
    attribution_data = {}
    if args.attribution and os.path.isfile(args.attribution):
        attr = _read(args.attribution)
        for a in attr.get("attributions", []):
            attribution_data[a["method_id"]] = a

    sass_data = {}
    sass_overall_status = "unknown"
    if args.sass_check and os.path.isfile(args.sass_check):
        sass = _read(args.sass_check)
        sass_overall_status = sass.get("status", "unknown")
        for c in sass.get("checks", []):
            sass_data[c["method_id"]] = c

    profile_data = {}
    if args.ncu_status and os.path.isfile(args.ncu_status):
        profile_data = _read(args.ncu_status)
    profile_rep = profile_data.get("ncu_rep")
    profile_ok = bool(
        profile_data.get("profile_status") == "success"
        and profile_rep
        and os.path.isfile(profile_rep)
    )

    # Decide improvement
    best_before = state.get("best_metric_ms")
    threshold = 1.0 - (state.get("noise_threshold_pct", 2.0) / 100.0)
    improved = False
    speedup_vs_best_before = None
    if validation_passed and new_ms and new_ms > 0:
        if best_before is None:
            improved = True
        else:
            speedup_vs_best_before = best_before / new_ms
            improved = new_ms < best_before * threshold

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

        verification_status = sass_info.get("verification_status") or (
            "verified" if sass_info.get("verified") is True
            else "failed" if sass_info.get("verified") is False
            else "unknown"
        )
        contributed = attr_info.get("contributed", None)
        attr_ms = attr_info.get("attribution_ms", None)

        m_entry = dict(m)

        if verification_status == "failed":
            # SASS signature missing — implementation failed
            m_entry["note"] = f"SASS patterns not found: {sass_info.get('patterns_missing', [])}"
            _merge_unique(state["implementation_failed_methods"], [m_entry])
        elif contributed is False:
            # Attribution says it didn't help
            m_entry["note"] = f"attribution_ms={attr_ms}"
            _merge_unique(state["ineffective_methods"], [m_entry])
        elif contributed is True and verification_status == "verified":
            if attr_ms is not None:
                m_entry["attribution_ms"] = attr_ms
            if speedup_vs_best_before is not None:
                m_entry["speedup_vs_best_before"] = speedup_vs_best_before
            _merge_unique(state["effective_methods"], [m_entry])
        else:
            m_entry["note"] = (
                f"verification_status={verification_status}; "
                f"attribution={'present' if contributed is not None else 'missing'}"
            )
            _merge_unique(state["unverified_methods"], [m_entry])

    # Update best
    if validation_passed and improved:
        state["best_file"] = os.path.abspath(args.kernel)
        state["best_metric_ms"] = new_ms
        state["metric_name"] = metric_name

    profiled_before = state.get("best_profiled_metric_ms")
    profiled_improved = False
    if validation_passed and profile_ok and new_ms and new_ms > 0:
        if profiled_before is None:
            profiled_improved = True
        else:
            profiled_improved = new_ms < float(profiled_before) * threshold
        if profiled_improved:
            state["best_profiled_file"] = os.path.abspath(args.kernel)
            state["best_profiled_metric_ms"] = new_ms
            state["best_profiled_ncu_rep"] = os.path.abspath(profile_rep)
            state["best_ncu_rep"] = os.path.abspath(profile_rep)

    # Load roofline data if available
    iter_dir = os.path.join(state["run_dir"], f"iterv{args.iter}")
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

    status = (
        "improved" if (validation_passed and improved)
        else "regressed" if validation_passed
        else "failed_validation"
    )
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
        "metric_name": metric_name,
        "profile_status": profile_data.get("profile_status", "missing"),
        "profile_analysis_status": profile_data.get("analysis_status", "missing"),
        "ncu_rep": os.path.abspath(profile_rep) if profile_rep else None,
        "sass_status": sass_overall_status,
        "became_benchmark_winner": bool(validation_passed and improved),
        "became_fully_profiled_winner": profiled_improved,
        "status": status,
        "retries": int(args.retries),
    })

    strategy = state.get("strategy_memory") or {}
    strategy_path = strategy.get("path")
    strategy_scope = strategy.get("scope_key")
    if strategy_path and strategy_scope:
        method_outcomes = []
        effective_ids = {_method_key(m) for m in state.get("effective_methods", [])}
        ineffective_ids = {_method_key(m) for m in state.get("ineffective_methods", [])}
        failed_ids = {_method_key(m) for m in state.get("implementation_failed_methods", [])}
        for m in methods_list:
            mid = _method_key(m)
            if mid in effective_ids:
                outcome, reason = "positive", "attribution_and_sass_verified"
            elif mid in ineffective_ids:
                outcome, reason = "negative", "ablation_not_helpful"
            elif mid in failed_ids:
                outcome, reason = "rejected", "sass_verification_failed"
            else:
                continue
            method_outcomes.append({
                "method_id": mid,
                "outcome": outcome,
                "reason": reason,
                "evidence": {"iter": int(args.iter), "metric_ms": new_ms},
            })
        bundle_outcome = {
            "method_ids": [m["id"] for m in methods_list],
            "outcome": (
                "rejected" if not validation_passed
                else "positive" if improved
                else "negative"
            ),
            "evidence": {
                "iter": int(args.iter),
                "metric_ms": new_ms,
                "profile_status": profile_data.get("profile_status", "missing"),
            },
        }
        constraints = record_strategy(
            strategy_path,
            strategy_scope,
            {
                "backend": (state.get("benchmark_options") or {}).get("backend"),
                "baseline": state.get("baseline_file_original"),
                "reference": state.get("ref_file"),
                "dims": state.get("dims"),
                "arch": (state.get("benchmark_options") or {}).get("arch"),
            },
            method_outcomes,
            bundle_outcome,
        )
        state["strategy_memory"]["constraints"] = constraints

    _write(args.state, state)
    _update_manifest(state)
    print(json.dumps({
        "iter": args.iter,
        "status": status,
        "new_ms": new_ms,
        "best_ms": state["best_metric_ms"],
        "best_profiled_ms": state.get("best_profiled_metric_ms"),
        "improved": improved,
        "profiled_improved": profiled_improved,
        "speedup_vs_best_before": speedup_vs_best_before,
    }, indent=2))


# ---------------------------------------------------------------------------
# set-best-ncu-rep
# ---------------------------------------------------------------------------

def cmd_set_best_ncu(args: argparse.Namespace) -> None:
    state = _read(args.state)
    state["best_ncu_rep"] = os.path.abspath(args.ncu_rep)
    _write(args.state, state)
    print(json.dumps({"best_ncu_rep": state["best_ncu_rep"]}, indent=2))


# ---------------------------------------------------------------------------
# seed baseline metric
# ---------------------------------------------------------------------------

def cmd_set_baseline_metric(args: argparse.Namespace) -> None:
    state = _read(args.state)
    bench = _read(args.bench)
    if not bench.get("correctness", {}).get("passed", True):
        sys.exit("Baseline failed correctness validation — cannot proceed.")
    ms, metric_name = _kernel_metric(bench)
    if ms is None:
        sys.exit("Baseline bench has no kernel timing.")
    state["best_metric_ms"] = ms
    state["metric_name"] = metric_name
    _write(args.state, state)
    _update_manifest(state)
    print(json.dumps({"baseline_ms": ms}, indent=2))


def cmd_show(args: argparse.Namespace) -> None:
    state = _read(args.state)
    print(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
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
    pi.add_argument("--preflight", type=str, default="")
    pi.add_argument("--benchmark-options", type=str, default="{}")
    pi.add_argument("--ncu-options", type=str, default="{}")
    pi.add_argument("--strategy-memory", type=str, default=DEFAULT_STRATEGY_MEMORY)
    pi.add_argument("--noise-threshold-pct", type=float, default=2.0)
    pi.add_argument("--ptr-size", type=int, default=0)
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
    pu.add_argument("--ncu-status", type=str, default=None,
                    help="Path to kernel_profile_status.json from full NCU profiling")
    pu.add_argument("--retries", type=int, default=0)
    pu.add_argument("--skip-validation", action="store_true")
    pu.add_argument("--allow-ineffective", action="store_true")
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

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
