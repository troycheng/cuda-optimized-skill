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
  "noise_threshold_pct": float,
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
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from numbers import Real
from pathlib import Path


# Keep sibling imports working both as a CLI script and via importlib file specs.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from artifact_store import (  # noqa: E402
    ArtifactStore,
    CURRENT_SCHEMA_VERSION,
    atomic_write_json,
)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _read(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
_EVIDENCE_STATUSES = {"confirmed_win", "confirmed_loss", "inconclusive", "invalid"}
_STATISTIC_FIELDS = (
    "statistic",
    "estimate_pct",
    "ci_low_pct",
    "ci_high_pct",
    "status",
)


def _resolved_path(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("decision candidate_file and kernel must be non-empty paths")
    return str(Path(path).expanduser().resolve(strict=False))


def _validate_decision_statistics(
    payload, *, required: bool, field_name: str = "statistics"
) -> dict | None:
    if payload is None and not required:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError(f"decision.json {field_name} must be a JSON object")
    missing = [field for field in _STATISTIC_FIELDS if field not in payload]
    if missing:
        raise ValueError(
            f"decision.json {field_name} missing required field: {missing[0]}"
        )
    statistic = payload["statistic"]
    if not isinstance(statistic, str) or not statistic.strip():
        raise ValueError(
            f"decision.json {field_name}.statistic must be a string"
        )
    status = payload["status"]
    if type(status) is not str or status not in _EVIDENCE_STATUSES:
        raise ValueError(
            f"decision.json {field_name}.status must be a known string"
        )
    clean = dict(payload)
    for field in ("estimate_pct", "ci_low_pct", "ci_high_pct"):
        value = payload[field]
        if value is None and not required:
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                f"decision.json {field_name}.{field} must be finite"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(
                f"decision.json {field_name}.{field} must be finite"
            )
        clean[field] = numeric
    return clean


def _load_decision(
    path: str, *, candidate_file: str
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
    if _resolved_path(declared_candidate) != _resolved_path(candidate_file):
        raise ValueError(
            "decision.json candidate_file does not match the update kernel"
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

    env = {}
    if args.env and os.path.isfile(args.env):
        env = _read(args.env)

    try:
        dims = json.loads(args.dims) if args.dims else {}
    except json.JSONDecodeError as e:
        sys.exit(f"--dims must be valid JSON: {e}")

    budget = {
        "iterations_total": int(args.iterations),
        "ncu_num": int(args.ncu_num),
        "branches": int(args.branches),
    }
    store = ArtifactStore(run_dir)
    manifest = store.initialize(
        inputs={"baseline": baseline, "ref": ref},
        budget=budget,
        environment=env,
    )

    baseline_copy_dir = os.path.join(run_dir, "baseline")
    baseline_copy = os.path.join(baseline_copy_dir, os.path.basename(baseline))
    shutil.copy2(baseline, baseline_copy)

    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "run_dir": run_dir,
        "input_hash": manifest["input_hash"],
        "budget": budget,
        "mode": "kernel-only",
        "workload": None,
        "baseline_file": baseline_copy,
        "baseline_file_original": baseline,
        "ref_file": ref,
        "best_file": baseline_copy,
        "best_kernel_statistics": None,
        "best_workload_statistics": None,
        "best_metric_ms": None,
        "best_ncu_rep": None,
        "env": env,
        "env_path": os.path.abspath(args.env) if args.env and os.path.isfile(args.env) else None,
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
        "candidates": {},
        "history": [],
        "roofline_history": [],
        "frontier": [],
        "created_at": ts,
    }
    state_path = os.path.join(run_dir, "state.json")
    _write(state_path, state)

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
    state = _read_state(args.state)
    bench = _read(args.bench)
    methods = _read(args.methods_json)

    iter_dir = os.path.join(state["run_dir"], f"iterv{args.iter}")
    decision_path = getattr(args, "decision", None) or os.path.join(
        iter_dir, "decision.json"
    )
    (
        decision,
        decision_status,
        decision_statistics,
        workload_statistics,
    ) = _load_decision(
        decision_path, candidate_file=args.kernel
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
    if args.sass_check and os.path.isfile(args.sass_check):
        sass = _read(args.sass_check)
        for c in sass.get("checks", []):
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

        sass_verified = sass_info.get("verified", True)  # Default True if no check
        contributed = attr_info.get("contributed", None)
        attr_ms = attr_info.get("attribution_ms", None)

        m_entry = dict(m)

        if not sass_verified:
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

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
