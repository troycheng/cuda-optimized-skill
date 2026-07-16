#!/usr/bin/env python3
"""Run benchmark.py for a given iteration (or for the baseline seed step).

Subcommands:
  seed-baseline   Run benchmark.py on the baseline to capture initial timing.
  benchmark       Run benchmark.py on iterv{i}/kernel.<ext>.

Both write JSON under the appropriate directory. `benchmark` additionally
captures stderr so the agent can inspect it on validation failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping
from numbers import Real
from pathlib import Path


_BUNDLED_BENCHMARK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.py")
_EVIDENCE_STATUSES = {"confirmed_win", "confirmed_loss", "inconclusive", "invalid"}
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


def _read(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dims_argv(dims: dict) -> list[str]:
    return [f"--{k}={v}" for k, v in dims.items()]


def _ptr_size_argv(ptr_size: int) -> list[str]:
    return ["--ptr-size", str(ptr_size)] if ptr_size and ptr_size > 0 else []


def _decision_summary(decision: Mapping) -> dict:
    """Extract the sole statistic used by kernel decision consumers."""
    if not isinstance(decision, Mapping):
        raise ValueError("decision.json must contain a JSON object")
    status = decision.get("status")
    if type(status) is not str or status not in _DECISION_STATUSES:
        raise ValueError("decision.json status must be a recognized status")
    candidate_file = decision.get("candidate_file")
    if not isinstance(candidate_file, str) or not candidate_file.strip():
        raise ValueError("decision.json candidate_file must be a non-empty path")
    statistics = decision.get("statistics")
    if not isinstance(statistics, Mapping):
        raise ValueError("decision.json statistics must be a JSON object")
    required = (
        "statistic",
        "estimate_pct",
        "ci_low_pct",
        "ci_high_pct",
        "status",
    )
    missing = [field for field in required if field not in statistics]
    if missing:
        raise ValueError(f"decision.json statistics missing {missing[0]}")
    if not isinstance(statistics["statistic"], str) or not statistics[
        "statistic"
    ].strip():
        raise ValueError("decision.json statistics.statistic must be a string")
    if type(statistics["status"]) is not str or statistics[
        "status"
    ] not in _EVIDENCE_STATUSES:
        raise ValueError("decision.json statistics.status must be a known string")
    for field in ("estimate_pct", "ci_low_pct", "ci_high_pct"):
        value = statistics[field]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"decision.json statistics.{field} must be finite")
        if not math.isfinite(float(value)):
            raise ValueError(f"decision.json statistics.{field} must be finite")
    return {field: statistics[field] for field in required}


def _load_decision_summary(path: str) -> dict:
    if not os.path.isfile(path):
        raise ValueError(f"decision.json missing: {path}")
    try:
        decision = _read(path)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"decision.json is malformed: {error}") from error
    return _decision_summary(decision)


def _run_bench(
    *,
    benchmark_py: str,
    solution: str,
    ref: str,
    dims: dict,
    ptr_size: int,
    json_out: str,
    stderr_out: str,
    warmup: int,
    repeat: int,
) -> int:
    cmd = [
        sys.executable, benchmark_py, solution,
        "--ref", ref,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--json-out", json_out,
    ] + _ptr_size_argv(ptr_size) + _dims_argv(dims)
    print(f"[bench] {' '.join(cmd)}", file=sys.stderr)

    Path(json_out).parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    except OSError as e:
        with open(stderr_out, "w", encoding="utf-8") as f:
            f.write(f"Failed to exec benchmark: {e}\n")
        return -1

    with open(stderr_out, "w", encoding="utf-8") as f:
        f.write("---STDOUT---\n")
        f.write(r.stdout or "")
        f.write("\n---STDERR---\n")
        f.write(r.stderr or "")

    # benchmark.py exits 1 on validation failure but still writes json-out
    # (we rely on the correctness.passed field)
    if not os.path.isfile(json_out):
        # Catastrophic — write a minimal JSON for downstream consumers
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump({
                "correctness": {"passed": False, "checked": True},
                "kernel": None,
                "reference": None,
                "error": {
                    "code": "benchmark_crashed",
                    "stage": "subprocess",
                    "message": (r.stderr or "")[-2000:],
                },
            }, f, indent=2)
    return r.returncode


def cmd_seed_baseline(args: argparse.Namespace) -> None:
    state = _read(args.state)
    run_dir = state["run_dir"]
    out_dir = os.path.join(run_dir, "baseline")
    os.makedirs(out_dir, exist_ok=True)
    json_out = os.path.join(out_dir, "bench.json")
    stderr_out = os.path.join(out_dir, "bench.stderr.txt")

    _run_bench(
        benchmark_py=os.path.abspath(args.benchmark),
        solution=state["baseline_file"],
        ref=state["ref_file"],
        dims=state.get("dims", {}),
        ptr_size=state.get("ptr_size", 0),
        json_out=json_out,
        stderr_out=stderr_out,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    # Push baseline ms into state via state.py CLI (keep one place that writes state)
    # We call the sibling script so concurrent locking semantics live in one file.
    sibling = os.path.join(os.path.dirname(__file__), "state.py")
    rc = subprocess.call([
        sys.executable, sibling, "set-baseline-metric",
        "--state", args.state, "--bench", json_out,
    ])
    sys.exit(rc)


def cmd_benchmark(args: argparse.Namespace) -> None:
    state = _read(args.state)
    run_dir = state["run_dir"]
    iter_dir = os.path.join(run_dir, f"iterv{args.iter}")
    candidates = [os.path.join(iter_dir, "kernel.cu"), os.path.join(iter_dir, "kernel.py")]
    kernel = next((c for c in candidates if os.path.isfile(c)), None)
    if not kernel:
        sys.exit(f"No iterv{args.iter}/kernel.(cu|py) found.")

    json_out = os.path.join(iter_dir, "bench.json")
    stderr_out = os.path.join(iter_dir, "bench.stderr.txt")
    decision_path = os.path.join(iter_dir, "decision.json")
    decision_summary = _load_decision_summary(decision_path)
    _run_bench(
        benchmark_py=os.path.abspath(args.benchmark),
        solution=kernel,
        ref=state["ref_file"],
        dims=state.get("dims", {}),
        ptr_size=state.get("ptr_size", 0),
        json_out=json_out,
        stderr_out=stderr_out,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    # Return the result summary on stdout for the orchestrator to consume
    res = _read(json_out)
    summary = {
        "iter": args.iter,
        "kernel": kernel,
        "passed": bool(res.get("correctness", {}).get("passed", False)),
        "ms": (res.get("kernel") or {}).get("average_ms"),
        "ref_ms": (res.get("reference") or {}).get("average_ms"),
        "speedup_vs_ref": res.get("speedup_vs_reference"),
        "error": res.get("error"),
        "bench_json": json_out,
        "stderr_log": stderr_out,
        **decision_summary,
        "decision": dict(decision_summary),
    }
    print(json.dumps(summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("seed-baseline")
    ps.add_argument("--state", required=True)
    ps.add_argument("--benchmark", default=_BUNDLED_BENCHMARK,
                    help="Path to benchmark.py (default: bundled)")
    ps.add_argument("--warmup", type=int, default=10)
    ps.add_argument("--repeat", type=int, default=20)
    ps.set_defaults(func=cmd_seed_baseline)

    pb = sub.add_parser("benchmark")
    pb.add_argument("--state", required=True)
    pb.add_argument("--iter", type=int, required=True)
    pb.add_argument("--benchmark", default=_BUNDLED_BENCHMARK,
                    help="Path to benchmark.py (default: bundled)")
    pb.add_argument("--warmup", type=int, default=10)
    pb.add_argument("--repeat", type=int, default=20)
    pb.set_defaults(func=cmd_benchmark)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
