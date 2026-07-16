#!/usr/bin/env python3
"""Compute evidence-aware bottleneck gaps and allocate method budgets.

Utilization metrics provide heuristic compute, memory, and latency gaps. A
measured Roofline result is emitted only when explicit device peaks, workload
FLOPs, transferred bytes, and kernel time are all available.

Writes iterv{i}/roofline.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Explicit GPU and workload inputs
# ---------------------------------------------------------------------------

def _get_gpu_spec(env: dict) -> dict:
    """Read caller-provided peak FLOPS and bandwidth without guessing."""
    gpus = env.get("gpus") or [{}]
    gpu = env.get("selected_gpu") or gpus[0]
    return {
        "peak_flops_tflops": _positive_float(gpu.get("peak_flops_tflops")),
        "peak_bw_gbs": _positive_float(gpu.get("peak_bw_gbs")),
    }


def _positive_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 and math.isfinite(parsed) else None


def _get_workload(ncu_top: dict) -> dict:
    workload = ncu_top.get("workload") or {}
    return {
        "flops": _positive_float(workload.get("flops", ncu_top.get("workload_flops"))),
        "bytes": _positive_float(workload.get("bytes", ncu_top.get("workload_bytes"))),
        "kernel_time_ms": _positive_float(
            workload.get("kernel_time_ms", ncu_top.get("kernel_time_ms"))
        ),
    }


# ---------------------------------------------------------------------------
# Δ computation from ncu metrics
# ---------------------------------------------------------------------------

def _safe_float(v, default=0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _find_metric(ncu_top: dict, patterns: list[str], axis: str = None) -> float | None:
    """Find the first matching metric value from ncu_top.json axes."""
    axes_to_search = [axis] if axis else ["compute", "memory", "latency"]
    for ax in axes_to_search:
        metrics_list = ncu_top.get(ax, [])
        for m in metrics_list:
            name = m.get("metric", m.get("name", ""))
            for pat in patterns:
                if pat in name:
                    value = _safe_float(m.get("value"), default=None)
                    if value is not None:
                        return value
    return None


def compute_deltas(ncu_top: dict, env: dict) -> dict:
    """Compute Δ_c, Δ_m, Δ_l from ncu_top.json metrics.

    Missing metrics are represented as missing evidence, not as a full gap.
    """
    degraded = ncu_top.get("degraded", False)

    if degraded:
        return {
            "delta_compute": 0.0,
            "delta_memory": 0.0,
            "delta_latency": 0.0,
            "compute_util_pct": None,
            "memory_util_pct": None,
            "max_stall_pct": None,
            "evidence_axes": {"compute": False, "memory": False, "latency": False},
            "analysis_model": "utilization_gap",
            "analysis_quality": "unavailable",
            "ai_ridge": None,
            "arithmetic_intensity": None,
            "achieved_tflops": None,
            "achieved_bandwidth_gbs": None,
            "roofline_bound": None,
            "degraded": True,
        }

    # --- Compute gap ---
    # Primary: tensor core utilization for GEMM-like; FP32 pipe for others
    tensor_pct = _find_metric(ncu_top, [
        "pipe_tensor_op_hmma_cycles_active",
        "pipe_tensor_op_imma_cycles_active",
    ], "compute")
    fp32_pct = _find_metric(ncu_top, [
        "pipe_fp32_cycles_active",
    ], "compute")
    sm_throughput = _find_metric(ncu_top, [
        "sm__throughput",
    ], "compute")

    compute_values = [v for v in (tensor_pct, fp32_pct, sm_throughput) if v is not None]
    compute_util = max(compute_values) / 100.0 if compute_values else None
    delta_c = max(0.0, 1.0 - compute_util) if compute_util is not None else 0.0

    # --- Memory gap ---
    dram_throughput_value = _find_metric(ncu_top, [
        "dram__throughput",
        "gpu__compute_memory_throughput",
    ], "memory")
    dram_throughput_pct = (
        dram_throughput_value / 100.0 if dram_throughput_value is not None else None
    )
    delta_m = (
        max(0.0, 1.0 - dram_throughput_pct)
        if dram_throughput_pct is not None
        else 0.0
    )

    # --- Latency gap ---
    # Take the maximum stall percentage across all stall types
    stall_metrics = []
    for m in ncu_top.get("latency", []):
        name = m.get("metric", m.get("name", ""))
        name_l = name.lower()
        # Ignore pcsamp counters (counts), use pct-based stall metrics only.
        if "pcsamp" in name_l:
            continue
        if "pct" not in name_l:
            continue
        if "stalled" in name_l or "warp_latency" in name_l:
            v = _safe_float(m.get("value", 0.0))
            stall_metrics.append(min(max(v, 0.0), 100.0))

    max_stall_pct = max(stall_metrics) / 100.0 if stall_metrics else None
    delta_l = (
        min(1.0, max(0.0, max_stall_pct))
        if max_stall_pct is not None
        else 0.0
    )

    evidence_axes = {
        "compute": compute_util is not None,
        "memory": dram_throughput_pct is not None,
        "latency": max_stall_pct is not None,
    }

    # A true Roofline classification requires explicit device and workload data.
    spec = _get_gpu_spec(env)
    workload = _get_workload(ncu_top)
    measured_inputs = [
        spec["peak_flops_tflops"],
        spec["peak_bw_gbs"],
        workload["flops"],
        workload["bytes"],
        workload["kernel_time_ms"],
    ]
    measured = all(value is not None for value in measured_inputs)

    ai_ridge = None
    arithmetic_intensity = None
    achieved_tflops = None
    achieved_bandwidth_gbs = None
    roofline_bound = None
    if measured:
        ai_ridge = (
            spec["peak_flops_tflops"] * 1e12
            / (spec["peak_bw_gbs"] * 1e9)
        )
        arithmetic_intensity = workload["flops"] / workload["bytes"]
        seconds = workload["kernel_time_ms"] / 1000.0
        achieved_tflops = workload["flops"] / seconds / 1e12
        achieved_bandwidth_gbs = workload["bytes"] / seconds / 1e9
        roofline_bound = "compute" if arithmetic_intensity >= ai_ridge else "bandwidth"

    if measured:
        analysis_quality = "measured_roofline"
        analysis_model = "roofline"
    elif any(evidence_axes.values()):
        analysis_quality = "heuristic"
        analysis_model = "utilization_gap"
    else:
        analysis_quality = "unavailable"
        analysis_model = "utilization_gap"

    return {
        "delta_compute": round(delta_c, 4),
        "delta_memory": round(delta_m, 4),
        "delta_latency": round(delta_l, 4),
        "compute_util_pct": round(compute_util * 100, 2) if compute_util is not None else None,
        "memory_util_pct": round(dram_throughput_pct * 100, 2) if dram_throughput_pct is not None else None,
        "max_stall_pct": round(max_stall_pct * 100, 2) if max_stall_pct is not None else None,
        "evidence_axes": evidence_axes,
        "analysis_model": analysis_model,
        "analysis_quality": analysis_quality,
        "ai_ridge": round(ai_ridge, 4) if ai_ridge is not None else None,
        "arithmetic_intensity": (
            round(arithmetic_intensity, 4) if arithmetic_intensity is not None else None
        ),
        "achieved_tflops": round(achieved_tflops, 4) if achieved_tflops is not None else None,
        "achieved_bandwidth_gbs": (
            round(achieved_bandwidth_gbs, 4) if achieved_bandwidth_gbs is not None else None
        ),
        "roofline_bound": roofline_bound,
        "degraded": False,
    }


# ---------------------------------------------------------------------------
# Axis budget allocation
# ---------------------------------------------------------------------------

TOTAL_BUDGET = 3
MAX_PER_AXIS = 2
NEAR_PEAK_THRESHOLD = 0.15
# Tie-break order: memory > latency > compute (memory changes shift roofline
# position most; this is a structural choice, not a tunable param)
TIE_BREAK_ORDER = ["memory", "latency", "compute"]


def allocate_budget(delta_c: float, delta_m: float, delta_l: float) -> dict:
    """Allocate method budget per axis.

    Allocate up to three methods proportionally across evidenced gaps, with a
    per-axis cap of two. Zero and negligible gaps never receive a method.
    """
    deltas = {"compute": delta_c, "memory": delta_m, "latency": delta_l}

    eligible = {axis: gap for axis, gap in deltas.items() if gap >= 0.10}
    budgets = {"compute": 0, "memory": 0, "latency": 0}
    if not eligible:
        return budgets

    target_budget = min(TOTAL_BUDGET, MAX_PER_AXIS * len(eligible))
    total_delta = sum(eligible.values())
    raw = {
        axis: target_budget * gap / total_delta
        for axis, gap in eligible.items()
    }
    for axis in eligible:
        budgets[axis] = min(MAX_PER_AXIS, int(math.floor(raw[axis])))

    tie_rank = {axis: len(TIE_BREAK_ORDER) - i for i, axis in enumerate(TIE_BREAK_ORDER)}
    while sum(budgets.values()) < target_budget:
        candidates = [axis for axis in eligible if budgets[axis] < MAX_PER_AXIS]
        if not candidates:
            break
        best_axis = max(
            candidates,
            key=lambda axis: (
                raw[axis] - budgets[axis],
                eligible[axis],
                tie_rank[axis],
            ),
        )
        budgets[best_axis] += 1

    return budgets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(state_path: str, iteration: int) -> dict:
    with open(state_path, "r") as f:
        state = json.load(f)

    run_dir = state["run_dir"]
    iter_dir = os.path.join(run_dir, f"iterv{iteration}")
    ncu_top_path = os.path.join(iter_dir, "ncu_top.json")

    if not os.path.isfile(ncu_top_path):
        sys.exit(f"ncu_top.json not found at {ncu_top_path}")

    with open(ncu_top_path, "r") as f:
        ncu_top = json.load(f)

    env = state.get("env", {})

    # Compute deltas
    deltas = compute_deltas(ncu_top, env)

    dc = deltas["delta_compute"]
    dm = deltas["delta_memory"]
    dl = deltas["delta_latency"]

    # Check near-peak
    evidence_axes = deltas.get("evidence_axes", {})
    complete_evidence = all(
        evidence_axes.get(axis, False)
        for axis in ("compute", "memory", "latency")
    )
    near_peak = (complete_evidence and
                 dc < NEAR_PEAK_THRESHOLD and
                 dm < NEAR_PEAK_THRESHOLD and
                 dl < NEAR_PEAK_THRESHOLD)

    # Determine primary bound only from axes that actually have evidence.
    axis_gaps = {"compute": dc, "memory": dm, "latency": dl}
    evidenced_gaps = {
        axis: gap for axis, gap in axis_gaps.items() if evidence_axes.get(axis, False)
    }
    if near_peak:
        bound = "near_peak"
    elif not evidenced_gaps:
        bound = "unknown"
    else:
        primary_axis = max(evidenced_gaps, key=evidenced_gaps.get)
        bound = "bandwidth" if primary_axis == "memory" else primary_axis

    # Allocate budgets
    axis_budget = allocate_budget(dc, dm, dl)

    result = {
        **deltas,
        "bound": bound,
        "near_peak": near_peak,
        "axis_budget": axis_budget,
    }

    # Write roofline.json
    out_path = os.path.join(iter_dir, "roofline.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--iter", type=int, required=True)
    args = p.parse_args()
    run(args.state, args.iter)


if __name__ == "__main__":
    main()
