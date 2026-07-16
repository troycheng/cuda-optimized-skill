#!/usr/bin/env python3
"""Branch-and-Select using one paired statistical decision for every candidate.

All K branches share the same method combination (from methods.json) but
differ in hyperparameters (tile size, num_stages, num_warps, etc.).

The agent generates K kernels under iterv{i}/branches/b{1..K}/kernel.<ext>.

Correctness is checked first. Only correctness-passing candidates enter the
paired AB/BA timing engine, and only statistically confirmed winners are
eligible for the shortlist or promotion.

Writes iterv{i}/branch_results.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from numbers import Real
from pathlib import Path


_BUNDLED_BENCHMARK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.py")

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paired_benchmark import run_paired  # noqa: E402
from paired_stats import classify_pairs  # noqa: E402
from artifact_store import sha256_file  # noqa: E402


_PAIRED_STATUSES = {"confirmed_win", "confirmed_loss", "inconclusive", "invalid"}
_COMPLETED_STATUSES = {"confirmed_win", "confirmed_loss", "inconclusive"}
_STATISTIC_FIELDS = (
    "statistic",
    "estimate_pct",
    "ci_low_pct",
    "ci_high_pct",
    "status",
)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _bounded_candidate_error(error: Exception, limit: int = 560) -> str:
    """Return a bounded, single-line diagnostic without hiding its type."""
    detail = " ".join(str(error).split())
    message = f"invalid_statistics: {type(error).__name__}: {detail}"
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def _promote_kernel(source: str, destination: str, iter_dir: str) -> None:
    """Atomically publish the selected kernel and remove a stale suffix."""
    source_path = Path(source)
    destination_path = Path(destination)
    iteration_path = Path(iter_dir)

    if os.path.abspath(source) != os.path.abspath(destination):
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination_path.name}.",
            suffix=".tmp",
            dir=str(iteration_path),
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            shutil.copy2(source_path, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, destination_path)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise

    for suffix in (".cu", ".py"):
        stale = iteration_path / f"kernel{suffix}"
        if stale == destination_path:
            continue
        if stale.is_symlink() or stale.is_file():
            stale.unlink()
        elif stale.exists():
            raise ValueError(f"stale champion path is not a file: {stale}")


def _dims_argv(dims: dict) -> list[str]:
    return [f"--{k}={v}" for k, v in dims.items()]


def _ptr_size_argv(ptr_size: int) -> list[str]:
    return ["--ptr-size", str(ptr_size)] if ptr_size and ptr_size > 0 else []


def _bench_kernel(
    benchmark_py: str,
    kernel_path: str,
    ref_path: str,
    dims: dict,
    ptr_size: int,
    json_out: str,
    warmup: int = 10,
    repeat: int = 20,
) -> dict:
    """Run benchmark.py on a kernel. Returns parsed result or error dict."""
    cmd = [
        sys.executable, benchmark_py, kernel_path,
        "--ref", ref_path,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--json-out", json_out,
    ] + _ptr_size_argv(ptr_size) + _dims_argv(dims)

    Path(json_out).parent.mkdir(parents=True, exist_ok=True)
    stderr_out = json_out.replace(".json", ".stderr.txt")

    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="ignore",
        )
    except OSError as e:
        return {"error": str(e), "passed": False}

    # Save stderr for debugging
    with open(stderr_out, "w", encoding="utf-8") as f:
        f.write("---STDOUT---\n")
        f.write(r.stdout or "")
        f.write("\n---STDERR---\n")
        f.write(r.stderr or "")

    if os.path.isfile(json_out):
        return _load_json(json_out)

    return {
        "error": "no_json_output",
        "stderr": (r.stderr or "")[-2000:],
        "passed": False,
    }


def _paired_candidate(
    baseline_file: str,
    candidate_file: str,
    *,
    backend: str,
    dims: dict,
    ptr_size: int,
    arch: str,
    nvcc_bin: str,
    seed: int,
    blocks: int,
    warmup: int,
    min_effect_pct: float,
    confidence: float = 0.95,
    bootstrap_samples: int = 10000,
    max_temperature_delta_c: float = 5,
    max_clock_delta_pct: float = 5,
) -> dict:
    """Collect paired samples and return the shared classification payload."""
    paired = run_paired(
        baseline_file,
        candidate_file,
        backend=backend,
        dims=copy.deepcopy(dims),
        ptr_size=ptr_size,
        arch=arch,
        nvcc_bin=nvcc_bin,
        seed=seed,
        blocks=blocks,
        warmup=warmup,
        max_temperature_delta_c=max_temperature_delta_c,
        max_clock_delta_pct=max_clock_delta_pct,
    )
    return classify_pairs(
        paired["pairs"],
        direction="lower",
        min_effect_pct=min_effect_pct,
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def _finite_statistic(value, field: str, *, required: bool) -> float | None:
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"statistics.{field} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"statistics.{field} must be a finite number")
    return parsed


def _validate_statistics(payload) -> dict:
    """Return a detached, ranking-safe statistics payload."""
    if not isinstance(payload, Mapping):
        raise ValueError("statistics must be a mapping")
    missing = [field for field in _STATISTIC_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"statistics missing required field: {missing[0]}")
    statistic = payload["statistic"]
    if not isinstance(statistic, str) or not statistic.strip():
        raise ValueError("statistics.statistic must be a non-empty string")
    status = payload["status"]
    if not isinstance(status, str) or status not in _PAIRED_STATUSES:
        raise ValueError("statistics.status is invalid")
    numeric_required = status == "confirmed_win"
    clean = copy.deepcopy(dict(payload))
    for field in ("estimate_pct", "ci_low_pct", "ci_high_pct"):
        clean[field] = _finite_statistic(
            payload[field], field, required=numeric_required
        )
    return clean


def _state_arch(state: dict) -> str:
    env = state.get("env") or {}
    arch = state.get("arch") or env.get("primary_sm_arch")
    if not arch:
        gpus = env.get("gpus") or []
        if gpus and isinstance(gpus[0], Mapping):
            arch = gpus[0].get("sm_arch")
    if not isinstance(arch, str) or not arch.strip():
        raise ValueError("state must provide arch or env.primary_sm_arch")
    return arch


def _state_nvcc_bin(state: dict) -> str:
    env = state.get("env") or {}
    nvcc = env.get("nvcc") or {}
    value = state.get("nvcc_bin") or nvcc.get("path") or "nvcc"
    if not isinstance(value, str) or not value.strip():
        raise ValueError("state nvcc_bin must be a non-empty string")
    return value


def run(state_path: str, iteration: int, benchmark_py: str = None,
        warmup: int = 10, repeat: int = 20) -> dict:
    state = _load_json(state_path)
    run_dir = state["run_dir"]
    iter_dir = os.path.join(run_dir, f"iterv{iteration}")
    bench_py = benchmark_py or _BUNDLED_BENCHMARK
    branches_dir = os.path.join(iter_dir, "branches")
    ref_file = state["ref_file"]
    dims = state.get("dims", {})
    ptr_size = state.get("ptr_size", 0)
    num_branches = state.get("branches", 4)
    baseline_file = state.get("best_file") or state.get("baseline_file")
    if not isinstance(baseline_file, str) or not baseline_file.strip():
        raise ValueError("state must provide best_file or baseline_file")
    budget = copy.deepcopy(state.get("budget") or {})
    blocks = budget.get("max_pairs", repeat)
    backend = state.get("backend", "auto")
    arch = _state_arch(state)
    nvcc_bin = _state_nvcc_bin(state)
    seed = state.get("seed", 0)
    min_effect_pct = state.get("min_effect_pct", 0.5)
    confidence = state.get("confidence", 0.95)
    bootstrap_samples = state.get("bootstrap_samples", 10000)
    max_temperature_delta_c = state.get("max_temperature_delta_c", 5)
    max_clock_delta_pct = state.get("max_clock_delta_pct", 5)

    # Discover branches
    branch_dirs = []
    for i in range(1, num_branches + 1):
        bd = os.path.join(branches_dir, f"b{i}")
        if os.path.isdir(bd):
            # Check if there's a kernel file
            kernel = None
            for ext in (".cu", ".py"):
                candidate = os.path.join(bd, f"kernel{ext}")
                if os.path.isfile(candidate):
                    kernel = candidate
                    break
            if kernel:
                branch_dirs.append({"index": i, "dir": bd, "kernel": kernel})

    if not branch_dirs:
        # Fallback: check if there's a single kernel directly in iter_dir
        for ext in (".cu", ".py"):
            candidate = os.path.join(iter_dir, f"kernel{ext}")
            if os.path.isfile(candidate):
                branch_dirs.append({
                    "index": 0, "dir": iter_dir, "kernel": candidate,
                })
                break

    if not branch_dirs:
        sys.exit(f"No branch kernels found under {branches_dir}")

    print(f"[branch_explore] Found {len(branch_dirs)} branches", file=sys.stderr)

    # Correctness-check and then statistically compare all passing branches.
    results = []
    for branch in branch_dirs:
        idx = branch["index"]
        kernel = branch["kernel"]
        json_out = os.path.join(branch["dir"], "bench.json")

        print(f"[branch {idx}] Benchmarking {os.path.basename(kernel)}...",
              file=sys.stderr)

        bench_result = _bench_kernel(
            bench_py, kernel, ref_file, dims, ptr_size, json_out, warmup, repeat,
        )

        passed = bool(bench_result.get("correctness", {}).get("passed", False))
        kernel_stats = bench_result.get("kernel") or {}
        average_ms = kernel_stats.get("average_ms")
        median_ms = kernel_stats.get("median_ms")
        ms = median_ms if median_ms is not None else average_ms

        result = {
            "branch_index": idx,
            "kernel": kernel,
            "correctness": "passed" if passed else "failed",
            "passed": passed,
            "ms": ms,
            "average_ms": average_ms,
            "median_ms": median_ms,
            "p95_ms": kernel_stats.get("p95_ms"),
            "cv_pct": kernel_stats.get("cv_pct"),
            "error": bench_result.get("error"),
            "statistics": None,
            "status": "rejected_correctness" if not passed else "invalid",
        }

        if passed:
            try:
                statistics = _validate_statistics(
                    _paired_candidate(
                        baseline_file,
                        kernel,
                        backend=backend,
                        dims=copy.deepcopy(dims),
                        ptr_size=ptr_size,
                        arch=arch,
                        nvcc_bin=nvcc_bin,
                        seed=seed,
                        blocks=blocks,
                        warmup=warmup,
                        min_effect_pct=min_effect_pct,
                        confidence=confidence,
                        bootstrap_samples=bootstrap_samples,
                        max_temperature_delta_c=max_temperature_delta_c,
                        max_clock_delta_pct=max_clock_delta_pct,
                    )
                )
            except Exception as error:
                result["error"] = _bounded_candidate_error(error)
            else:
                result["statistics"] = statistics
                result["status"] = statistics["status"]

        results.append(result)

        status = "PASS" if passed else "FAIL"
        ms_str = f"{ms:.4f} ms" if ms else "N/A"
        print(f"[branch {idx}] {status}  {ms_str}", file=sys.stderr)

    shortlist = sorted(
        (result for result in results if result["status"] == "confirmed_win"),
        key=lambda result: result["statistics"]["estimate_pct"],
        reverse=True,
    )
    completed_comparisons = sum(
        result["status"] in _COMPLETED_STATUSES for result in results
    )
    measurement_failures = sum(
        result["passed"] and result["status"] == "invalid"
        for result in results
    )

    if not shortlist:
        output = {
            "iter": iteration,
            "status": "no_confirmed_kernel_win",
            "branches": results,
            "champion": None,
            "shortlist": [],
            "frontier": [],
            "total_branches": len(branch_dirs),
            "valid_branches": sum(result["passed"] for result in results),
            "completed_comparisons": completed_comparisons,
            "measurement_failures": measurement_failures,
        }
        _write_json(os.path.join(iter_dir, "branch_results.json"), output)
        _write_json(
            os.path.join(iter_dir, "decision.json"),
            {
                "status": "no_confirmed_kernel_win",
                "candidate_file": None,
                "statistics": None,
            },
        )
        print(json.dumps(output, indent=2))
        return output

    champion = shortlist[0]

    # Copy champion kernel to iterv{i}/kernel.<ext>
    champ_kernel = champion["kernel"]
    ext = os.path.splitext(champ_kernel)[1]
    dest = os.path.join(iter_dir, f"kernel{ext}")
    _promote_kernel(champ_kernel, dest, iter_dir)
    candidate_sha256 = sha256_file(dest)

    # Also copy champion bench.json to iter_dir
    champ_bench = os.path.join(os.path.dirname(champ_kernel), "bench.json")
    dest_bench = os.path.join(iter_dir, "bench.json")
    if os.path.isfile(champ_bench) and os.path.abspath(champ_bench) != os.path.abspath(dest_bench):
        shutil.copy2(champ_bench, dest_bench)

    frontier_entries = [copy.deepcopy(item) for item in shortlist[1:]]

    output = {
        "iter": iteration,
        "status": "shortlist_ready",
        "champion": copy.deepcopy(champion),
        "shortlist": copy.deepcopy(shortlist),
        "selected_kernel": dest,
        "branches": results,
        "frontier": frontier_entries,
        "total_branches": len(branch_dirs),
        "valid_branches": sum(result["passed"] for result in results),
        "completed_comparisons": completed_comparisons,
        "measurement_failures": measurement_failures,
    }

    _write_json(os.path.join(iter_dir, "branch_results.json"), output)
    _write_json(
        os.path.join(iter_dir, "decision.json"),
        {
            "status": champion["statistics"]["status"],
            "candidate_file": os.path.abspath(dest),
            "candidate_sha256": candidate_sha256,
            "source_candidate_file": os.path.abspath(champ_kernel),
            "statistics": copy.deepcopy(champion["statistics"]),
        },
    )
    print(json.dumps(output, indent=2))
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--benchmark", default=None)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=20)
    args = p.parse_args()
    output = run(args.state, args.iter, args.benchmark, args.warmup, args.repeat)
    if (
        output.get("status") == "no_confirmed_kernel_win"
        and output.get("valid_branches", 0) == 0
    ):
        sys.exit(2)


if __name__ == "__main__":
    main()
