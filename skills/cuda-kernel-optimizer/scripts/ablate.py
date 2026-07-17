#!/usr/bin/env python3
"""Single-method ablation for attribution.

For each method applied in the champion kernel, this script expects the agent
to have generated an ablated kernel (champion minus that one method) under
  iterv{i}/ablations/{method_id}/kernel.<ext>

This script benchmarks each ablated kernel and computes attribution:
  attribution(m) = ms_ablated(m) - ms_champion

Positive → the method helped (removing it slowed things down).
Zero/Negative → the method did not help or actually hurt.

Writes iterv{i}/attribution.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import artifact_store  # noqa: E402


_BUNDLED_BENCHMARK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.py")


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _regular_identity(path: str | os.PathLike) -> dict:
    """Capture one no-follow file descriptor identity and its exact bytes."""
    directory_fd = None
    file_fd = None
    try:
        directory_fd, leaf, target = artifact_store._open_parent_directory(
            path, create=False
        )
        file_fd = os.open(
            leaf, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd
        )
        metadata = os.fstat(file_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"ablation evidence is not a regular file: {target}")
        current = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ValueError(f"ablation evidence changed while opening: {target}")
        chunks = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        final = os.fstat(file_fd)
        current = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        identity = (metadata.st_dev, metadata.st_ino, metadata.st_size)
        if identity != (final.st_dev, final.st_ino, final.st_size) or (
            current.st_dev, current.st_ino, current.st_size
        ) != identity:
            raise ValueError(f"ablation evidence changed while reading: {target}")
        if len(payload) != metadata.st_size:
            raise ValueError(f"ablation evidence size changed while reading: {target}")
        return {
            "path": str(target),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "size": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": payload,
        }
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


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
    warmup: int = 5,
    repeat: int = 15,
) -> dict | None:
    """Run benchmark.py on a single kernel and return the parsed JSON result."""
    cmd = [
        sys.executable, benchmark_py, kernel_path,
        "--ref", ref_path,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--json-out", json_out,
    ] + _ptr_size_argv(ptr_size) + _dims_argv(dims)

    Path(json_out).parent.mkdir(parents=True, exist_ok=True)

    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="ignore",
        )
    except OSError as e:
        print(f"[ablate] benchmark failed: {e}", file=sys.stderr)
        return None

    if os.path.isfile(json_out):
        return _load_json(json_out)
    return None


def run(state_path: str, iteration: int, benchmark_py: str = None) -> dict:
    state = _load_json(state_path)
    run_dir = state["run_dir"]
    iter_dir = os.path.join(run_dir, f"iterv{iteration}")
    bench_py = benchmark_py or _BUNDLED_BENCHMARK

    # Load champion timing
    champion_bench = os.path.join(iter_dir, "bench.json")
    if not os.path.isfile(champion_bench):
        sys.exit(f"Champion bench.json not found at {champion_bench}")
    try:
        champion_identity = _regular_identity(champion_bench)
        champion_bench = champion_identity["path"]
        champion_data = json.loads(champion_identity["bytes"].decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        sys.exit(f"Champion bench.json is unsafe or malformed: {error}")
    champion_ms = (champion_data.get("kernel") or {}).get("average_ms")
    if champion_ms is None:
        sys.exit("Champion has no timing data")

    # Load methods
    methods_path = os.path.join(iter_dir, "methods.json")
    if not os.path.isfile(methods_path):
        sys.exit(f"methods.json not found at {methods_path}")
    methods_data = _load_json(methods_path)
    methods_list = methods_data.get("methods", [])

    ref_file = state["ref_file"]
    dims = state.get("dims", {})
    ptr_size = state.get("ptr_size", 0)
    noise_threshold = state.get("noise_threshold_pct", 2.0)

    attributions = []
    ablation_dir = os.path.join(iter_dir, "ablations")

    for m in methods_list:
        mid = m.get("id", "unknown")
        method_dir = os.path.join(ablation_dir, mid.replace(".", "_"))

        # Find ablated kernel
        ablated_kernel = None
        for ext in (".cu", ".py"):
            candidate = os.path.join(method_dir, f"kernel{ext}")
            try:
                _regular_identity(candidate)
            except (OSError, ValueError):
                continue
            ablated_kernel = str(Path(candidate).absolute())
            break

        if ablated_kernel is None:
            # No ablated kernel provided — skip, assume neutral
            attributions.append({
                "method_id": mid,
                "ablated_kernel": None,
                "ablated_ms": None,
                "champion_ms": champion_ms,
                "attribution_ms": 0.0,
                "attribution_pct": 0.0,
                "contributed": False,
                "note": "no_ablated_kernel_provided",
            })
            continue

        # Benchmark ablated kernel
        ablated_json_out = os.path.join(method_dir, "bench.json")
        try:
            kernel_before = _regular_identity(ablated_kernel)
        except (OSError, ValueError) as error:
            attributions.append({
                "method_id": mid,
                "contributed": False,
                "note": f"unsafe_or_drifted_ablation_evidence: {error}",
            })
            continue
        result = _bench_kernel(
            bench_py, ablated_kernel, ref_file, dims, ptr_size, ablated_json_out,
        )

        if result is None or not result.get("correctness", {}).get("passed", False):
            # Ablated kernel failed validation — method is likely essential
            attributions.append({
                "method_id": mid,
                "ablated_kernel": ablated_kernel,
                "ablated_ms": None,
                "champion_ms": champion_ms,
                "attribution_ms": None,
                "attribution_pct": None,
                "contributed": True,
                "note": "ablated_kernel_failed_validation_method_essential",
            })
            continue

        ablated_ms = (result.get("kernel") or {}).get("average_ms")
        if ablated_ms is None:
            attributions.append({
                "method_id": mid,
                "ablated_kernel": ablated_kernel,
                "ablated_ms": None,
                "champion_ms": champion_ms,
                "attribution_ms": 0.0,
                "attribution_pct": 0.0,
                "contributed": False,
                "note": "no_timing_in_ablated_bench",
            })
            continue

        try:
            stable_champion = _regular_identity(champion_bench)
            kernel_after = _regular_identity(ablated_kernel)
            bench_identity = _regular_identity(ablated_json_out)
            if (
                stable_champion != champion_identity
                or kernel_after != kernel_before
                or json.loads(bench_identity["bytes"].decode("utf-8")) != result
            ):
                raise ValueError("ablation evidence changed during collection")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            attributions.append({
                "method_id": mid,
                "contributed": False,
                "note": f"unsafe_or_drifted_ablation_evidence: {error}",
            })
            continue

        # Compute attribution
        attr_ms = ablated_ms - champion_ms
        attr_pct = (attr_ms / champion_ms * 100) if champion_ms > 0 else 0.0
        contributed = attr_pct > noise_threshold

        attributions.append({
            "method_id": mid,
            "champion_bench": champion_bench,
            "champion_bench_sha256": champion_identity["sha256"],
            "ablated_kernel": kernel_before["path"],
            "ablated_kernel_sha256": kernel_before["sha256"],
            "ablated_bench": bench_identity["path"],
            "ablated_bench_sha256": bench_identity["sha256"],
            "ablated_ms": round(ablated_ms, 4),
            "champion_ms": round(champion_ms, 4),
            "attribution_ms": round(attr_ms, 4),
            "attribution_pct": round(attr_pct, 2),
            "contributed": contributed,
        })

    try:
        if _regular_identity(champion_bench) != champion_identity:
            raise ValueError("champion bench identity drifted before attribution write")
    except (OSError, ValueError) as error:
        sys.exit(f"Champion bench.json changed before attribution write: {error}")

    output = {
        "iter": iteration,
        "champion_ms": round(champion_ms, 4),
        "noise_threshold_pct": noise_threshold,
        "attributions": attributions,
    }

    out_path = os.path.join(iter_dir, "attribution.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(json.dumps(output, indent=2))
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--benchmark", default=None)
    args = p.parse_args()
    run(args.state, args.iter, args.benchmark)


if __name__ == "__main__":
    main()
