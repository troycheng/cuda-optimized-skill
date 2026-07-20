#!/usr/bin/env python3
"""Real SM120 readiness probes used only by the opt-in RTX 5090 lane."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


SCHEMA = "cuda-workload-optimizer/readiness-probe-v1"
CUDA_SOURCE = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

extern "C" CUresult CUDAAPI cuProfilerStart(void);
extern "C" CUresult CUDAAPI cuProfilerStop(void);

__global__ void readiness_kernel(int* value) {
    if (threadIdx.x == 0) atomicAdd(value, 7);
}

int main() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 10;
    int* device = nullptr;
    int host = 0;
    if (cudaMalloc(&device, sizeof(int)) != cudaSuccess) return 11;
    if (cudaMemset(device, 0, sizeof(int)) != cudaSuccess) return 12;
    if (cuProfilerStart() != CUDA_SUCCESS) return 13;
    readiness_kernel<<<1, 32>>>(device);
    if (cuProfilerStop() != CUDA_SUCCESS) return 14;
    if (cudaDeviceSynchronize() != cudaSuccess) return 15;
    if (cudaMemcpy(&host, device, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 16;
    cudaFree(device);
    std::printf("SM=%d.%d VALUE=%d\n", prop.major, prop.minor, host);
    return (prop.major == 12 && prop.minor == 0 && host == 7) ? 0 : 17;
}
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(argv: list[str], *, timeout: float = 90) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _bounded_log(result: subprocess.CompletedProcess, limit: int = 4096) -> str:
    return ((result.stdout or "") + "\n" + (result.stderr or ""))[-limit:]


def _event(work_dir: Path, name: str) -> None:
    payload = json.dumps(
        {"event": name, "epoch": time.time()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    descriptor = os.open(
        work_dir / "events.jsonl",
        os.O_WRONLY | os.O_APPEND | os.O_CREAT,
        0o600,
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _emit(requirement_id: str, status: str, observations: dict) -> int:
    output = os.environ.get("CUDA_OPTIMIZER_READINESS_OUTPUT")
    if not output:
        raise SystemExit("CUDA_OPTIMIZER_READINESS_OUTPUT is required")
    payload = {
        "schema_version": SCHEMA,
        "requirement_id": requirement_id,
        "status": status,
        "observations": observations,
        "artifacts": [],
    }
    Path(output).write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


def _compile_binary(work_dir: Path, stem: str) -> tuple[Path | None, dict]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return None, {"reason": "nvcc_unavailable"}
    source = work_dir / f"{stem}.cu"
    binary = work_dir / stem
    source.write_text(CUDA_SOURCE, encoding="utf-8")
    result = _run(
        [
            nvcc,
            "-std=c++17",
            "-O2",
            "-lineinfo",
            "-arch=sm_120",
            str(source),
            "-lcuda",
            "-o",
            str(binary),
        ]
    )
    if result.returncode != 0 or not binary.is_file():
        return None, {
            "reason": "target_compile_failed",
            "returncode": result.returncode,
            "log_tail": _bounded_log(result),
        }
    return binary, {
        "nvcc": os.path.realpath(nvcc),
        "binary_sha256": _sha256(binary),
    }


def _cuda_foundation(requirement_id: str, work_dir: Path) -> int:
    binary, observations = _compile_binary(work_dir, "foundation_sm120")
    if binary is None:
        return _emit(requirement_id, "unavailable", observations)
    execute = _run([str(binary)])
    if execute.returncode != 0 or "SM=12.0 VALUE=7" not in execute.stdout:
        observations.update(
            {
                "reason": "gpu_execute_failed",
                "returncode": execute.returncode,
                "log_tail": _bounded_log(execute),
            }
        )
        return _emit(requirement_id, "failed", observations)
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        observations["reason"] = "cuobjdump_unavailable"
        return _emit(requirement_id, "unavailable", observations)
    sass = _run([cuobjdump, "--dump-sass", str(binary)])
    if sass.returncode != 0 or not sass.stdout.strip():
        observations.update(
            {
                "reason": "sass_dump_failed",
                "returncode": sass.returncode,
                "log_tail": _bounded_log(sass),
            }
        )
        return _emit(requirement_id, "failed", observations)
    observations.update(
        {
            "sm_arch": "sm_120",
            "gpu_execute": True,
            "sass_sha256": hashlib.sha256(sass.stdout.encode("utf-8")).hexdigest(),
            "cuobjdump": os.path.realpath(cuobjdump),
        }
    )
    _event(work_dir, "foundation-complete")
    return _emit(requirement_id, "ready", observations)


def _nsys(requirement_id: str, work_dir: Path) -> int:
    nsys = shutil.which("nsys")
    if nsys is None:
        return _emit(requirement_id, "unavailable", {"reason": "nsys_unavailable"})
    binary, observations = _compile_binary(work_dir, "nsys_sm120")
    if binary is None:
        return _emit(requirement_id, "unavailable", observations)
    prefix = work_dir / "nsys-smoke"
    profile = _run(
        [
            nsys,
            "profile",
            "--force-overwrite=true",
            "--trace=cuda",
            "--output",
            str(prefix),
            str(binary),
        ]
    )
    report = prefix.with_suffix(".nsys-rep")
    if profile.returncode != 0 or not report.is_file():
        observations.update(
            {
                "reason": "nsys_profile_failed",
                "returncode": profile.returncode,
                "log_tail": _bounded_log(profile),
            }
        )
        return _emit(requirement_id, "failed", observations)
    stats = _run(
        [nsys, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", str(report)]
    )
    if stats.returncode != 0 or "readiness_kernel" not in stats.stdout:
        observations.update(
            {
                "reason": "nsys_stats_failed",
                "returncode": stats.returncode,
                "log_tail": _bounded_log(stats),
            }
        )
        return _emit(requirement_id, "failed", observations)
    observations.update(
        {
            "report_sha256": _sha256(report),
            "stats_sha256": hashlib.sha256(stats.stdout.encode("utf-8")).hexdigest(),
        }
    )
    return _emit(requirement_id, "ready", observations)


def _ncu(requirement_id: str, work_dir: Path) -> int:
    ncu = shutil.which("ncu")
    if ncu is None:
        return _emit(requirement_id, "unavailable", {"reason": "ncu_unavailable"})
    binary, observations = _compile_binary(work_dir, "ncu_sm120")
    if binary is None:
        return _emit(requirement_id, "unavailable", observations)
    report_prefix = work_dir / "ncu-smoke"
    profile = _run(
        [
            ncu,
            "--set",
            "basic",
            "--kernel-name-base",
            "demangled",
            "--kernel-name",
            "regex:readiness_kernel",
            "--profile-from-start",
            "off",
            "--launch-count",
            "1",
            "--target-processes",
            "all",
            "--export",
            str(report_prefix),
            "--force-overwrite",
            str(binary),
        ],
        timeout=120,
    )
    observations.update(
        {
            "target_filter": "regex:readiness_kernel",
            "target_range": "cuProfilerStart/cuProfilerStop",
            "launch_count": 1,
        }
    )
    log = _bounded_log(profile, limit=8192)
    if "ERR_NVGPUCTRPERM" in log or (
        "permission" in log.lower() and "counter" in log.lower()
    ):
        observations.update(
            {"counter_access": False, "counter_access_error": "ERR_NVGPUCTRPERM"}
        )
        return _emit(requirement_id, "degraded", observations)
    reports = sorted(work_dir.glob("ncu-smoke*.ncu-rep"))
    if profile.returncode != 0 or not reports:
        observations.update(
            {
                "reason": "ncu_target_profile_failed",
                "returncode": profile.returncode,
                "log_tail": log,
            }
        )
        return _emit(requirement_id, "failed", observations)
    observations.update(
        {
            "counter_access": True,
            "report_sha256": _sha256(reports[-1]),
        }
    )
    return _emit(requirement_id, "ready", observations)


def _sanitizer(requirement_id: str, work_dir: Path) -> int:
    sanitizer = shutil.which("compute-sanitizer")
    if sanitizer is None:
        return _emit(
            requirement_id,
            "unavailable",
            {"reason": "compute_sanitizer_unavailable"},
        )
    binary, observations = _compile_binary(work_dir, "sanitizer_sm120")
    if binary is None:
        return _emit(requirement_id, "unavailable", observations)
    result = _run(
        [sanitizer, "--tool", "memcheck", "--error-exitcode", "86", str(binary)],
        timeout=120,
    )
    log = _bounded_log(result, limit=8192)
    if result.returncode != 0 or "ERROR SUMMARY: 0 errors" not in log:
        observations.update(
            {
                "reason": "compute_sanitizer_failed",
                "returncode": result.returncode,
                "log_tail": log,
            }
        )
        return _emit(requirement_id, "failed", observations)
    observations["memcheck_zero_errors"] = True
    return _emit(requirement_id, "ready", observations)


def _workload_smoke(
    requirement_id: str,
    work_dir: Path,
    workload: Path | None,
    fail: bool,
) -> int:
    _event(work_dir, "workload-smoke-start")
    if fail:
        return _emit(requirement_id, "failed", {"reason": "injected_workload_failure"})
    if workload is None or workload.is_symlink() or not workload.is_file():
        return _emit(requirement_id, "failed", {"reason": "workload_missing"})
    spec = importlib.util.spec_from_file_location("sm120_readiness_workload", workload)
    if spec is None or spec.loader is None:
        return _emit(requirement_id, "failed", {"reason": "workload_import_failed"})
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import torch

    state = module.setup(N=1_048_576, seed=20260720)
    inputs = state["inputs"]
    module.run_kernel(**inputs)
    torch.cuda.synchronize()
    expected = inputs["x"] * inputs["x"] + 1.0
    maximum_error = float((inputs["out"] - expected).abs().max().item())
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        module.run_kernel(**inputs)
    end.record()
    end.synchronize()
    latency_ms = float(start.elapsed_time(end)) / 10.0
    if maximum_error > 1e-5 or latency_ms <= 0:
        return _emit(
            requirement_id,
            "failed",
            {"max_abs_error": maximum_error, "latency_ms": latency_ms},
        )
    _event(work_dir, "workload-smoke-complete")
    return _emit(
        requirement_id,
        "ready",
        {"max_abs_error": maximum_error, "latency_ms": latency_ms},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("cuda-foundation", "nsys", "ncu", "sanitizer", "workload-smoke", "sleep"),
    )
    parser.add_argument("--requirement-id", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--fail", action="store_true")
    args = parser.parse_args(argv)
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "sleep":
        time.sleep(60)
        return _emit(args.requirement_id, "ready", {})
    if args.fail and args.mode != "workload-smoke":
        return _emit(
            args.requirement_id,
            "failed",
            {"reason": "injected_probe_failure"},
        )
    try:
        if args.mode == "cuda-foundation":
            return _cuda_foundation(args.requirement_id, work_dir)
        if args.mode == "nsys":
            return _nsys(args.requirement_id, work_dir)
        if args.mode == "ncu":
            return _ncu(args.requirement_id, work_dir)
        if args.mode == "sanitizer":
            return _sanitizer(args.requirement_id, work_dir)
        return _workload_smoke(
            args.requirement_id,
            work_dir,
            args.workload,
            args.fail,
        )
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
        return _emit(
            args.requirement_id,
            "failed",
            {"reason": type(error).__name__, "detail": str(error)[:512]},
        )


if __name__ == "__main__":
    raise SystemExit(main())
