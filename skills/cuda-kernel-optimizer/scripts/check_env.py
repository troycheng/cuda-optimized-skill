#!/usr/bin/env python3
"""Detect local GPU / CUDA / Triton / CUTLASS environment.

Writes a JSON snapshot consumed by later steps. Safe to run even if some
tools are missing — fields are simply null.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], timeout: int = 10) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="ignore",
        )
        return r.returncode, r.stdout or "", r.stderr or ""
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        return -1, "", str(e)


def _detect_gpus() -> list[dict]:
    gpus: list[dict] = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(i)
                gpus.append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "compute_capability": f"{major}.{minor}",
                    "sm_arch": f"sm_{major}{minor}",
                    "total_memory_mb": torch.cuda.get_device_properties(i).total_memory // (1024 * 1024),
                })
    except Exception as e:
        return [{"error": f"torch probe failed: {e}"}]
    return gpus


def _resolve_tool(requested: str, fallback: str) -> str | None:
    requested = (requested or fallback).strip()
    if os.path.isfile(os.path.expanduser(requested)):
        return os.path.abspath(os.path.expanduser(requested))
    return shutil.which(requested)


def _detect_nvcc(requested: str = "nvcc") -> dict:
    path = _resolve_tool(requested, "nvcc")
    if not path:
        return {"available": False, "requested": requested, "path": None, "version": None}
    rc, out, _ = _run([path, "--version"])
    version = None
    if rc == 0:
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Cuda compilation tools"):
                version = line
                break
    return {"available": True, "requested": requested, "path": path, "version": version or out.strip()}


def _detect_ncu(requested: str = "ncu") -> dict:
    path = _resolve_tool(requested, "ncu")
    if not path:
        return {
            "available": False,
            "requested": requested,
            "path": None,
            "version": None,
            "metrics_query_available": None,
            "can_read_counters": None,
        }
    rc, out, _ = _run([path, "--version"])
    version = None
    if rc == 0:
        for line in out.splitlines():
            match = re.match(r"\s*Version\s+([^\s]+)", line)
            if match:
                version = match.group(1)
                break
    if version is None and out:
        version = out.strip().splitlines()[0]
    # This checks metric metadata only. Counter permissions require a real profile.
    rc2, _, err2 = _run([path, "--query-metrics"], timeout=5)
    metrics_query_available = rc2 == 0
    return {
        "available": True,
        "requested": requested,
        "path": path,
        "version": version,
        "metrics_query_available": metrics_query_available,
        "can_read_counters": None,
        "note": None if metrics_query_available else (
            err2.strip()[:400] or "ncu metric metadata query failed"
        ),
    }


def _detect_driver() -> dict:
    path = shutil.which("nvidia-smi")
    if not path:
        return {
            "available": False,
            "path": None,
            "driver_versions": [],
            "max_cuda_version": None,
        }

    rc, out, _ = _run(
        [path, "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    driver_versions = []
    if rc == 0:
        driver_versions = list(
            dict.fromkeys(line.strip() for line in out.splitlines() if line.strip())
        )

    header_rc, header, _ = _run([path])
    max_cuda_version = None
    if header_rc == 0:
        match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)*)", header)
        if match:
            max_cuda_version = match.group(1)

    return {
        "available": True,
        "path": path,
        "driver_versions": driver_versions,
        "max_cuda_version": max_cuda_version,
    }


def _detect_cutlass() -> dict:
    # Mirrors benchmark.py's find_cutlass_include_dir
    candidates: list[str] = []
    for var in ("CUTLASS_PATH", "CUTLASS_INCLUDE_DIR"):
        v = os.environ.get(var, "").strip()
        if v:
            candidates.append(v)
            candidates.append(os.path.join(v, "include"))
    candidates.extend(sorted(glob.glob("/usr/local/cutlass*/include")))
    candidates.extend(["/usr/local/cutlass/include", "/opt/cutlass/include"])
    seen = set()
    for c in candidates:
        if not c:
            continue
        r = os.path.abspath(c)
        if r in seen:
            continue
        seen.add(r)
        if os.path.isdir(os.path.join(r, "cutlass")) and os.path.isdir(os.path.join(r, "cute")):
            return {"available": True, "include_dir": r}
    return {"available": False, "include_dir": None}


def _detect_python_libs() -> dict:
    libs: dict = {}
    for name in ("torch", "triton", "cutlass"):  # cutlass python if any
        try:
            mod = __import__(name)
            libs[name] = {"available": True, "version": getattr(mod, "__version__", "unknown")}
        except Exception:
            libs[name] = {"available": False, "version": None}
    return libs


def collect_env(
    *,
    gpu: int = 0,
    requested_arch: str = "",
    nvcc_bin: str = "nvcc",
    ncu_bin: str = "ncu",
) -> dict:
    gpus = _detect_gpus()
    selected_gpu = next((g for g in gpus if g.get("index") == gpu), None)
    primary_arch = selected_gpu.get("sm_arch") if selected_gpu else None
    return {
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "gpus": gpus,
        "selected_gpu_index": gpu,
        "selected_gpu": selected_gpu,
        "requested_arch": requested_arch or None,
        "primary_sm_arch": primary_arch,
        "nvcc": _detect_nvcc(nvcc_bin),
        "ncu": _detect_ncu(ncu_bin),
        "driver": _detect_driver(),
        "cutlass": _detect_cutlass(),
        "libs": _detect_python_libs(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="./env.json", help="Output JSON path")
    p.add_argument("--gpu", type=int, default=0, help="GPU index selected for the run")
    p.add_argument("--arch", type=str, default="", help="Requested compiler architecture")
    p.add_argument("--nvcc-bin", type=str, default="nvcc")
    p.add_argument("--ncu-bin", type=str, default="ncu")
    args = p.parse_args()

    env = collect_env(
        gpu=args.gpu,
        requested_arch=args.arch,
        nvcc_bin=args.nvcc_bin,
        ncu_bin=args.ncu_bin,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, ensure_ascii=False)

    # Also print a compact summary to stdout for the agent and user to inspect.
    print(json.dumps({
        "gpu": (env.get("selected_gpu") or {}).get("name"),
        "sm_arch": env["primary_sm_arch"],
        "nvcc": env["nvcc"].get("version"),
        "ncu": env["ncu"].get("available"),
        "ncu_metrics_query_available": env["ncu"].get("metrics_query_available"),
        "ncu_can_read_counters": env["ncu"].get("can_read_counters"),
        "cutlass": env["cutlass"].get("available"),
        "torch": env["libs"].get("torch", {}).get("version"),
        "triton": env["libs"].get("triton", {}).get("version"),
        "out": args.out,
    }, indent=2))

    # Useful for callers: exit 0 regardless — env is informational, not a gate
    sys.exit(0)


if __name__ == "__main__":
    main()
