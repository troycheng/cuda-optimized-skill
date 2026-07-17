"""Produce one normalized probe from real CUDA work and nvidia-smi sampling."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import torch


def _sample_gpu_busy() -> float:
    left = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
    right = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
    for _ in range(12):
        torch.mm(left, right)
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=20,
    )
    torch.cuda.synchronize()
    first = output.splitlines()[0].strip()
    value = float(first)
    if not 0 <= value <= 100:
        raise ValueError(f"nvidia-smi returned invalid utilization: {value}")
    return value


def main() -> None:
    issues = []
    metrics = {}
    status = "ok"
    try:
        metrics["gpu_busy_pct"] = _sample_gpu_busy()
    except (OSError, ValueError, subprocess.SubprocessError, IndexError) as error:
        status = "degraded"
        issues.append(
            {
                "id": "environment:gpu-utilization-sample",
                "category": "environment",
                "severity": "warning",
                "message": f"GPU utilization sample unavailable: {type(error).__name__}",
            }
        )
    payload = {
        "schema_version": "cuda-workload-optimizer/probe-v1",
        "probe_id": "timeline",
        "kind": "timeline",
        "status": status,
        "metrics": metrics,
        "issues": issues,
        "artifacts": [],
    }
    Path(os.environ["CUDA_OPTIMIZER_OUTPUT"]).write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
