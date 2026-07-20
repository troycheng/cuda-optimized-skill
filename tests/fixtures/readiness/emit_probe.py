#!/usr/bin/env python3
"""Emit deterministic readiness or workload evidence for CPU-only tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: emit_probe.py ID STATUS")
    probe_id, status = sys.argv[1:]
    readiness_output = os.environ.get("CUDA_OPTIMIZER_READINESS_OUTPUT")
    workload_output = os.environ.get("CUDA_OPTIMIZER_OUTPUT")
    if readiness_output:
        payload = {
            "schema_version": "cuda-workload-optimizer/readiness-probe-v1",
            "requirement_id": probe_id,
            "status": status,
            "observations": {"fixture": True},
            "artifacts": [],
        }
        Path(readiness_output).write_text(json.dumps(payload), encoding="utf-8")
        return 0
    if workload_output:
        payload = {
            "schema_version": "cuda-workload-optimizer/probe-v1",
            "probe_id": probe_id,
            "kind": "timeline",
            "status": "ok",
            "metrics": {"gpu_busy_pct": 50.0, "cpu_busy_pct": 50.0},
            "issues": [],
            "artifacts": [],
        }
        Path(workload_output).write_text(json.dumps(payload), encoding="utf-8")
        return 0
    raise SystemExit("missing optimizer output environment variable")


if __name__ == "__main__":
    raise SystemExit(main())
