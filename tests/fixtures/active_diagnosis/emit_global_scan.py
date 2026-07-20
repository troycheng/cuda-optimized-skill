from __future__ import annotations

import json
import os
from pathlib import Path


Path(os.environ["CUDA_OPTIMIZER_OUTPUT"]).write_text(
    json.dumps(
        {
            "schema_version": "cuda-workload-optimizer/probe-v1",
            "probe_id": "timeline",
            "kind": "timeline",
            "status": "ok",
            "metrics": {
                "gpu_busy_pct": 42,
                "cpu_busy_pct": 91,
                "data_wait_pct": 45,
            },
            "issues": [],
            "artifacts": [],
        }
    ),
    encoding="utf-8",
)

coverage = []
for layer in (
    "cpu",
    "gpu",
    "framework",
    "transfer",
    "communication",
    "io",
    "synchronization",
    "idle",
):
    coverage.append(
        {
            "layer": layer,
            "status": "observed" if layer in {"cpu", "gpu"} else "not_observed",
            "reason": None if layer in {"cpu", "gpu"} else "not present in trace window",
        }
    )

Path(os.environ["CUDA_OPTIMIZER_ACTIVE_DIAGNOSIS_OUTPUT"]).write_text(
    json.dumps(
        {
            "schema_version": "cuda-optimizer/global-scan-draft-v1",
            "regime": {
                "shape_distribution_sha256": "1" * 64,
                "dynamic_branch_sha256": "2" * 64,
                "execution_regime_sha256": "3" * 64,
            },
            "boundary_ambiguous": False,
            "window": {"start_us": 0.0, "end_us": 1000.0},
            "coverage": coverage,
            "nodes": [
                {
                    "node_id": "cpu-launch",
                    "layer": "cpu",
                    "lane": "thread-7",
                    "kind": "cuda_api",
                    "label": "cudaLaunchKernel",
                    "duration_us": 900.0,
                    "occurrences": 4,
                    "timing_status": "observed",
                    "first_start_us": 0.0,
                    "last_end_us": 900.0,
                    "attribution_status": "explained",
                    "evidence_ids": ["ev-global-scan"],
                },
                {
                    "node_id": "gpu-kernel",
                    "layer": "gpu",
                    "lane": "stream-0",
                    "kind": "kernel",
                    "label": "decode_attention",
                    "duration_us": 900.0,
                    "occurrences": 4,
                    "timing_status": "observed",
                    "first_start_us": 100.0,
                    "last_end_us": 1000.0,
                    "attribution_status": "not_applicable",
                    "evidence_ids": ["ev-global-scan"],
                },
            ],
            "edges": [
                {
                    "source": "cpu-launch",
                    "target": "gpu-kernel",
                    "relation": "calls",
                    "overlap_us": None,
                    "evidence_ids": ["ev-global-scan"],
                }
            ],
            "hot_path": ["cpu-launch", "gpu-kernel"],
            "uncovered_intervals": [],
            "conclusion_level": "observed",
        }
    ),
    encoding="utf-8",
)
