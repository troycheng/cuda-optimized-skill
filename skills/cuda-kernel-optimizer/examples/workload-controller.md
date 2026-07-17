# Workload controller example

This example starts with a runnable workload, diagnoses why the GPU is idle,
and evaluates one project-scoped change. Replace the absolute paths and commands
with the user's real environment; keep the schemas and boundaries unchanged.

## 1. Describe the run

`control.json`:

```json
{
  "schema_version": "cuda-workload-optimizer/control-v1",
  "project_root": "/workspace/service",
  "workload_manifest": "/workspace/service/workload.json",
  "baseline_candidate": {"name": "baseline", "revision": "abc123"},
  "budget": "balanced",
  "mutation": {
    "project_paths": ["src", "configs"],
    "environment_root": "/workspace/service-env-copy",
    "host_policy": "recommend_only"
  },
  "probes": [
    {
      "id": "timeline",
      "kind": "timeline",
      "argv": ["python3", "/workspace/tools/collect_timeline.py"],
      "timeout_seconds": 300
    }
  ],
  "reviewer": {
    "argv": ["reviewer-cli", "--json"],
    "timeout_seconds": 120
  }
}
```

The budget is explicit in the file. Use `balanced` when the user has not chosen
between `fast`, `balanced`, and `thorough`. `project_paths` must not overlap.
`environment_root` is a user-owned isolated copy, not a host Python, CUDA, or
driver directory.

Start the controller:

```bash
python3 scripts/workload_controller.py run \
  --control /workspace/control.json \
  --run-dir /workspace/workload_run
```

The probe command receives `CUDA_OPTIMIZER_OUTPUT`,
`CUDA_OPTIMIZER_RUN_DIR`, and `CUDA_OPTIMIZER_PROJECT_ROOT`. It writes one strict
object to `CUDA_OPTIMIZER_OUTPUT`:

```json
{
  "schema_version": "cuda-workload-optimizer/probe-v1",
  "probe_id": "timeline",
  "kind": "timeline",
  "status": "ok",
  "metrics": {
    "gpu_busy_pct": 43.2,
    "cpu_busy_pct": 91.0,
    "data_wait_pct": 38.5
  },
  "issues": [],
  "artifacts": [
    {
      "name": "raw/timeline.sqlite",
      "sha256": "0f4a3a2e445d6f40f31071f0f0892b64debd2bd8242a2ce3b63d508a878ad19d"
    }
  ]
}
```

Missing metrics stay missing; do not write zero for evidence that was not
collected. Supported percentage fields are listed in
`references/workload_diagnosis_policy.json` and the control design. The
controller stores bounded stdout and stderr separately from the probe facts.

## 2. Register one bounded change

Read `workload_run/diagnosis.json`. If data wait dominates, a ChangeSet can look
like this:

```json
{
  "schema_version": "cuda-workload-optimizer/change-v1",
  "id": "round-1-dataloader-workers",
  "hypothesis": "The input pipeline leaves the GPU idle between batches.",
  "diagnosis_ids": ["cpu_data:data-wait"],
  "scope": "project",
  "candidate": {"name": "dataloader-workers-8", "revision": "worktree"},
  "paths": ["configs/serve.json"],
  "commands": [
    ["python3", "-m", "unittest", "tests.test_serve_config"]
  ],
  "rollback": "restore_frozen_snapshot",
  "expected_metrics": [
    "data_wait_pct",
    "gpu_busy_pct",
    "p50_latency_ms"
  ]
}
```

Register before editing:

```bash
python3 scripts/workload_controller.py register-change \
  --control /workspace/control.json \
  --run-dir /workspace/workload_run \
  --change-set /workspace/change.json
```

Codex now edits only `configs/serve.json`. For dependency or container changes,
set `scope` to `isolated_environment`; paths then resolve inside the declared
environment copy. `host` is not a valid scope.

## 3. Review and evaluate

The optional reviewer receives a digest-bound request on stdin. It may return:

```json
{
  "schema_version": "cuda-workload-optimizer/review-v1",
  "request_digest": "the-digest-from-the-request",
  "verdict": "challenge",
  "concerns": [
    {
      "severity": "medium",
      "category": "experiment",
      "message": "Repeat after a fixed warmup window."
    }
  ],
  "suggested_experiments": ["Use the same request trace for both roles."]
}
```

Only `support`, `challenge`, and `insufficient` are valid verdicts. The reviewer
cannot return commands or approve promotion. The local process still has the
current user's OS permissions; put it in a read-only sandbox when that matters.

Evaluate the registered candidate:

```bash
python3 scripts/workload_controller.py evaluate \
  --run-dir /workspace/workload_run
```

The controller checks the actual diff, runs the correctness argv, reviews the
proposal when configured, and evaluates randomized baseline/candidate pairs on
the frozen workload. Promotion requires a confirmed primary-metric win and all
constraints passing. Otherwise the frozen project or environment snapshot is
restored.

Inspect or resume without replaying completed stages:

```bash
python3 scripts/workload_controller.py status --run-dir /workspace/workload_run
python3 scripts/workload_controller.py resume --run-dir /workspace/workload_run
```

Host-level opportunities belong in `host_recommendations.md`. Include evidence,
risk, expected effect, and a manual verification command; never apply them from
the workload controller.
