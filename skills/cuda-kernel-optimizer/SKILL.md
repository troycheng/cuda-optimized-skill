---
name: cuda-kernel-optimizer
description: "Use when optimizing, tuning, diagnosing, or profiling CUDA, CUTLASS, Triton, PyTorch, vLLM, TensorRT-LLM, or another GPU workload; when assessing an existing NCU report; or when the workload, correctness reference, benchmark, profiler, or target environment is incomplete."
---

# CUDA Kernel and Workload Optimizer

Optimize a user-provided GPU workload against its correctness reference. A
kernel result supports a kernel claim; an end-to-end claim requires a
user-approved real workload. Never download, invent, or silently substitute a
workload.

Do not assume the bottleneck is inside a kernel. Check `kernel`, `framework`,
`cpu_data`, `transfer`, `communication`, `io`, `environment`, and `mixed`
causes. Let measured headroom choose the direction; do not apply a universal method ranking across layers.

## Route before loading details

Read only the row that matches the task. Do not load the whole catalog.

| Situation | Route |
|---|---|
| Missing correctness reference, stable kernel benchmark, workload, or environment | Run `python3 <skill>/scripts/readiness.py --help`; read `references/environment_readiness.md` |
| CUDA, CUTLASS, or Triton kernel | Read `references/performance_iteration.md` |
| Bottleneck unknown in a full workload | Use `examples/workload-controller.md` |
| Serving KPI or changing serving state | Read `references/serving_evidence_protocol.md` and `references/nonstationary_serving_evidence.md` |
| Long or resumable run | Read `references/long_running_control.md` |
| Existing `.ncu-rep` only | Read `references/ncu_metrics_guide.md`; keep analysis read-only |
| Stack or architecture uncertainty | Read `references/version_stack_audit.md`, `references/compatibility.md`, and `references/offline_knowledge.md` |
| Direction choice or plateau | Read `references/research_augmentation.md` |

## Before mutation

Establish source access, correctness, representative cases, reproducible build,
paired timing, objective, allowed paths, and host boundaries. If only source is
available, return static hypotheses and an environment plan, not an optimization
result. Run readiness yourself; required failures stop baseline and profiling.
Host repair stays `recommend_only`.

Use `balanced` by default; respect `quick` or `thorough` when selected. Each
budget has a soft target and a hard ceiling. The soft target guides effort. The
hard ceiling is only a safety limit.

## Candidate gate

Before execution, declare `claim_layer`, `cheapest_falsifier`,
`estimated_cost`, `minimum_effect`, `rejection_condition`, and
`promotion_condition`. Evaluate strictly in this order:

1. static review or an independent small test;
2. build and minimum correctness;
3. short paired performance screen;
4. profiler, only when it can resolve a live uncertainty;
5. formal paired performance;
6. full service test, only for a serving claim.

A failed stage blocks every later stage. Stop when the measured effect upper
bound is below the contract threshold. Continue past the soft target when the
uncertainty still overlaps the threshold and the direction has credible
headroom. Do not continue merely to use the budget. Infrastructure repair uses
at most `min(3 minutes, 10% of hard ceiling)`.

Freeze objective, constraints, environment, paths, and stability policy with
`scripts/workload_contract.py`; calibrate with
`scripts/stability_calibration.py`. Query only a few evidence-matched cards:

```bash
python3 <skill>/scripts/capability_query.py --help
python3 <skill>/scripts/knowledge_query.py --arch sm_120 --layer kernel --bottleneck gemm --limit 5
```

Exact SM capabilities are required; never inherit features by numeric ordering.
The bundled snapshot must work offline. Read `references/optimizer_limits.md`
when facts or workload evidence are missing.

## Controller boundary

For a resumable run, let the Controller own state. `scripts/evidence_controller.py`
seals allowlisted evidence; `scripts/planner_boundary.py` admits candidates.
The frozen `audit_every_candidates` value controls baseline audits. A changed
workload, source, objective, or environment starts a new contract and ledger.

Run help instead of loading command inventories:

```bash
python3 <skill>/scripts/orchestrate.py --help
python3 <skill>/scripts/workload_controller.py --help
```

This skill does not provide an OS sandbox. Modify only declared paths and
approved isolated environments. Treat driver, GPU counter permission, clocks,
power, services, and container runtime as `recommend_only`. If NCU reports
`ERR_NVGPUCTRPERM`, record it and continue only at a lower valid evidence layer.

For formal evidence, read `references/evidence_automation.md` and
`references/direction_admission.md`. Keep `performance_verdict` separate from
`evidence_integrity`. A `comparable_paired_state` allows interpretation but is
not a win; `inconclusive_nonstationary` requires new evidence. Missing or stale
evidence must fail closed.

External search and multi-model challenge are optional. Use them only for
direction selection, a clear plateau, or final review; run independent checks in
parallel with a 180-second total wait. Use primary sources, redact private material,
preserve disagreement, and adjudicate locally. Network or provider failure falls
back to the offline workflow. External models are never promotion
authorities.

Preserve only useful durable output: readiness report and claim ceiling,
manifest, checkpoint, candidate lineage, raw paired samples, constraints,
`decision.json`, evidence integrity, and `summary.md`. Report the verified
result, strongest supported claim, rejected candidates, missing evidence, host
recommendations, and next highest-value action.
