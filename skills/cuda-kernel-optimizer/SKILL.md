---
name: cuda-kernel-optimizer
description: "Use when optimizing, tuning, diagnosing, or profiling CUDA, CUTLASS, Triton, PyTorch, vLLM, TensorRT-LLM, or another GPU workload; when assessing an existing NCU report; or when the workload, correctness reference, benchmark, profiler, or target environment is incomplete."
---

# CUDA Kernel and Workload Optimizer
## Operating rule

Optimize the user's real objective against a correctness reference. Measure on
the target path, keep raw evidence, and retain only verified changes. A faster
kernel is not an end-to-end win unless the user-provided workload also improves.

Do not assume the bottleneck is inside a kernel. Check framework scheduling,
CPU/data work, transfers, communication, I/O, allocator behavior, serving state,
and the environment when the supplied workload crosses those layers.

## Route before loading details

Read only the reference required by the current route. Do not load the whole
catalog or all evidence protocols.

| Situation | First action | Read on demand |
|---|---|---|
| Missing workload, reference, benchmark, or environment | Inventory gaps, then run capability readiness before baseline | `references/environment_readiness.md` |
| Single CUDA, CUTLASS, or Triton kernel | Establish correctness and paired kernel timing | `references/performance_iteration.md` |
| Bottleneck unknown across a full workload | Run workload diagnosis before choosing a mutation | `examples/workload-controller.md` |
| Serving KPI validation | Freeze strata, identities, guards, and experiment design | `references/serving_evidence_protocol.md` |
| Time-varying serving state | Check comparability before interpreting speed | `references/nonstationary_serving_evidence.md` |
| Long run, many candidates, or resume after interruption | Freeze the run and let the Controller own state | `references/long_running_control.md` |
| Existing `.ncu-rep` only | Analyze read-only; do not launch the original target | `references/ncu_metrics_guide.md` |
| Compare GPU software-stack versions | Freeze one variable and rebuild derived artifacts per stack | `references/version_stack_audit.md` |
| Architecture, API, or tool version uncertain | Query bundled knowledge, then verify locally or with primary sources | `references/offline_knowledge.md` |
| Major direction choice or reasoning plateau | Use optional search or independent challenge | `references/research_augmentation.md` |

## Establish the claim ceiling
Inventory these facts before mutation:

- target source or profiler artifact;
- correctness reference and representative cases;
- reproducible build/import command and isolated environment;
- stable kernel benchmark with warmup and paired raw samples;
- user-approved real workload and objective;
- serving experiment when a serving claim is required;
- allowed project paths, constraints, and host boundaries.

Create a small JSON inventory with `requested_claim` set to `kernel`, `workload`,
or `serving`. `source_available` means the current task can read the source. Use
`python3 <skill>/scripts/readiness.py --help` and report missing foundations.

Before a new baseline, create a closed readiness contract and use Controller
`control-v2`. Run foundation capabilities before workload capabilities. If a
`required` item fails, return `readiness_action`; do not start the baseline or
high-cost profiling. Diagnostic failure may continue only as a recorded lower
evidence layer.

Only missing items required by the requested claim should be reported. If only
source is available, provide static hypotheses and a foundation plan;
do not call them optimization results. A kernel benchmark supports only a kernel
claim. A representative workload is required for an end-to-end claim. Never
download, invent, or silently substitute a workload.

The only automatic repair is explicitly authorized, hash-locked `isolated_pip`
inside the declared environment. Host changes remain `recommend_only`. Run
readiness yourself; do not hand internal commands to the user.

## Choose the budget
Use `balanced` by default. Respect a user-selected budget.

| Budget | Wall-time ceiling | Intended use |
|---|---:|---|
| `quick` | 45 minutes | Triage and narrow candidates |
| `balanced` | 3 hours | Default optimization and validation |
| `thorough` | 10 hours | Broader search and deeper evidence |

The ceiling is not a target. Stop earlier when evidence is conclusive, no
eligible direction remains, or the claim ceiling blocks promotion.

## Core workflow
1. Freeze the objective, constraints, representative cases, artifacts, mutation
   roots, environment, budget, and stability policy with
   `scripts/workload_contract.py`.
2. Confirm correctness, collect baseline-only pairs, and run
   `scripts/stability_calibration.py`. Admit candidates only while the
   Controller-attested environment state is `green`.
3. Profile the full path at the highest available claim layer. With `control-v2`
   active diagnosis, build the hash-bound context, state competing falsifiable
   hypotheses, and let the Controller select and execute one frozen evidence action.
   Update hypotheses from its outcome and repeat only while the profile budget allows.
4. Treat outcome support/opposition, current readiness, request history, and evidence
   digests as Controller facts. Do not reinterpret an outcome, repeat an equivalent
   request, accept a partial interrupted result, or make exclusive explanations pass.
5. Admit only directions with measured headroom. A direction experiment may run in a
   Controller-created project copy, which is cooperative isolation, not an OS sandbox.
   Read `references/direction_admission.md` for stop and reopen rules.
6. Query a few capability cards by exact task, architecture, observed signals,
   and available evidence. Use the hard context budget and load only returned
   playbooks:

   ```bash
   python3 <skill>/scripts/capability_query.py --help
   ```

7. State one falsifiable hypothesis, one bounded change, the expected metric
   movement, and the condition that would reject it.
8. Evaluate in order: correctness, paired timing, required constraints, then
   compiler/profiler evidence. Diagnostics explain a result; they do not promote it.
9. Keep the candidate only when its permitted claim passes. Restore or isolate
   rejected changes and record why they failed.
10. Resume from durable state. Do not repeat failed mechanisms unless new
    evidence changes the bottleneck or applicability.

Run script help rather than loading command inventories into context:

```bash
python3 <skill>/scripts/orchestrate.py --help
python3 <skill>/scripts/workload_controller.py --help
python3 <skill>/scripts/evidence.py --help
```

## Long-running control
For a multi-candidate or resumable run, read
`references/long_running_control.md`. Keep the Planner and Controller separate:

- `scripts/evidence_controller.py` runs allowlisted evidence adapters and seals
  their normalized output;
- `scripts/planner_boundary.py` admits a preregistered candidate only when its
  summary, capability query, execution gates, evidence age, and time match;
- the append-only run ledger replays calibrations, audits, admissions, and
  candidate results before work resumes.

The frozen `audit_every_candidates` value limits registrations between baseline
audits. Only a Controller-attested calibration or audit may move the persistent
run into or back to `EXPLORING`. A changed workload, source, objective, or
environment requires a new contract and a new ledger.

## Workload and kernel decisions

Use `scripts/workload_diagnosis.py` or the workload controller to classify
`kernel`, `framework`, `cpu_data`, `transfer`, `communication`, `io`,
`environment`, or `mixed`. Let the measured share and objective determine the
direction; do not apply a universal method ranking across claim layers.

Codex owns analysis, code changes, and decisions. An optional local reviewer is
advisory only and communicates through JSON stdin/stdout. Reviewers and external
models never become promotion authorities.

For kernel candidates, use the real stage order documented by `orchestrate.py`.
Correctness and paired timing are promotion gates; sanitizer, SASS, compiler,
NCU, and Nsys evidence are required only when the frozen task or claim requires
them. Report an unavailable profiler exactly, including `ERR_NVGPUCTRPERM`, and
continue only at a lower valid evidence layer.

## Formal and nonstationary evidence

For shared-host or formal serving claims, read
`references/evidence_automation.md` and run the guard, seal, audit, and decision
workflow. Keep `performance_verdict` separate from `evidence_integrity`; missing,
stale, contaminated, contradictory, or identity-invalid evidence fails closed.

When load, queue depth, cache state, or another declared condition moves over
time, read `references/nonstationary_serving_evidence.md`. A
`comparable_paired_state` verdict permits later performance interpretation; it
does not claim a win. `inconclusive_nonstationary` requires a new or recollected
experiment.

## Knowledge and external research

Use `scripts/knowledge_query.py` for the legacy method catalog and
`scripts/capability_query.py` for evidence-bound playbooks. Return only
architecture-compatible cards.

```bash
python3 <skill>/scripts/knowledge_query.py --arch sm_120 --layer kernel --bottleneck gemm --limit 5
```

Exact SM capabilities are required; never inherit features by numeric ordering.
Cards without a matching observed signal remain `unverified`; do not admit a
direction from registry order alone. Historical speedup ranges are context, not
expected gain.

The bundled snapshot must work offline. Read `references/offline_knowledge.md`
and `references/compatibility.md` when versions or capabilities matter. Treat
stale or mismatched facts as unverified and probe locally.

External search and multi-model challenge are optional. Read
`references/research_augmentation.md` before sending any evidence outside the
environment. Use primary sources, redact private material, obtain independent
answers before cross-critique, preserve disagreement, and adjudicate locally.
Network or provider failure must fall back to the offline workflow.

Read `references/optimizer_limits.md` when the target behavior depends on
undocumented hardware, proprietary semantics, or missing workload facts.

For a Triton, TensorRT, CUDA, framework, or container upgrade, read
`references/version_stack_audit.md` and validate the frozen comparison with
`python3 <skill>/scripts/version_audit.py --help` before timing.

## Modification and host boundary

Modify only declared project paths and user-approved isolated environments.
This skill does not provide an OS sandbox. Treat driver, GPU counter permission,
clock, power, service, container runtime, kernel module, and other host changes
as `recommend_only` unless the user separately authorizes them.

Do not expose credentials, proprietary source, private inputs, hostnames, or raw
logs to external services without explicit approval.

## Durable output

Preserve the applicable artifacts rather than a prose-only conclusion:

- environment and objective identity;
- readiness report and claim ceiling;
- manifest, checkpoint, and candidate lineage;
- profiler or workload diagnosis summary;
- hypotheses, bounded changes, raw paired samples, and constraints;
- `decision.json`, evidence integrity, and `summary.md`;
- research sources, critiques, and unresolved disagreements when used.

Report separately: verified result, strongest supported claim, rejected
candidates, missing evidence, unchanged host recommendations, and the next
highest-value action.

## Reference index

- Compatibility and tools: `references/compatibility.md`
- Offline sources and query rules: `references/offline_knowledge.md`
- Environment preparation: `references/environment_readiness.md`
- Optimization limits: `references/optimizer_limits.md`
- Kernel method catalog: `references/optimization_catalog.md`
- NCU metrics: `references/ncu_metrics_guide.md`
- Systems, CUTLASS/CuTe, and Triton IR: `references/systems_and_ir_coverage.md`
- Runtime and serving claims: `references/serving_evidence_protocol.md`
- Formal evidence: `references/evidence_automation.md`
- Direction admission: `references/direction_admission.md`
- Performance rounds: `references/performance_iteration.md`
- Software-stack comparisons: `references/version_stack_audit.md`
- Long-running Controller: `references/long_running_control.md`
- Nonstationary serving: `references/nonstationary_serving_evidence.md`
- External research: `references/research_augmentation.md`
