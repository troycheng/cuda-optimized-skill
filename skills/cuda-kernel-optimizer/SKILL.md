---
name: cuda-kernel-optimizer
description: "Use when optimizing, tuning, or profiling a CUDA, CUTLASS, or Triton kernel against a reference implementation, especially for kernel benchmarking, real-workload validation, Nsight Compute, branch exploration, or SASS verification."
---

# CUDA Kernel and Workload Optimizer (V2.2)

## Principle

Optimize in two evidence layers:

1. The inner loop proves that a candidate is correct and faster than the current
   kernel with paired measurements.
2. When the user supplies a real workload, the outer loop proves that the kernel
   improvement survives the user's primary KPI and constraints.

A faster microbenchmark is not automatically an end-to-end win. Promotion is a
decision backed by durable evidence, not the lowest observed sample.

## Required and optional inputs

Require these before setup:

- baseline CUDA/CUTLASS `.cu` or Triton `.py` kernel;
- Python reference exposing `reference(**kwargs)`;
- kernel dimensions as a JSON object;
- optional user-provided workload owned by the user.

The reference is always required for correctness. A user-provided workload is
required for any `end_to_end_win` claim. Never infer, download, or manufacture a
representative workload. Without one, run in kernel-only mode and state that
scope explicitly.

Accept exactly one workload form:

- `--workload PATH`: Python adapter implementing `prepare`, `validate`,
  `benchmark`, `metrics`, and `cleanup`;
- `--workload-cmd 'COMMAND ...' --objective objective.json`: command parsed
  without a shell and an external objective;
- `--workload-manifest manifest.json`: strict manifest describing a Python or
  command workload, objective, and cases.

Start a Python adapter from `templates/workload.py`. Validate objectives against
`templates/objective.schema.json`. Treat adapter source, declared local
dependencies, objective, cases, baseline, reference, dimensions, backend,
budget, confidence, and minimum effect as frozen run inputs.

## Compute budget

Ask the user for a preset when it materially affects cost. If they do not choose,
use `balanced` by default.

| Preset | Max time | Branches | Rounds | Pairs min-max | Outer candidates | Cases | Sanitizer |
|---|---:|---:|---:|---:|---:|---:|---|
| `quick` | 2700 s | 4 | 2 | 20-50 | 1 | 3 | targeted |
| `balanced` | 10800 s | 8 | 4 | 20-100 | 2 | 10 | targeted |
| `thorough` | 36000 s | 16 | 8 | 30-200 | 3 | unlimited | full |

Use `--budget custom` only with all required explicit limits. A budget deadline
stops admission of new work and preserves a resumable checkpoint; it does not
turn partial evidence into a win.

## Setup and resume

Run setup once:

```bash
python3 <skill>/scripts/orchestrate.py setup \
  --baseline ./kernel.cu \
  --ref ./ref.py \
  --dims '{"M":4096,"N":4096,"K":4096}' \
  --budget balanced \
  --workload ./workload.py
```

Omit the workload option for kernel-only mode. Substitute either
`--workload-cmd ... --objective ...` or `--workload-manifest ...` when that is
the user's chosen input form.

Setup checks the environment and input contracts, freezes a manifest, seeds the
baseline, and writes `checkpoint.json`. Do not edit frozen inputs inside a run.
After interruption, validate and inspect the next unfinished stage with:

```bash
python3 <skill>/scripts/orchestrate.py resume --run-dir ./run_YYYYMMDD_HHMMSS
```

Resume never replays a completed stage. Follow the reported `next_stage` and
`next_iteration`.

## Dual-loop workflow

For each admitted round:

1. Read `state.json`, `checkpoint.json`, the current best source, and available
   profiler/sanitizer/compiler evidence.
2. When counters are readable, profile the current best and use evidenced
   compute, memory, and latency gaps. Read `references/optimization_catalog.md`
   and `references/ncu_metrics_guide.md` before selecting methods.
3. Record selected methods and rejected higher-priority alternatives in
   `methods.json` and `analysis.md`; do not persist hidden reasoning.
4. Generate the budgeted branch variants under `itervN/branches/`. Keep the
   method intent constant while varying tiles, stages, warps, or implementation
   details.
5. Run the deterministic close step:

   ```bash
   python3 <skill>/scripts/orchestrate.py close-iter \
     --run-dir ./run_YYYYMMDD_HHMMSS --iter N
   ```

6. Inner loop gates candidates through reference correctness, sanitizer policy,
   compiler/SASS provenance, and telemetry-gated randomized paired AB/BA timing.
7. Only `confirmed_win` kernel evidence may enter the outer shortlist. Preserve
   `confirmed_loss`, `inconclusive`, invalid, and failed candidates without
   promoting them.
8. In full mode, evaluate shortlisted candidates with paired baseline/candidate
   observations on the frozen user workload. Enforce the primary metric's
   `min_effect_pct` and every constraint's `max_regression_pct`.
9. Apply the terminal decision atomically. Continue to the next round only when
   the checkpoint and remaining budget permit it.

Finalize only after the decision stage is complete:

```bash
python3 <skill>/scripts/orchestrate.py finalize \
  --run-dir ./run_YYYYMMDD_HHMMSS
```

## Paired verdict and promotion rules

Use randomized AB/BA pairs and confidence intervals. A valid promotion requires
finite statistics, enough valid pairs, the configured confidence, and a lower
confidence bound meeting the minimum practical effect. Invalid telemetry blocks
remain recorded but do not count as valid pairs.

- `confirmed_win`: candidate clears correctness and statistical gates.
- `confirmed_loss`: evidence supports no acceptable win.
- `inconclusive`: evidence is insufficient or the confidence interval crosses a
  decision boundary. Spend more budget only if admission allows; never promote.

Terminal outcomes:

- `kernel_only_win`: a confirmed kernel win in kernel-only mode. It says nothing
  about application throughput or latency.
- `end_to_end_win`: full mode only; both kernel evidence and the user workload's
  primary KPI are confirmed wins and all constraints pass.
- failure, loss, timeout, or inconclusive outcomes keep the current best.

Do not update `best_file` from an average, one sample, a profiler estimate, or an
inner-loop win in full mode. `decision.json` is the promotion authority.

## Degraded and failure behavior

- If target profiling reports `ERR_NVGPUCTRPERM`, record the exact command,
  return code, and log; mark counter coverage unavailable and continue with
  correctness, paired timing, source, compiler, sanitizer, and SASS evidence.
  Never add privileges or change driver policy without explicit authorization.
- `ncu --query-metrics` proves metric metadata exists, not that the target may
  read hardware counters.
- Missing profiler metrics remain missing. They cannot prove `near_peak` or a
  measured Roofline classification.
- If the sanitizer policy selects no applicable tool, record `not_applicable`.
  If a selected tool cannot provide full coverage but the policy permits
  continuation, record `degraded` prominently. Never label degraded coverage as
  passed. Read `references/sanitizer_policy.json` for the exact routing policy.
- A failed correctness, sanitizer, compiler-provenance, or SASS gate cannot be
  rescued by good timing.
- Bound retries and retain the failure artifacts. Do not loop indefinitely.
- Reject non-finite metrics, malformed workload output, changed frozen inputs,
  unsafe symlinks, and candidate/source hash drift.

## Output contract

Representative durable layout:

```text
run_YYYYMMDD_HHMMSS/
├── manifest.json                 # frozen inputs and input_hash
├── state.json                    # candidates, best evidence, history
├── checkpoint.json               # resumable stage boundary
├── env.json                      # toolchain and profiler capability
├── workload/spec.json            # frozen workload snapshot or null
├── baseline/
│   ├── kernel.{cu,py}
│   └── bench.json
├── itervN/
│   ├── analysis.md
│   ├── methods.json
│   ├── branches/...
│   ├── sanitizer.json
│   ├── sanitizer/*.json
│   ├── sass_check.json
│   ├── compiler_evidence/manifest.json
│   ├── branches/<candidate>/paired_samples.jsonl
│   ├── workload/<candidate-id>/paired_samples.jsonl
│   ├── decision.json             # authoritative promotion decision
│   └── *.ncu.log                 # success or exact degraded failure
└── summary.md                    # separate kernel and workload conclusions
```

Some backend- or outcome-specific artifacts are optional. Never claim an
optional artifact exists without checking it. Raw `paired_samples.jsonl`, the
frozen objective, candidate hashes, classifier configuration, and
`decision.json` make the reported confidence result independently recomputable.

The final summary must state, in this order: terminal result and budget; frozen
inputs/environment; kernel estimate, confidence interval, pairs, correctness,
and SASS status; real-workload KPI and constraints or the absence of a workload;
profiler/sanitizer/compiler coverage; candidates; raw artifact paths; resume
status.

## References to read on demand

- `references/compatibility.md`: supported versions, public APIs, architecture
  routing, and RTX 5090/SM120 facts.
- `references/optimization_catalog.md`: method triggers, skip rules, and
  combination constraints.
- `references/ncu_metrics_guide.md`: profiler metric interpretation.
- `references/sanitizer_policy.json`: targeted/full sanitizer routing.
- `references/sass_signatures.json`: instruction signatures.
- `templates/objective.schema.json`: strict workload objective schema.
- `templates/workload.py`: user-owned Python workload starter.
- `templates/iteration_report.md`: per-round decision record.
- `examples/walkthrough.md`: annotated kernel-only and full-mode walkthrough.
