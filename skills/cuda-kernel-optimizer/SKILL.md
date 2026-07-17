---
name: cuda-kernel-optimizer
description: "Use when optimizing, tuning, or profiling a CUDA, CUTLASS, or Triton kernel against a reference implementation, especially for kernel benchmarking, real-workload validation, Nsight Compute, existing NCU report analysis, branch exploration, SASS verification, or runtime and serving evidence."
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
  command workload, objective, and cases. It requires `kind`, `source`, and
  `cases`; use an embedded objective or external `--objective`, never both.

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

## Setup, open, and resume

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
baseline, and writes `checkpoint.json`. Setup does not profile or create branch directories.
Do not edit frozen inputs inside a run.

Before generating round N candidates, explicitly open the iteration:

```bash
python3 <skill>/scripts/orchestrate.py open-iter \
  --run-dir ./run_YYYYMMDD_HHMMSS --iter N
```

`open-iter` attempts current-best profiling, writes Roofline evidence, and
creates the budgeted branch directories. After interruption, validate and
inspect the next unfinished stage with:

```bash
python3 <skill>/scripts/orchestrate.py resume --run-dir ./run_YYYYMMDD_HHMMSS
```

Resume never replays a completed stage. Follow the reported `next_stage` and
`next_iteration`.

### Analyze an existing report

Analyze an existing `.ncu-rep` without launching its target:

```bash
python3 scripts/analyze_ncu_rep.py REPORT --out-dir OUTPUT
```

The standalone bundle records `counter_access: not_probed`; it does not prove
current counter permissions, source execution, or an end-to-end result. Source
binding and other parameters are optional. Run
`python3 scripts/analyze_ncu_rep.py --help` for details.

## Dual-loop workflow

For each admitted round:

1. Run `open-iter --run-dir ... --iter N`. It profiles the current best when
   counters are readable, computes the Roofline evidence, and prepares branch
   directories.
2. Read `state.json`, `checkpoint.json`, the current best source, and available
   profiler evidence. Read `references/optimization_catalog.md` and
   `references/ncu_metrics_guide.md` before selecting methods.
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

6. `close-iter` follows the real lifecycle: reference correctness first, then
   telemetry-gated randomized paired AB/BA timing. The sanitizer policy is
   applied to the statistically confirmed shortlist, and SASS is checked only
   for the final eligible candidate. In short: correctness → paired → sanitizer → SASS.
   Candidate profiling occurs before sanitizer and may be repeated if
   sanitizer selection changes the finalist.
7. Compiler provenance and SASS results are evidence and method-classification
   inputs, not hard promotion gates. Source/artifact identity drift still fails
   closed because the evidence would no longer describe the tested candidate.
8. In full mode, evaluate eligible candidates with paired baseline/candidate
   observations on the frozen user workload. Enforce the primary metric's
   `min_effect_pct` and every constraint's `max_regression_pct`.
9. Apply the terminal decision atomically. Continue to the next round only when
   the checkpoint and remaining budget permit it.

Finalize only after the decision stage is complete:

```bash
python3 <skill>/scripts/orchestrate.py finalize \
  --run-dir ./run_YYYYMMDD_HHMMSS
```

### Reuse completed-run strategy evidence

Strategy memory is opt-in. Record a completed v2.2 run, then request hints for
an exact manifest scope:

```bash
python3 scripts/strategy_memory.py record --memory MEMORY --run-dir RUN_DIR --out OUT
python3 scripts/strategy_memory.py suggest --memory MEMORY --manifest MANIFEST --out OUT
```

Always provide an explicit `--memory`; there is no default memory, and the
orchestrator does not call this tool. Suggestions are advisory search hints.
They never alter run state: `decision.json` owns promotion. Run
`python3 scripts/strategy_memory.py --help` for command details.

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

- `kernel_only_win`: confirms only a kernel win. It is the expected successful
  terminal label in kernel-only mode, but can also be terminal in full mode when
  workload failure/loss/inconclusive evidence prevents an end-to-end claim. It
  says nothing about application throughput or latency, and it never promotes
  the global best in full mode.
- `end_to_end_win`: full mode only; both kernel evidence and the user workload's
  primary KPI are confirmed wins and all constraints pass. This is the only
  full-mode outcome that promotes the global best.
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
- Failed correctness and sanitizer gates cannot be rescued by good timing.
  Compiler/SASS evidence must be reported honestly, but an unavailable or
  negative classification is not by itself a hard promotion veto.
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
│   ├── workload/<candidate-hash-prefix>/paired_samples.jsonl
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
- `references/systems_and_ir_coverage.md`: read only for systems-path,
  CUTLASS/CuTe, or Triton IR evidence tasks.
- `references/serving_evidence_protocol.md`: read only for runtime or serving
  evidence and claims.
- `references/sanitizer_policy.json`: targeted/full sanitizer routing.
- `references/sass_signatures.json`: instruction signatures.
- `templates/objective.schema.json`: strict workload objective schema.
- `templates/workload.py`: user-owned Python workload starter.
- `templates/iteration_report.md`: per-round decision record.
- `examples/walkthrough.md`: annotated kernel-only and full-mode walkthrough.
