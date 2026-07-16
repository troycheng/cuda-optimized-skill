# CUDA Kernel Optimizer v2.2 Dual-Loop Design

**Date:** 2026-07-16

**Status:** Approved design

**Target:** `skills/cuda-kernel-optimizer`

## Summary

Version 2.2 turns the existing kernel-oriented branch-and-select workflow into
a budget-aware dual-loop optimizer:

- an inner loop uses correctness gates and statistically sound paired
  measurements to screen CUDA, CUTLASS, and Triton candidates; and
- an outer loop validates shortlisted candidates against a real workload that
  the user explicitly supplies.

The real workload is the only surface that can establish an end-to-end win.
Kernel benchmarks remain valuable evidence, but a kernel-only result must not
be presented as an application-level improvement.

## Goals

1. Make performance claims reproducible and resistant to measurement noise.
2. Combine fast kernel exploration with real PyTorch, vLLM, or other
   user-provided workload validation.
3. Give users selectable compute budgets with a balanced default.
4. Preserve raw evidence, compiler provenance, decisions, and checkpoints.
5. Degrade honestly when profiling counters are unavailable.
6. Retain the current CUDA, CUTLASS, and Triton workflow and RTX 5090 coverage.

## Non-goals

- Discovering, downloading, or inventing a representative real workload.
- Modifying workload weights or acceptance targets without user input.
- Claiming an end-to-end win from a microbenchmark alone.
- Automatically escalating privileges to access GPU performance counters.
- Replacing user-defined objectives with a hidden composite score.
- Building a general distributed experiment scheduler in this release.

## Confirmed product decisions

- Use a dual-loop funnel: trusted kernel screening followed by real-workload
  validation.
- Require the user to provide the real workload.
- Allow a clearly labeled `kernel-only` mode when no workload is provided.
- Offer `quick`, `balanced`, `thorough`, and `custom` budgets.
- Make `balanced` the default with a three-hour wall-clock limit.
- Use adaptive paired A/B measurements and a 95% bootstrap confidence
  interval.
- Default `min_effect_pct` to 0.5 when the user does not specify it.
- Preserve inconclusive results instead of forcing a champion.
- Use the workload's primary KPI for promotion and treat all declared
  constraints as hard gates.

## Inputs

The existing baseline kernel, reference implementation, dimensions, toolchain,
and iteration inputs remain. Version 2.2 adds an optional user-owned workload
input and a budget selection.

### Workload forms

Support the following mutually exclusive forms:

1. `--workload ./workload.py`: a structured Python adapter.
2. `--workload-cmd "./run_benchmark.sh"`: an existing executable command.
3. `--workload-manifest ./workload.json`: a fixed user-defined workload
   matrix.

Only one form may be active in a run. If more than one is supplied, preflight
fails with a precise error.

The structured adapter declares its objective through `metrics()`. The command
form requires `--objective ./objective.json`. A manifest either embeds the same
objective object or uses `--objective`; conflicting embedded and external
objectives fail preflight.

### Workload ownership

The skill may instrument, measure, and freeze a snapshot derived from the
user-provided workload. It must not infer or create a workload when none is
provided. Captured shapes and weights are derivative evidence, not a new source
of truth.

### Structured adapter contract

The canonical `workload.py` form exposes these operations:

```python
def prepare(candidate): ...
def validate(candidate): ...
def benchmark(candidate): ...
def metrics(): ...
def cleanup(): ...
```

`candidate` identifies the baseline or candidate artifact being evaluated.
`benchmark` returns raw observations. `metrics` declares the primary KPI,
direction, minimum effect, and hard constraints. The runner invokes `cleanup`
after success, failure, timeout, or interruption.

The command and manifest forms are normalized internally to the same lifecycle
and result schema.

### Objective contract

A normalized objective has this shape:

```json
{
  "primary_metric": {
    "name": "p50_latency_ms",
    "direction": "lower"
  },
  "min_effect_pct": 1.0,
  "constraints": [
    {
      "name": "p99_latency_ms",
      "max_regression_pct": 0.5
    },
    {
      "name": "peak_memory_mb",
      "max_regression_pct": 0.0
    }
  ]
}
```

The primary KPI decides promotion. Every constraint must pass. When the user
declares multiple optimization objectives instead of one primary KPI, the
decision engine maintains a Pareto frontier and does not silently collapse the
objectives into a weighted score.

## Budget model

Budgets are hard wall-clock limits and search policies, not correctness levels.
No budget may skip baseline preflight or required correctness checks.

| Preset | Wall time | Inner search | Paired blocks | Outer-loop policy |
|---|---:|---|---:|---|
| `quick` | 45 minutes | 4 candidates per round, at most 2 rounds | 20-50 | Validate the final candidate on up to 3 user-selected cases; otherwise run the opaque workload once |
| `balanced` | 3 hours | 8 candidates per round, at most 4 rounds | 20-100 | Validate the top 2 candidates per round on 5-10 user-selected cases; otherwise run the opaque workload once per candidate |
| `thorough` | 10 hours | 12-16 candidates per round, at most 8 rounds | 30-200 | Validate the top 3 per round on the full workload; run full sanitizer gates on finalists |
| `custom` | User-defined | User-defined | User-defined | User-defined within safety gates |

Supported overrides include wall time, branch count, round count, maximum paired
blocks, confidence level, and the number of candidates sent to the outer loop.
Overrides are recorded in the manifest.

Reserve the final five minutes for cleanup and checkpoint persistence. Before
starting a stage, require its conservative runtime estimate to fit before that
reserve. When the execution deadline is reached, stop the active stage at its
defined safe cancellation boundary, classify incomplete evidence as
`inconclusive`, clean up, and write the checkpoint within the reserved time.

## Architecture

### 1. Input and baseline freezer

- Run the existing environment and kernel/reference preflight.
- Validate the workload adapter or command contract when present.
- Normalize the objective and budget.
- Copy or hash all input files and configuration.
- Record toolchain, GPU, source, workload, and environment hashes.
- Establish independent kernel and real-workload baselines.

The frozen workload configuration must not change during a run. A changed
configuration starts a new run or requires an explicit fork of the checkpoint.

### 2. Budget scheduler

- Track elapsed and estimated remaining time.
- Allocate work across compilation, paired sampling, profiling, sanitizers, and
  outer-loop validation.
- Stop new work early enough to preserve the cleanup and checkpoint reserve.
- Preserve deterministic checkpoint boundaries after each completed stage.

Correctness and result persistence are non-negotiable. Search breadth, sample
count, profiler depth, and outer-loop frequency are budget-controlled.

### 3. Inner-loop experiment engine

The existing evidence-guided method selection and branch generation remain the
candidate source. Each candidate passes through a staged funnel:

1. compile and capture compiler artifacts;
2. run single- and multi-seed correctness;
3. run targeted Compute Sanitizer checks when the change touches memory,
   synchronization, asynchronous copies, or warp specialization;
4. collect paired baseline/candidate samples;
5. classify the statistical result; and
6. send only confirmed winners to the outer loop.

Candidate generation may use staged elimination so clearly invalid or losing
candidates do not consume the full budget.

### 4. Paired measurement engine

The unit of observation is a baseline/candidate pair. Each block randomizes the
order as `AB` or `BA`, performs equivalent warm-up and reset operations, and
records both raw measurements plus environment telemetry.

After each batch, compute the paired percentage improvement distribution and a
95% bootstrap confidence interval. Ranking and global promotion use the same
primary statistic. Version 2.2 removes the current mismatch where branch
selection prefers `median_ms` while global state promotion reads `average_ms`.

Default classification:

- `confirmed_win`: the complete confidence interval is at or beyond
  `min_effect_pct` in the favorable direction;
- `confirmed_loss`: the complete confidence interval is at or beyond the same
  threshold in the unfavorable direction; and
- `inconclusive`: neither condition holds when the sample or time limit is
  reached.

The user may set `min_effect_pct`. If omitted, use 0.5%. Additional sampling is
allowed only while it fits the selected budget.

### 5. Outer-loop workload evaluator

The evaluator compares the current best and the shortlisted candidate with the
same frozen user workload. It randomizes execution order and follows the
adapter's cache, process-restart, setup, and cleanup policy.

The outer-loop result can change the next round's search evidence, but it cannot
rewrite the user workload, KPI, weights, or constraints. Only a statistically
confirmed workload improvement with every constraint passing can replace the
global best.

### 6. Decision engine

Every evaluated candidate receives one terminal status:

- `rejected_compile`
- `rejected_correctness`
- `rejected_constraint`
- `confirmed_loss`
- `inconclusive`
- `kernel_only_win`
- `end_to_end_win`
- `pareto_frontier`

`kernel_only_win` means trusted kernel evidence exists but the workload is
absent or has not shown a confirmed improvement. It is never rendered as an
end-to-end success.

If a kernel improves but the workload does not, retain the candidate and mark
the result as Amdahl-limited or workload-inconclusive according to the measured
evidence. If objectives trade off, retain the candidate on the Pareto frontier.

### 7. Artifact and checkpoint store

Use a versioned run schema:

```text
run_<timestamp>/
├── manifest.json
├── workload/
├── baseline/
├── candidates/
│   └── <candidate_id>/
│       ├── source/
│       ├── correctness.json
│       ├── paired_samples.jsonl
│       ├── statistics.json
│       ├── workload_result.json
│       ├── profiler/
│       ├── compiler_evidence/
│       └── decision.json
├── frontier.json
├── checkpoint.json
└── summary.md
```

`manifest.json` includes `schema_version`, normalized inputs, hashes, GPU and
tool versions, budget, and decision thresholds. Raw paired observations are
append-only. Derived statistics must be reproducible from those observations.

Compiler evidence includes available source, TTIR/TTGIR or equivalent IR, LLVM
IR, PTX, SASS, binary hash, and compilation options. Missing stages are recorded
as unavailable rather than fabricated.

Checkpoint recovery accepts the exact frozen input hashes and compatible schema
version. Schema changes require an explicit migration; otherwise recovery fails
with an actionable compatibility error.

## Error handling

- Compilation failure rejects only the candidate and preserves logs.
- Correctness failure rejects the candidate immediately and preserves the
  failing seed and inputs needed to reproduce it.
- A transient workload failure may be retried twice. Continued instability
  terminates that candidate without using partial data to declare a win.
- A telemetry violation invalidates the entire paired block. The runner never
  removes an individual slow observation after seeing the result.
- Nsight Compute counter denial records the command, return code, and
  `ERR_NVGPUCTRPERM`, then continues with timing, workload, source, and compiler
  evidence. It never adds privileges automatically.
- Interruption completes safe cleanup, atomically writes the checkpoint, and
  exits with the run resumable.
- Budget exhaustion cancels the active validation at its defined safe boundary,
  persists completed evidence, and classifies unresolved candidates as
  `inconclusive` before the hard deadline.

## Summary contract

The final summary separates:

1. input, workload, environment, and budget;
2. kernel-level results and confidence intervals;
3. real-workload results and confidence intervals;
4. correctness, sanitizer, profiler, and compiler evidence;
5. promoted best, frontier candidates, and rejected candidates; and
6. unresolved or degraded evidence.

The headline may say `end_to_end_win` only when the outer-loop promotion gate
passes. A run without a user workload prominently states `kernel-only`.

## Test strategy

### Unit tests

- Synthetic paired distributions covering confirmed win, confirmed loss, and
  inconclusive outcomes.
- AB/BA randomization and reproducible seeds.
- Confidence interval, direction, minimum-effect, and constraint semantics.
- Budget estimation, stop-before-start, and hard-limit behavior.
- Pareto frontier and terminal decision classification.
- Schema serialization and checkpoint compatibility.

### Contract and fault-injection tests

- Structured adapter, command, and manifest workload forms.
- Conflicting workload input flags.
- Compilation failure, wrong output, flaky workload, timeout, and cleanup.
- NCU unavailable and `ERR_NVGPUCTRPERM` degradation.
- Telemetry-contaminated paired blocks.
- Interruption and resume from every stable checkpoint boundary.

### Controlled GPU acceptance

Retain the RTX 5090 CUDA, CUTLASS, and Triton fixtures. Add gold scenarios with:

- a deliberate and repeatable improvement;
- byte-identical or functionally identical candidates that must not be called a
  win;
- a difference inside the noise band that must remain inconclusive; and
- incorrect or synchronization-unsafe candidates that correctness or sanitizer
  gates reject.

Run the controlled matrix on the current toolchain lane and retain the existing
compatibility lane coverage.

### Real-workload acceptance

Use a user-provided isolated PyTorch or vLLM workload on the RTX 5090 host. The
release passes only when:

- the workload adapter is frozen and reproducible;
- identical implementations are not reported as improvements;
- noise-band differences remain inconclusive;
- a kernel-only improvement is not described as an end-to-end win;
- a true workload improvement can be recomputed from raw samples;
- interrupted work resumes without repeating completed stages or losing data;
  and
- each preset respects its hard wall-clock policy.

## Acceptance criteria

1. `balanced` is the default and records a three-hour hard limit.
2. Every performance comparison uses the unified paired decision engine.
3. Branch ranking and global promotion use the same statistic.
4. Every win includes raw observations, a confidence interval, and a declared
   minimum effect.
5. Full mode requires a user-provided workload; absent workload produces only
   a labeled kernel-only result.
6. Only a workload-confirmed candidate can become an end-to-end best.
7. Correctness and configured constraints are never relaxed by budget choice.
8. NCU permission failures degrade without privilege changes or fabricated
   metrics.
9. Checkpoints resume safely with matching frozen inputs and schema.
10. CPU tests, controlled RTX 5090 tests, and a user-provided real-workload test
    pass before release.
