---
name: cuda-kernel-optimizer
description: "Use when optimizing, tuning, or profiling a CUDA, CUTLASS, Triton, or GPU workload implementation, especially when the bottleneck may be in kernels, framework scheduling, data input, transfers, communication, I/O, or the runtime environment."
---

# CUDA Kernel and Workload Optimizer (V2.6)

## Principle

Optimize in two evidence layers:

1. The inner loop proves that a candidate is correct and faster than the current
   kernel with paired measurements.
2. When the user supplies a real workload, the outer loop proves that the kernel
   improvement survives the user's primary KPI and constraints.

A faster microbenchmark is not automatically an end-to-end win. Promotion is a
decision backed by durable evidence, not the lowest observed sample.

## Performance-first iteration gate

V2.6 keeps measurement infrastructure subordinate to performance work. Before
editing the first candidate, create one immutable lineage anchor that binds the
baseline source, environment, and prevalidated measurement paths. Each round
then binds one falsifiable hypothesis: mechanism, target metric and direction,
minimum effect, mutation scope, and budget.

After the round, classify its structured record mechanically; do not choose the
class in prose:

```bash
python3 <skill>/scripts/iteration_guard.py init \
  --registry measurement-paths.json \
  --baseline-source-sha256 <sha256> --environment-sha256 <sha256> \
  --measurement-path paired-kernel@1 --out iteration-anchor.json

python3 <skill>/scripts/iteration_guard.py check \
  --anchor iteration-anchor.json --record round-0001.json \
  --evidence-manifest evidence/manifest.json \
  --out round-0001-decision.json
```

- `candidate_evaluated` requires a source change bound to an integrity-passing
  V2.5 seal, audit, and decision closure. Inline correctness or timing claims do
  not count.
- `measurement_blocked` means a candidate was declared but the sealed closure
  did not complete.
- `infrastructure_only` means no valid candidate was evaluated. It is not an
  optimization result.
- The guard never claims `performance_gain`. It forwards a consistent
  `confirmed_win` to the existing paired-evidence and promotion gate, which
  remains the only authority for a gain. A loss or inconclusive candidate is
  useful search evidence, not a gain.

Infrastructure is capped at 15% of the round and never more than 20 minutes,
with one infrastructure repair. Exceed either cap, or complete two consecutive
`measurement_blocked`/`infrastructure_only` rounds in the same anchor-derived
hash chain, and switch only to a frozen path with a different implementation
digest. If none exists, stop the direction; do not build another runner inside
the optimization round. A registry change is a separate maintenance task. Read
`references/performance_iteration.md` for the anchor, record, evidence, fallback,
and reporting contracts. Validate inputs against
`templates/iteration_binding.schema.json`,
`templates/iteration_lineage.schema.json`,
`templates/performance_iteration.schema.json`, and
`templates/measurement_path_registry.schema.json`.

## Required and optional inputs

Choose one execution mode before setup. Do not invent a dummy kernel, reference,
or ChangeSet merely to fit a controller:

- `kernel-loop`: require a baseline CUDA/CUTLASS `.cu` or Triton `.py` kernel,
  a Python `reference(**kwargs)`, and dimensions as a JSON object;
- `workload-only`: require a user-owned runnable workload, frozen baseline and
  candidate identities, and an objective. Use this for framework, CPU/data,
  transfer, communication, I/O, or runtime changes that have no kernel mutation;
- `serving-stack`: require a frozen deployment comparison, correctness policy,
  request replay, endpoint objective, and environment protocol. TensorRT,
  Triton, CUDA, engine, backend, container, or compiler upgrades belong here.

The Python reference is mandatory only for `kernel-loop`. Every mode requires a
correctness oracle appropriate to its claim layer. A user-provided workload is
required for any `end_to_end_win` claim. Never infer, download, or manufacture a
representative workload. Without one, stop at the highest lower evidence layer
and state that scope explicitly.

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

## Workload controller

Use the V2.4.1 controller when the user has a user-provided runnable workload and
the bottleneck is not known to be inside one kernel. Codex is the primary optimizer:
it reads the evidence, writes a bounded ChangeSet, edits only the declared scope,
and interprets the result. An optional local reviewer can challenge the diagnosis
or experiment over JSON stdin/stdout, but it is advisory only. It cannot execute a
change or promote a candidate, and the protocol does not provide an OS sandbox.

The deterministic diagnosis classes are `kernel`, `framework`, `cpu_data`,
`transfer`, `communication`, `io`, `environment`, and `mixed`. Probe commands
come from the user environment and write a normalized probe artifact. This keeps
Nsight Systems, framework profilers, internal observability, and application
metrics usable without binding the skill to one profiler.

The fixed stage order is:

```text
baseline -> probes -> diagnosis -> change -> review -> evaluation -> decision
```

Start and inspect the evidence:

```bash
python3 <skill>/scripts/workload_controller.py run \
  --control ./control.json --run-dir ./workload_run
python3 <skill>/scripts/workload_controller.py status --run-dir ./workload_run
```

After reading `diagnosis.json`, write a ChangeSet, then freeze it before editing:

```bash
python3 <skill>/scripts/workload_controller.py register-change \
  --control ./control.json --run-dir ./workload_run \
  --change-set ./change.json
```

Codex may edit declared project paths or a user-owned `isolated_environment`.
The controller verifies the actual diff, binds it to the candidate, optionally
requests review, and performs paired workload evaluation. ChangeSet `commands`
must remain empty; correctness uses the workload adapter's `validate()`:

```bash
python3 <skill>/scripts/workload_controller.py evaluate \
  --run-dir ./workload_run
python3 <skill>/scripts/workload_controller.py resume \
  --run-dir ./workload_run
```

`host_policy` must be `recommend_only`. Host changes are never executed; write
them to `host_recommendations.md` with evidence and manual checks. Reject a
ChangeSet that escapes its paths, changes the frozen workload, exceeds its
deadline, or asks for host scope. See `examples/workload-controller.md` for the
complete control, probe, ChangeSet, and reviewer contracts.

The ChangeSet controller is not the entry point for an already-built external
serving stack. Use `references/serving_evidence_protocol.md` and freeze a
deployment experiment directly; compare exact runtime artifacts rather than
creating a meaningless filesystem diff.

## Formal evidence automation

Use the V2.5 evidence CLI for promotion-grade shared-host, matched-runtime, or
serving attempts. Read `references/evidence_automation.md` before creating its
schemas or artifacts. Site-owned collectors produce normalized continuous
samples; the generic skill validates them but does not claim universal GPU or
container telemetry collection.

Audit target/peer/sibling GPU identities, CPU/NUMA scope, PID/container
allowlists, memory/swap pressure, clocks, temperature, power, thermal state,
watcher-ready handshake, maximum gap, and joint clean window for each required
correctness, sanitizer, diagnostic, and timing phase:

```bash
python3 <skill>/scripts/evidence.py guard-audit \
  --policy guard-policy.json --samples guard-samples.jsonl \
  --markers phase-markers.json --out guard-audit.json
```

Freeze the complete formal schedule and statistics before timing. Formal
workload evaluation uses that exact schedule, frozen CI/pair/win settings, and
zero single-role retries; its automated aggregation is currently median-only.
Require execution-path hits for every expected case, then remove diagnostics,
rebuild, rehash, and bind the residue-free timed binary.

End every attempt in exactly one immutable state: `valid`,
`invalid_contaminated`, `invalid_identity`, `partial`, or `superseded`. Close the
evidence in order:

```bash
python3 <skill>/scripts/evidence.py seal --attempt attempt.json --out seal.json
python3 <skill>/scripts/evidence.py audit --seal seal.json --out audit.json
python3 <skill>/scripts/evidence.py decide \
  --seal seal.json --audit audit.json --out decision.json \
  --manifest evidence-manifest.json
```

Keep the performance verdict separate from `evidence_integrity`. A confirmed
win with failed integrity retains the baseline. Nsys/NCU bundles are explanatory
and `non_promotional`. Audit an imported serving run only with `audit-imported`;
write output outside its source tree. See `references/migration_v2_5.md` before
using V2.4.1 manifests, which remain compatible but legacy and unsealed.

After installation, run the CPU/static check without CUDA or network access:

```bash
python3 <skill>/scripts/self_check.py --skill-dir <skill>
```

## Layered experiment funnel

For expensive GPU or serving work, freeze a funnel before measurement. Every
stage declares one authority:

| Gate | May do | Must not do |
|---|---|---|
| `static_reject` | reject impossible or unsafe variants from source, ABI, SASS, registers, spills, barriers, or topology | claim runtime speed |
| `rank_only` | rank correct variants with an identical Event or standalone harness | promote a winner |
| `reject_only` | reject candidates with short fresh-process exact-runtime pairs | promote a winner |
| `promotion` | accept or reject using the frozen formal objective and guardrails | reuse diagnostics or quick samples as formal rows |

Generate 3-5 variants for one new mechanism when useful, batch compile them,
and apply declared static invariants before spending GPU time. Run correctness
before timing. Only the direction champion reaches formal exact-runtime testing;
only a formal exact-runtime winner reaches matched runtime and endpoint tests.
Profiler rows, diagnostic binaries, partial attempts, contaminated windows, and
quick-screen rows are never promotion evidence.

For controller-based screening, set `evaluation_gate: "reject_only"` in the
control manifest. The controller will roll back even a positive candidate and
record `reject_only_stage_cannot_promote`. Use a new, independent promotion run
with its own frozen schedule and samples for advancement.

Freeze the complete position-balanced schedule, experimental unit, estimator,
resampling unit, minimum valid pair count, no-exclusion rule, retries, and stage
authority. Independent `random.choice(AB, BA)` is not position balance. A single
valid pair can never produce `confirmed_win`. Once formal timed work starts,
do not retry one role independently or discard a slow valid row; retry only a
whole pair for a predeclared pre-measurement infrastructure failure.

## Kernel-loop compute budget

Ask the user for a preset when it materially affects cost. If they do not choose,
use `balanced` by default.

The workload controller has a separate workload budget namespace:
`fast`/`balanced`/`thorough` mean 3/5/9 paired workload blocks and are not the
kernel-loop `quick`/`balanced`/`thorough` pair counts below. Always write which
namespace is being used. A short workload budget must use `evaluation_gate:
"reject_only"` when it is part of a funnel.

| Preset | Max time | Branches | Rounds | Pairs min-max | Outer candidates | Cases | Sanitizer |
|---|---:|---:|---:|---:|---:|---:|---|
| `quick` | 2700 s | 4 | 2 | 20-50 | 1 | 3 | targeted |
| `balanced` | 10800 s | 8 | 4 | 20-100 | 2 | 10 | targeted |
| `thorough` | 36000 s | 16 | 8 | 30-200 | 3 | unlimited | full |

Use `--budget custom` only with all required explicit limits. A budget deadline
stops admission of new work, caps external-process timeouts to the remaining
budget, and preserves a resumable checkpoint; partial or late evidence never
becomes a win. A Python workload adapter runs in process and may return after
the wall-clock deadline when blocked in native code. It must bound its own
operations; use a command workload when the controller must be able to kill it.

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
python3 <skill>/scripts/analyze_ncu_rep.py REPORT --out-dir OUTPUT
```

The standalone bundle records `counter_access: not_probed`; it does not prove
current counter permissions, source execution, or an end-to-end result. Source
binding and other parameters are optional. Run
`python3 <skill>/scripts/analyze_ncu_rep.py --help` for details.

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

6. `close-iter` follows the generic lifecycle: reference correctness first,
   telemetry-gated position-balanced paired timing, sanitizer, then final SASS.
   In short: correctness → paired → sanitizer → SASS.
   For a mechanism with declared ABI, resource, barrier, spill, opcode, or
   topology invariants, add a pre-GPU static screen; failure is a hard reject.
   Candidate profiling may be repeated if sanitizer selection changes the
   finalist, but profiler measurements remain explanatory only.
7. Compiler provenance and SASS results are evidence and method-classification
   inputs and, by default, are not hard promotion gates. Generic heuristics such
   as an arbitrary register threshold cannot reject a candidate; explicitly
   declared mechanism invariants may be hard static rejection gates.
   Source/artifact identity drift always fails closed.
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
python3 <skill>/scripts/strategy_memory.py record --memory MEMORY --run-dir RUN_DIR --out OUT
python3 <skill>/scripts/strategy_memory.py suggest --memory MEMORY --manifest MANIFEST --out OUT
```

Always provide an explicit `--memory`; there is no default memory, and the
orchestrator does not call this tool. Suggestions are advisory search hints.
They never alter run state: `decision.json` owns promotion. Run
`python3 <skill>/scripts/strategy_memory.py --help` for command details.

## Paired verdict and promotion rules

Use a frozen, position-balanced AB/BA schedule and confidence intervals. A valid
promotion requires finite statistics, the predeclared minimum valid pairs, the
configured confidence, and a lower confidence bound meeting the minimum
practical effect. Invalid telemetry blocks remain recorded but do not count as
valid pairs.

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

- `references/performance_iteration.md`: V2.6 performance hypothesis, derived
  work class, infrastructure budget, prevalidated fallback, and stop rules.
- `references/compatibility.md`: supported versions, public APIs, architecture
  routing, and RTX 5090/SM120 facts.
- `references/optimization_catalog.md`: method triggers, skip rules, and
  combination constraints.
- `references/ncu_metrics_guide.md`: profiler metric interpretation.
- `references/systems_and_ir_coverage.md`: read only for systems-path,
  CUTLASS/CuTe, or Triton IR evidence tasks.
- `references/serving_evidence_protocol.md`: read only for runtime or serving
  evidence and claims.
- `references/evidence_automation.md`: V2.5 guard, frozen design,
  execution-path, identity, seal/audit/decision, import, and profiler contracts.
- `references/migration_v2_5.md`: V2.4.1 compatibility and non-mutating
  migration notes.
- `references/sanitizer_policy.json`: targeted/full sanitizer routing.
- `references/sass_signatures.json`: instruction signatures.
- `templates/objective.schema.json`: strict workload objective schema.
- `templates/guard_policy.schema.json`, `templates/experiment_design.schema.json`,
  `templates/attempt.schema.json`, `templates/execution_path.schema.json`,
  `templates/serving_experiment.schema.json`,
  `templates/artifact_identities.schema.json`,
  `templates/profiler_bundle.schema.json`,
  `templates/performance_verdict.schema.json`, and
  `templates/evidence_manifest.schema.json`: V2.5 formal evidence schemas.
- `templates/workload.py`: user-owned Python workload starter.
- `templates/iteration_report.md`: per-round decision record.
- `examples/walkthrough.md`: annotated kernel-only and full-mode walkthrough.
