# Serving Integration Evidence Protocol

Use this protocol when a CUDA, CUTLASS, or Triton kernel is loaded by TensorRT,
Triton Inference Server, CUDA Graphs, a custom backend, or another serving
runtime. It supplements the operator loop; it does not replace the deployment's
correctness and load-test harnesses.

## Claim only the layer that was measured

Keep these evidence layers separate:

| Layer | Evidence | Allowed claim |
| --- | --- | --- |
| Generated code | final cubin/SASS, resources, spills | the compiler emitted the intended mechanism |
| Isolated operator | reference validation plus repeated timing | the operator improved for the tested shapes |
| Matched runtime | same engine, inputs, runtime, and launch path; plugin A/B | the implementation improved inside that runtime |
| Engine/runtime | controlled artifact or runtime A/B | the engine/runtime stack improved |
| Serving endpoint | clean-window load test with identical model and request configuration | the tested endpoint metric improved |

Do not turn a lower-layer win into an endpoint claim. An endpoint win also does
not prove that the custom kernel improved; runtime, tactics, batching, queueing,
copies, or CPU work may be responsible.

## Freeze comparison identity

Before measuring, write a manifest for both labels. Include:

- source revision and build command;
- plugin/library, engine, model, and configuration hashes;
- CUDA driver/toolkit, TensorRT, Triton Server, and dependent library versions;
- GPU identity, clocks or power policy when controlled, and process launch command;
- input corpus identity, shape distribution, precision, batch, concurrency,
  protocol, warmup, and sample count;
- engine build log, tactic provenance when available, and CUDA Graph mode.

Change one factor at a time unless the explicit question is whole-stack
performance. A freshly rebuilt TensorRT engine is a new artifact even when its
model and builder flags look identical; hash it and retain its build evidence.

Use neutral physical labels in automation. Do not let filenames such as
`candidate` or `best` determine the expected outcome.

## Apply a correctness ladder

Advance only after the current tier passes:

1. Validate the isolated operator against the reference over normal, boundary,
   adversarial, and randomized shapes. Check every output and declared tolerance.
2. For synchronization, memory, or pointer changes, run the applicable
   Compute Sanitizer tools, including memcheck or racecheck.
3. Exercise the real runtime harness in ordinary launch and CUDA Graph modes.
   Check repeated launches, dynamic sizes, empty/tail cases, finite values,
   output counts, and untouched regions where relevant.
4. Replay representative production tensors or a versioned corpus repeatedly.
   Compare candidate disagreement with the same-stack repeat variation before
   attributing differences to the candidate.

State the exact oracle. Finite/count/tail checks are safety invariants, not
proof of semantic ground truth. Do not call a result bitwise accurate when the
acceptance criterion was tolerance-based or the runtime is nondeterministic.

## Use a staged performance ladder

1. **Static gate**: inspect final SASS, kernel topology, launch count, registers,
   shared memory, local memory, and spills. Compiler folding or reordering can
   invalidate a source-level optimization story.
2. **Operator gate**: collect independent timing samples after warmup. Retain
   the raw distribution, not only a mean.
3. **Matched-runtime gate**: use the same engine and runtime while switching only
   the implementation. Use Nsight Systems or equivalent timeline evidence for
   repeated aggregate GPU time, launch count, copies, graph behavior, and gaps.
4. **Stack decomposition**: separately test engine/runtime and implementation
   changes. A newer runtime may improve the whole stack while making the custom
   plugin slower; report both facts.
5. **Endpoint gate**: use the identical Triton model configuration, input set,
   transport, concurrency, dynamic-batching policy, and measurement window.
   Report throughput and the required latency percentiles together.

Do not advance a candidate whose correctness failed or whose apparent gain is
inside the declared noise threshold.

## Protect shared-host measurements

Predeclare a contamination guard before starting. Observe at least:

- foreign GPU processes, utilization, memory use, clocks, thermals, and power;
- foreign CPU-heavy builds or profilers, host load, and memory pressure;
- the tested server's health, restarts, and configuration identity.

Require a sustained clean window, not a single clean sample. If contamination
appears anywhere in a trial, abort and archive the entire trial as invalid. Do
not delete inconvenient samples after seeing their values, combine partial
orders, or treat an interrupted run as a result.

On a shared host, wait passively. Never kill, pause, reprioritize, pin, or alter
unrelated processes without explicit authorization.

## Design comparisons against drift

- Interleave labels with a balanced order such as ABBA or BAAB; repeat complete
  blocks so warming and temporal drift affect both labels.
- Include physical-label reversal when the harness permits it to catch path,
  filename, or launch-order bias.
- Preserve each raw run, timestamp, order, guard verdict, and failure reason.
- Analyze paired deltas when runs share a block. Report central tendency,
  dispersion, win count, and a confidence interval or bootstrap interval.
- Promote only when the improvement is larger than the predeclared noise floor,
  directionally consistent, and supported by a confidence interval that excludes
  no improvement for the primary metric.

Do not inspect an incomplete sequence and change direction from its partial
values. Decide the stopping and rejection rules before the measurements.

## Respect tool boundaries

- **NCU** explains a selected kernel's mechanism. Replay and cache effects can
  perturb timing, so do not use one NCU duration to overrule repeated clean
  runtime measurements.
- **Nsight Systems** attributes repeated runtime time, launches, copies, graph
  behavior, and inter-kernel gaps. It does not by itself prove correctness.
- **Endpoint load tests** establish user-visible serving behavior, including
  queueing and CPU/transport effects. They do not isolate a kernel mechanism.
- **SASS/resource inspection** verifies emitted code and compiler consequences.
  It does not establish a speedup.

Use external search or other AI systems to expand hypotheses, never as measured
evidence. Verify commands, APIs, architecture support, and metric meanings in
current primary documentation and the installed toolchain. Compiler output and
captured artifacts outrank an optimization narrative.

## Promotion record

For every promoted or rejected candidate, preserve:

- immutable artifact identities and exact commands;
- correctness tier reached and oracle used;
- valid raw samples and contamination verdicts;
- effect size, uncertainty, and the exact evidence layer;
- profiler and SASS evidence used for mechanism attribution;
- rejection reason or remaining risk.

Stage candidates outside production. Do not mutate the production model
repository or deployment while measuring alternatives. Keep rejected candidates
and invalid trials clearly labeled so negative knowledge is reusable.

The final recommendation must distinguish:

1. best isolated operator;
2. best matched-runtime implementation;
3. best engine/runtime stack;
4. best validated endpoint deployment.

These may be different artifacts. Report that outcome directly rather than
forcing one winner across all layers.
