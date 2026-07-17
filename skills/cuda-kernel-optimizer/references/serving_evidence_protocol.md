# Serving Evidence Protocol

Use the narrowest claim supported by the evidence. A faster instruction path is
useful evidence, but it is not a serving result.

For the executable V2.5 contracts and `scripts/evidence.py` commands, read
`references/evidence_automation.md`. This document explains the claim boundary;
the executable validators enforce the normalized evidence files.

## Claim ladder

| Layer | Minimum evidence | Allowed claim |
|---|---|---|
| Generated code | Compiler output or SASS bound to the candidate binary and target architecture | The intended mechanism was emitted for that binary. |
| Isolated operator | Reference validation and paired A/B timing on identical inputs and device conditions | The tested operator improved for that input set. |
| Matched runtime | Paired A/B runs with the same engine, model, inputs, batching, cache policy, and runtime settings | The implementation improved in that runtime configuration. |
| Serving endpoint | A clean window load test with identical model, request mix, concurrency policy, and endpoint configuration | The tested endpoint metric improved in that window. |

Generated code only proves that a mechanism was emitted; it does not prove that
the path executed or improved performance. Operator timing does not prove a serving benefit.
Generated-code or operator evidence alone cannot support a deployment claim.

## Connect the result to the run verdict

The outer evaluation belongs to the user-provided workload. It must preserve
the production objective and representative request distribution; a synthetic
kernel benchmark cannot replace it.

- `kernel_only_win` means the paired kernel result improved but the real
  workload did not confirm the configured end-to-end objective. Do not promote
  it as a serving win.
- `end_to_end_win` means the candidate passed the run's correctness and paired
  gates and the user-provided workload confirmed the configured objective. The
  claim is still limited to the captured workload, environment, and test
  window.

## Serving A/B evidence

Collect baseline and candidate measurements as paired A/B observations. Keep
model weights, request replay, admission control, batching, cache state,
warm-up, concurrency, and measurement boundaries identical. Randomize or
interleave pair order when time drift could bias one side. Freeze the complete
position-balanced schedule before the run. For fresh-process comparisons,
balance both variant order and ordinal position; when more than one physical GPU
is unavoidable, cross over physical labels instead of binding one version to one
device.

A serving result also needs:

| Evidence | Record |
|---|---|
| Raw request evidence | Request timestamps or pair IDs, input-shape and sequence-length distributions, concurrency, response status, and the raw objective samples. Redact payload content when required, but retain stable request identities. |
| Environment evidence | GPU and host identity, software and model versions, clocks or power policy, container image, runtime flags, cache policy, and competing processes. |
| Clean window | Warm-up boundary, start and end time, stable traffic policy, error rate, and confirmation that maintenance or unrelated rollouts did not overlap. |
| Contamination check | GPU occupancy and host load before and during the run, plus scheduler or process evidence sufficient to detect shared-host contamination. |

If shared-host contamination cannot be ruled out, mark the serving comparison
inconclusive or repeat it in a clean window. Do not average away interference.
Retain raw request and environment evidence with the summary so another person
can reproduce the pairing and check the claim.

### Shared-host clean-window gate

Promotion-grade evidence requires continuous, phase-bounded guard coverage, not
only a before/after snapshot. Freeze the target GPU UUID/PCI address, peer and
sibling GPUs, CPU/NUMA allocation, allowed PIDs or containers, sampling interval,
maximum sample gap, clock/power policy, and limits for temperature, power/thermal
braking, swap, memory pressure, and foreign CPU/GPU load.

Use separate `correctness`, `sanitizer`, `diagnostic`, and `timing` phases. A
watcher must acknowledge readiness before each phase starts. Clear only this
attempt's marker between phases and wait for a fresh joint clean window. Heavy
CPU work owned by correctness must not poison a later timing phase. During a
promotion timing phase, unknown telemetry, a sampling gap, an untrusted process,
or a contamination marker invalidates the attempt; do not analyze its timing.

### Attempt lifecycle and evidence seal

Assign every run an immutable attempt ID and one terminal state:
`valid`, `invalid_contaminated`, `invalid_identity`, `partial`, or `superseded`.
Never merge rows across attempts or turn a partial/invalid attempt into a result.
At terminal closure, seal hashes for the schedule and analysis code; runner and guard
scripts; source and binaries; plugin, engine, backend, and server; image digest
(not only a tag); workload/request corpus; raw samples; phase markers; and guard
telemetry. The final audit must distinguish `evidence_integrity=PASS` from the
performance verdict.

### Execution-path proof

Emitted SASS is not evidence that real inputs used the candidate path. If a
dispatch guard, fallback, tactic, graph, cache, or shape specialization can bypass
the mechanism, require a pre-timing coverage gate. Record expected cases and a
dispatch counter or trace proving the candidate symbol/topology executed. A
diagnostic binary cannot supply performance evidence: remove diagnostics,
rebuild, rehash, verify no diagnostic residue, and bind the timed plugin/engine
to the same source and configuration.

### Serving-stack identity

For TensorRT/Triton/CUDA comparisons, bind image digests, server and backend ABI
and hashes, plugin source and binary hashes, engine hash, builder parameters and
logs, tactics, runtime/model configuration, and request corpus. State the causal
scope precisely: same source is not same binary; a rebuilt engine is not the
same engine; changing server, TensorRT, CUDA, plugin compiler, and tactics is a
stack comparison, not a TensorRT-only result. Treat timing caches as
version-local unless compatibility is explicitly proven.

### Frozen statistics and retry policy

Predeclare the experimental unit (for example a fresh process, not each of
30,000 correlated requests), aggregation (`mean`, `median`, or ratio of sums),
resampling unit and method, confidence, minimum effect, wins requirement, and
relative plus absolute guardrails per concurrency/stratum. Do not add a new
bootstrap or exclusion rule after seeing results. Quick exact-runtime pairs are
reject-only; formal promotion uses an independent schedule and samples.

For endpoint evidence, freeze HTTP and/or gRPC c1/c2/c4/c8/c12 strata. An Nsys
or NCU explanation bundle is always `non_promotional`; profiler observations
cannot replace endpoint constraints or `evidence_integrity=PASS`.

Once timed work begins, retain every uncontaminated row. Do not retry one role
independently, remove a slow startup mode, or select a favorable subset. A whole
pair may be retried only for a frozen, pre-measurement infrastructure failure;
otherwise retain or invalidate it according to the frozen rule.

## Decision checklist

- [ ] The claim stops at the highest completed layer in the claim ladder.
- [ ] Correctness passed before any performance claim.
- [ ] The comparison used paired A/B samples with identical policies.
- [ ] The complete schedule is frozen and position-balanced.
- [ ] The minimum valid pair count and formal statistical plan were frozen.
- [ ] Any bypassable candidate path has execution-path coverage evidence.
- [ ] The user-provided workload matches the intended serving objective and
      request distribution.
- [ ] A clean window and shared-host contamination check were recorded.
- [ ] Continuous guard coverage spans every formal timing phase without gaps.
- [ ] Raw request evidence and environment evidence are durable and linked from
      the result.
- [ ] Tail latency, throughput, error rate, and resource cost were checked when
      they can trade off against the primary objective.
- [ ] `kernel_only_win` is not described as `end_to_end_win`.
- [ ] The sealed evidence digest covers the runner, guards, analysis, artifacts,
      raw data, and final audit.
