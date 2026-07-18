# CUDA Skill V2.5 Evidence Automation Design

## Scope

V2.5 adds a standard-library-only evidence automation layer beside the V2.4.1
kernel and workload controllers. It does not collect GPU data on this Mac, alter
host settings, connect to a remote GPU lane, or turn imported profiler data into
promotion evidence. Existing V2.4.1 manifests remain readable but cannot claim a
V2.5 formal evidence seal until migrated.

## Architecture

The implementation has four bounded components:

1. `experiment_design.py` validates a complete frozen, position-balanced formal
   schedule and its statistical/retry contract. `workload_evaluate.py` accepts
   this optional design and forbids per-role retries in formal mode.
2. `evidence_protocol.py` validates shared-host samples, phase handshakes,
   execution-path coverage, serving experiments, artifact identities, profiler
   bundles, attempts, seals, audits, decisions, and closure manifests.
3. `evidence.py` exposes the pure validation and immutable lifecycle as a CLI.
   It consumes normalized site-owned telemetry; it does not pretend to provide a
   universal GPU sampler. Nine closed V2.5 schemas publish the input and closure
   boundaries.
4. `self_check.py` validates the installed skill, templates, imports, and CPU
   example contracts without CUDA, NCU, Nsys, or a network connection.

## Shared-host guard

The guard policy freezes target, peer, and sibling GPU UUID/PCI identities;
CPU affinity and NUMA nodes; PID/container allowlists; sample interval and
maximum gap; a joint clean-window duration; and clock, temperature, power,
thermal, swap, memory-pressure, and foreign-load limits.

Normalized samples are monotonic records. Every required field must be present
for every frozen GPU and the CPU/memory surface. Each of `correctness`,
`sanitizer`, `diagnostic`, and `timing` is explicitly `required` or
`not_applicable`. A required phase needs a watcher-ready marker, a clean window
before readiness, continuous samples through phase end, and no gap beyond the
frozen limit. Unknown data, missing identities, untrusted actors, forbidden
throttle reasons, contamination markers, or a gap make the phase fail. Formal
timing never treats unknown as clean.

## Frozen experiment design

A formal design freezes every pair ID and AB/BA order, experimental and
resampling units, aggregation, CI method/configuration, confidence, minimum
valid pairs, wins requirement, relative and absolute guardrails, no-exclusion,
and retry rules. The only allowed formal retry is a whole pair for a declared
pre-measurement infrastructure failure. The existing evaluator cannot safely
restart a whole pair, so its formal API uses zero role retries and records every
scheduled pair.

## Execution-path and identity gates

Coverage evidence declares every expected case and positive dispatch hit count,
plus a counter, trace, or topology proof. Diagnostic binaries are explanatory
only. A timed binary must be rebuilt after diagnostic removal, have a new binary
hash, retain the intended source/configuration binding, and declare no diagnostic
residue.

Serving identity binds source and binary separately and records plugin, engine,
backend, server, image digest, builder/runtime versions, tactics, and timing
cache. A tag is never an image identity. Same source is not treated as the same
binary, and engine/tactic/timing-cache drift changes the comparison scope.

## Immutable attempt lifecycle

An attempt has one terminal state: `valid`, `invalid_contaminated`,
`invalid_identity`, `partial`, or `superseded`. The lifecycle is:

```text
attempt manifest -> seal -> audit -> decision -> closure manifest
```

The seal hashes the attempt, runner, guard, analysis, schedule, identities,
raw rows, phase markers, coverage evidence, performance verdict, and optional
profiler bundle. It is create-once. Audit independently rehashes the sealed
files and reports only `evidence_integrity=PASS|FAIL`; it does not reinterpret
performance. Decision independently rehashes the artifacts again, rejects a
forged or stale audit, and recomputes the semantic gates. It requires a valid
attempt, passing integrity, guard,
coverage, and serving constraints before a performance verdict can be
promotional. The final manifest hashes seal, audit, and decision and is the root
for `evidence_refs`.

Imported serving audits are read-only: inputs are opened without mutation and
all outputs must be written outside the imported tree. Missing V2.5 artifacts
produce a non-promotional compatibility result, never an inferred pass.

## Serving and profiler contracts

Serving experiments freeze HTTP and/or gRPC strata for concurrency
`c1/c2/c4/c8/c12`, warmup and measured request counts, fresh-process semantics,
request corpus identity, QPS/average/P95/P99 latency, server input/infer/output
timing, and per-stratum relative plus absolute must-pass constraints.

Nsys/NCU bundles bind reports and commands to the timed binary and may explain
timeline or kernel behavior. Their authority is always `non_promotional`; a
profiler bundle cannot satisfy correctness, timing, serving, guard, or coverage
gates.

## Compatibility

V2.4.1 kernel, workload, control, state, and objective manifests retain their
current APIs and behavior. V2.5 import audit labels them `legacy_unsealed` until a
frozen design, guard policy/samples, coverage proof, terminal attempt, and
identity set are supplied. Migration does not rewrite old evidence in place.

## Verification

Every new behavior receives a CPU unit or CLI test. Verification runs the full
`unittest` suite, skill `quick_validate.py`, `self_check.py`, `compileall`,
`git diff --check`, and focused reruns for any observed flaky test. No opt-in GPU
test is enabled.
