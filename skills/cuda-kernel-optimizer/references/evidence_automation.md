# V2.5 Evidence Automation

Use this reference for formal matched-runtime or serving attempts. It defines
the normalized contracts enforced by `scripts/evidence.py`. The generic guard
does not collect hardware, scheduler, PID, or container telemetry itself. A
site-owned watcher must produce the normalized continuous sample stream.

## Formal workflow

Freeze these inputs before any formal timed work:

- the complete position-balanced AB/BA schedule;
- experimental unit, aggregation, resampling unit, CI method and confidence;
- minimum valid pairs, wins requirement, and relative plus absolute guardrails;
- no-exclusion and whole-pair-only pre-measurement retry rules;
- shared-host identities, limits, sample interval, maximum gap, and phases;
- expected execution-path cases and the artifact identity boundary;
- serving protocols, strata, request corpus, counts, metrics, and constraints.

Then close one attempt in this order:

```text
seal -> audit -> decision -> evidence manifest
```

Never edit a sealed attempt. Start a new attempt ID and mark the replaced one
`superseded`.

## Shared-host guard

The guard policy freezes:

- target, peer, and sibling GPU UUID and PCI identities;
- CPU affinity and NUMA nodes;
- foreign PID and container allowlist entries;
- swap, memory pressure, and foreign CPU/GPU limits;
- minimum clock, maximum temperature and power, and forbidden power/thermal
  throttle reasons;
- continuous sample interval, maximum gap, and joint clean-window duration;
- explicit `correctness`, `sanitizer`, `diagnostic`, and `timing` phase status.

Every required phase needs a watcher-ready marker after a complete joint clean
window. Samples must cover the phase through its end without a gap above the
frozen maximum. Missing metrics, unknown identities, CPU/NUMA drift, an
untrusted actor, swap or memory pressure, clock/temperature/power/thermal limit,
contamination marker, missing watcher-ready handshake, or missing sample must
fail closed. An explicitly `not_applicable` phase needs a frozen reason; timing
is always required for formal performance evidence.

Run the executable audit over site-produced normalized inputs:

```bash
python3 <skill>/scripts/evidence.py guard-audit \
  --policy guard-policy.json \
  --samples guard-samples.jsonl \
  --markers phase-markers.json \
  --out guard-audit.json
```

Exit code `3` means a complete audit artifact was written with `status=FAIL`.
Exit code `2` means the contract or path was malformed. Neither is clean.

## Frozen statistics and raw rows

Use `templates/experiment_design.schema.json`. Runtime validation also checks
unique pair IDs, real AB/BA position balance, schedule-length bounds, finite
statistics, both guardrail forms, and the exact retry policy.

Once formal timing starts, do not retry only baseline or candidate. The current
formal evaluator uses `role_retries=0`; an external runner that implements a
whole-pair retry may do so only for the frozen pre-measurement infrastructure
reason. Retain every scheduled raw row. A slow valid row is not an exclusion
reason.

When `workload_evaluate.evaluate_pairs` receives the design, its CI confidence,
bootstrap sample count, seed, minimum valid pairs, wins requirement, and exact
schedule override ordinary call-site statistics. That evaluator currently
supports only `median_paired_improvement`; another frozen aggregation needs a
separate sealed analysis implementation and is rejected by this evaluator.

## Execution-path coverage

Generated code is not proof that the workload dispatched it. Before timing,
list every expected case and record a positive hit count from a dispatch
counter, trace, or topology proof. Fallback, tactic, graph, cache, and shape
specialization paths belong in the case set when they can bypass the candidate.

Diagnostic binaries are `non_promotional`. After coverage collection:

1. remove diagnostic counters and tracing;
2. rebuild and rehash the binary;
3. verify source and build-configuration binding;
4. verify no diagnostic residue;
5. bind the new timed binary hash to the formal rows.

The diagnostic and timed binary hashes must differ.

## Serving experiment and identity

Use `templates/serving_experiment.schema.json`. For each selected HTTP or gRPC
protocol, freeze every c1/c2/c4/c8/c12 stratum with fresh baseline and candidate
processes, warmup count, measured request count, and request corpus digest.
Require QPS, average latency, P95, P99, server input/infer/output timing, and
per-stratum relative plus absolute must-pass constraints.

Validate the boundary described by
`templates/artifact_identities.schema.json`. Bind TensorRT/Triton serving
artifacts separately:

- source hash and timed binary hash;
- plugin binary, source, compiler version, and ABI;
- engine hash, builder/runtime versions, plugin hash, tactic digest, and timing
  cache digest;
- backend and server hashes, versions, and ABIs;
- immutable image digest, never only a tag.

Same source is not the same binary. A rebuilt engine, changed tactic set, or
different timing cache changes the identity boundary. Treat timing-cache
compatibility as version-local unless separately proven.

## Attempt seal and decision

An attempt ends as exactly one of `valid`, `invalid_contaminated`,
`invalid_identity`, `partial`, or `superseded`. The seal covers the attempt
manifest, runner, guard, analysis, schedule, source, diagnostic and timed
binaries, plugin, engine, backend, server, image digest, raw rows, phase markers,
execution-path proof, formal design, serving experiment, performance verdict,
and optional profiler bundle/report files. The seal rejects a declared schedule,
binary, serving identity, image digest, or profiler report that does not match
the corresponding sealed artifact. Required kinds depend on the claim layer.

`audit` rehashes those files and emits only evidence-integrity findings. It does
not reinterpret performance. `decision` rehashes the evidence again, rejects a
stale or forged audit, recomputes semantic gates, and reads the performance
verdict separately. Keep seal, audit, decision, and the final manifest in one
evidence directory so every `evidence_refs` path closes locally. Promotion
requires a `valid` attempt, all formal gates, `evidence_integrity=PASS`, and a
promotional confirmed win. Use `templates/performance_verdict.schema.json` and
`templates/evidence_manifest.schema.json` for those two boundaries. The
performance verdict must bind the sealed analysis implementation, experiment
design, and raw rows by digest; it cannot assert `evidence_integrity` itself.

## Imported serving runs

`audit-imported` is a read-only imported-run path. Its output directory must be
outside the source tree. A V2.4.1 run without a V2.5 seal is reported as
`legacy_unsealed`, `evidence_integrity=UNKNOWN`, and non-promotional. Do not
rewrite or manufacture missing evidence.

## Nsys and NCU explanation bundle

Use `templates/profiler_bundle.schema.json`. Include both Nsys and NCU records,
even when one is explicitly unavailable, and bind every available report and
argv to the timed binary hash. Store observations and limitations. The authority
is always `non_promotional`: profiler data explains a result but cannot replace
correctness, shared-host guard, exact-runtime rows, serving constraints, or the
execution-path gate.
