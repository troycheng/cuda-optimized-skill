# Evidence & Safety

## Claim ladder

Evidence is layered. Correctness and diagnostics can reject a candidate early;
kernel timing can support a kernel-level claim; only a user-supplied complete
workload can support an end-to-end claim. Profiler output explains behavior but
does not promote a change.

## Frozen experiment

Before formal timed work, freeze the complete schedule, position balance,
experimental and resampling units, aggregation rule, confidence-interval method,
minimum valid pairs, win rule, relative and absolute guardrails, and no-exclusion
policy. Once timed work begins, retrying only one role is forbidden.

Correctness, constraints, paired raw rows, and the default 95% confidence
interval must agree with the frozen design. Missing or contradictory required
evidence must **fail closed**.

## Shared-host guard

Formal timing requires continuous shared-host samples for the target, peer, and
sibling GPUs; CPU and NUMA state; foreign process/container allowlists; swap and
memory pressure; and clock, temperature, power, and thermal conditions. Each
phase has its own policy, maximum sample gap, watcher-ready handshake, and joint
clean-window requirement.

Unknown state, missing samples, stale samples, or contamination cannot be treated
as clean timing evidence.

## Nonstationary comparisons

Shared-host cleanliness alone does not make two serving windows comparable.
Queue depth, offered load, cache state, or another declared state can move
between roles or between burn-in and timing. For these runs, the V2.8 gate
requires a predeclared balanced AB/BA plan, fixed-duration windows, and separate
pair and phase tolerances. Rows remain in chronological order and stay bound to
their raw source; no post-hoc deletion or regrouping is accepted.

This gate answers only whether paired state is comparable. It never turns metric
values into a speedup claim. See the
[nonstationary serving-evidence contract](../skills/cuda-kernel-optimizer/references/nonstationary_serving_evidence.md).

## Attempt and identity

An immutable attempt binds the runner, guard, analysis, schedule, source,
binary, plugin, engine, backend, server, image digest, raw rows, phase markers,
and audit record. Valid lifecycle states distinguish valid evidence from
contaminated, identity-invalid, partial, or superseded attempts.

Execution-path coverage must prove expected cases and hit counts. A diagnostic
binary cannot supply performance timing; removing diagnostics requires a rebuild,
new digest, and a binding between the proved path and the timed binary.

Formal serving runs cover c1/c2/c4/c8/c12 strata and bind TensorRT or Triton
artifacts at the binary, engine, tactic, timing-cache, plugin, backend, server,
and image boundaries. Same source does not imply same binary.

## Decision separation

`performance_verdict` answers whether the frozen performance objective passed.
`evidence_integrity` answers whether the evidence is complete, clean, immutable,
and auditable. A performance win with invalid evidence cannot be adopted.

The installed `self_check` is CPU/static only and does not validate a GPU environment.
It checks package metadata, Python scripts, and the bundled V2.5-V2.8 schemas
without running GPU or network work.

## Modification boundary

The skill may change only declared project paths and user-provided isolated
environments. It never changes host configuration automatically. Drivers,
permissions, clocks, power limits, services, containers, and system settings are
recommendations unless the user separately authorizes them.

The canonical formal contract is the
[V2.5 evidence automation reference](https://github.com/troycheng/cuda-optimized-skill/blob/main/skills/cuda-kernel-optimizer/references/evidence_automation.md).
