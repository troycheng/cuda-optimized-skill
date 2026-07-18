# Workflows

Choose the workflow from the strongest claim your inputs can support. A faster
kernel does not automatically establish a faster product workload.

## Performance-first iteration

Every optimization round begins with a falsifiable hypothesis and a bounded
candidate scope. The AI must produce a real candidate and correctness result;
when correctness passes, it must also produce comparable timing. The derived
classes `candidate_evaluated`, `measurement_blocked`, and
`infrastructure_only` keep completed experiments separate from tool work.

Measurement support has a fixed time and repair budget. Once exhausted, the AI
may use only a prevalidated fallback or stop that direction; it does not turn
the optimization round into runner development. See the
[performance-first iteration contract](../skills/cuda-kernel-optimizer/references/performance_iteration.md).

## Kernel optimization

Use this path for a CUDA, CUTLASS, or Triton implementation with a runnable
correctness reference. The skill can profile, inspect compiler and SASS evidence,
change authorized kernel code, and run paired measurements.

The result supports a **kernel-level claim** only. It does not establish serving
latency, throughput, or cost without a real workload.

## Complete workload

Use this path when the bottleneck may be in kernels, framework scheduling, CPU
processing, host-to-device transfers, communication, I/O, or the environment.
The workload, validation command, objective, constraints, and mutation roots
must be explicit.

An **end-to-end claim** requires the supplied workload evaluation to pass. A
kernel win may still be recorded even when the complete workload is unchanged,
but it cannot promote the product result.

## Serving validation

Use this path to test whether an implementation change improves a serving KPI.
The formal design freezes c1/c2/c4/c8/c12 strata, warmup and request counts,
fresh-process behavior, HTTP or gRPC mode, QPS/average/P95/P99 metrics, server
timing components, and per-stratum constraints.

A valid result also needs shared-host cleanliness, serving-stack artifact
identity, execution-path coverage, raw rows, and a sealed attempt. Performance
and evidence integrity remain separate decisions.

## Existing NCU report

Use this path when a `.ncu-rep` already exists and launching the profiled program
is not allowed. The importer performs **read-only** analysis and records exact
degradation when the report cannot be interpreted.

Importing a report does not prove current NCU counter permission, current binary
identity, or current environment cleanliness. Profiler output is diagnostic and
cannot promote a candidate by itself.

Review [Evidence & Safety](evidence-and-safety.md) before using a result for a
performance decision.
