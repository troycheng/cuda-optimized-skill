# Getting Started

## Install with Codex

Installation is performed by Codex. Ask Codex to install or update
`skills/cuda-kernel-optimizer` from
[troycheng/cuda-optimized-skill](https://github.com/troycheng/cuda-optimized-skill),
then start a new session so Codex reloads the skill instructions.

## Prepare the task

Provide as much of the following as currently exists. The skill first reports a
claim ceiling and helps prepare missing foundations before formal optimization:

1. A **runnable target**: kernel code, a complete workload, or an existing
   `.ncu-rep`.
2. A **correctness reference**: reference implementation, tests, validator, or
   comparable expected output.
3. The **test environment**: target GPU, driver, toolchain, dependencies, and
   access boundaries.
4. A **performance goal**: latency, throughput, memory, cost, or another primary
   KPI, including its direction and threshold.
5. **Constraints**: accuracy, checksums, output quality, memory limits, and any
   per-case requirements.
6. The **allowed modification scope**: project paths and isolated environment
   locations that may change.

A real workload must be supplied by the user. The skill does not download,
invent, or replace it with a microbenchmark. Without one, the strongest possible
result is a kernel-level claim.

If the runnable target, correctness reference, or stable benchmark is missing,
start with [Environment readiness](environment-readiness.md). Source-only work
may produce useful hypotheses, but not a performance result.

## Choose a budget

| Budget | Maximum wall time | Use it for |
|---|---:|---|
| `quick` | 45 minutes | Check an idea and narrow the candidate set |
| `balanced` | 3 hours | Default search and validation depth |
| `thorough` | 10 hours | Broader exploration and deeper evidence |

These are ceilings. A task may stop earlier when it has a conclusive result, no
eligible candidate remains, or required evidence is unavailable.

## First request

> Use cuda-kernel-optimizer to optimize the Triton kernel in this directory. Confirm the runnable reference, inputs, performance goal, constraints, allowed files, and target environment before profiling. Use the balanced budget and keep a change only when correctness and paired performance both pass.

Next, select the matching [workflow](workflows.md) and review the
[evidence and safety boundaries](evidence-and-safety.md).
