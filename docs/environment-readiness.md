# Preparing a workload and test environment

Optimization can begin only as far as the available evidence allows. The skill
first reports a **claim ceiling** so that static advice, a kernel benchmark, and
an end-to-end result are not confused.

The requested result is declared as `kernel`, `workload`, or `serving`. The gap
report includes only the foundation needed for that result; a kernel-only task
does not ask for a serving benchmark.

| What is available | What the skill can establish |
|---|---|
| Source only | Static hypotheses and a preparation plan |
| Source, correctness reference, reproducible build | Correctness and compiler evidence |
| Stable kernel benchmark | Kernel-level performance |
| Representative workload | End-to-end workload performance |
| Frozen serving experiment | Serving KPI performance |

## Minimum foundation

Provide or approve:

1. the source or profiler artifact to inspect;
2. a reference implementation, validator, or trusted output set;
3. representative shapes, dtypes, inputs, and tolerances;
4. a reproducible build or import command in an isolated environment;
5. warmup, paired timing, raw samples, and the performance objective;
6. the project paths that may change;
7. the real workload or replay when an end-to-end claim is required.

The skill will not invent a workload or silently replace it with a synthetic
microbenchmark. It can generate a project-local adapter, benchmark scaffold,
case inventory, profiler command, and remote-run checklist, but the user must
confirm that the cases and objective represent the real system.

## When the environment is incomplete

The readiness report identifies missing items and the strongest possible claim.
The agent may still inspect code, generated IR, PTX, SASS, or an existing report,
but those findings remain hypotheses until a matching measurement path exists.

Profiler counters are useful but not mandatory for every task. If NCU is
unavailable, the agent records the exact reason and uses a lower evidence layer
where possible. Stable correctness and target timing cannot be replaced by a
profiler.

Drivers, permissions, clocks, power limits, services, and other host settings
are recommendations only unless the user separately authorizes a change.
