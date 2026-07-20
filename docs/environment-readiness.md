# Preparing a workload and test environment

Optimization can begin only as far as the available evidence allows. The skill
first reports a **claim ceiling** so that static advice, a kernel benchmark, and
an end-to-end result are not confused. It then checks whether the declared
target can actually build, run, profile, and measure on the target environment.

The AI runs readiness automatically. The user does not need to copy internal
commands. The user-provided workload must represent the real system, and the
user gives explicit authorization for any isolated dependency repair.

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

The readiness report separates four outcomes:

| Status | Meaning | What happens next |
|---|---|---|
| `ready` | The capability was exercised successfully | Continue while the evidence is fresh |
| `degraded` | An optional or diagnostic layer is missing | Continue at a lower evidence layer |
| `user_action_required` | A host or policy decision belongs to the user | Record the exact action; do not change the host |
| `blocked` | A required capability failed | Stop before the baseline |

A required failure means the Controller does not run the baseline or later
high-cost profiling. It returns `readiness_action` with the exact blocker.
Diagnostic degradation, such as unavailable Nsys, may still permit a workload
run when every required capability is ready.

The agent may still inspect code, generated IR, PTX, SASS, or an existing report,
but those findings remain hypotheses until a matching measurement path exists.

Profiler counters are useful but not mandatory for every task. If NCU is
unavailable, the agent records the exact reason and uses a lower evidence layer
where possible. Stable correctness and target timing cannot be replaced by a
profiler.

The only automatic repair is hash-locked isolated pip into the environment named
in the contract. It requires explicit authorization, has a fixed time and retry
budget, and re-runs foundation probes after the environment identity changes.
All host changes remain recommendations, including drivers, permissions, clocks,
power limits, and services. Readiness never runs `sudo` or changes host policy.

## Protocol and local checks

The closed input and output formats are
`templates/readiness_contract.schema.json`,
`templates/readiness_probe.schema.json`, and
`templates/readiness_report.schema.json`. `control-v2` freezes the readiness
contract and report digests before the baseline. Freshness and environment
identity are checked again before high-cost probes or resume.

The installed `self_check` validates Python sources and schema/runtime agreement
using CPU/static checks. Passing `self_check` does not prove that the GPU
environment is ready; only target-side capability probes can establish that.
