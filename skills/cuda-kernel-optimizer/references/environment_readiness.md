# Environment readiness and claim ceiling

Use this workflow before optimization when the user has not supplied a runnable
target, correctness reference, stable measurement path, or representative
workload. Do not invent any of them.

## Readiness ladder

| Available foundation | Strongest permitted result |
|---|---|
| Source only | Static hypotheses; no speed claim |
| Source, reference, reproducible build | Correctness and compiler evidence |
| Stable kernel benchmark | Kernel performance claim |
| Representative workload | End-to-end workload claim |
| Frozen serving experiment | Serving KPI claim |

Run `scripts/readiness.py` with a small JSON inventory. Set `requested_claim` to
`kernel`, `workload`, or `serving`; omitted means `kernel`. `source_available`
means the current task can read the source. The script accepts a path or stdin
and reports only the missing foundation required for that target. Treat its
`claim_ceiling` as an upper bound, not a target to stretch.

## What to prepare

When the foundation is incomplete, help the user create project-local pieces:

- a reference or validator with representative cases and tolerances;
- a reproducible build/import command in an isolated environment;
- warmup, paired timing, raw samples, and a frozen aggregation rule;
- a shape and request inventory approved by the user;
- an adapter based on `templates/workload.py` and `scripts/workload_adapter.py`;
- profiler collection commands and an environment capability report;
- a remote-run checklist when the target GPU is elsewhere.

The user owns workload representativeness, business objectives, and acceptable
quality loss. Codex may propose a scaffold, but must ask the user to validate
those facts before making an end-to-end claim.

Do not change drivers, counter permissions, clocks, power limits, services, or
host policy. Report the exact blocker and provide a recommendation instead.
