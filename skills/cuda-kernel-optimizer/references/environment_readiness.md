# Environment readiness and claim ceiling

Use this workflow before optimization when the user has not supplied a runnable
target, correctness reference, stable measurement path, or representative
workload. Do not invent any of them.

Run capability readiness before baseline. Do not ask the user to run these
commands manually; the agent owns contract construction, bounded probes,
report interpretation, and safe resume.

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

For a new Controller run, use `control-v2` and a closed readiness contract.
Run foundation requirements before workload requirements. A failed `required`
item must return `readiness_action`; do not load the baseline evaluator or start
high-cost workload profiling. A failed diagnostic item may become `degraded`
when a lower valid evidence layer remains.

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
host policy. Keep all host work `recommend_only`: report the exact blocker and
provide a recommendation instead.

The only executable remediation is contract-authorized `isolated_pip` with a
regular hash-locked requirements file and a fixed repair/time budget. It may
write only to the approved isolated environment. Recompute the full environment
identity and restart foundation probes after a successful install. If the
requirements digest, interpreter identity, report marker, TTL, or environment
identity changes unexpectedly, fail closed and keep the old run for audit.

`self_check` is CPU/static packaging validation. It does not establish target
GPU readiness.
