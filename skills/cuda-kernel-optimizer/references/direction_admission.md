# V2.7 direction admission

Use this protocol before a V2.6 candidate round when more than one optimization
direction exists, a direction has already been stopped, or recent improvements
are small relative to the remaining bottleneck. It prevents mechanism renaming
from resetting a stop decision. It does not replace profiling or performance
measurement.

## Boundary

`direction_guard.py` is a deterministic, read-only decision tool. It never runs
a target, benchmark, profiler, compiler, workload, or external reviewer. It
never edits project code or host configuration. It does not claim a measured
performance gain; V2.5 evidence and the relevant promotion gate remain the
authorities for correctness and speed.

Inputs are supplied by the user or by existing measurement tools. The guard
validates and hashes them but does not collect them.

## Direction identity

A direction family is derived from:

- claim layer: `kernel`, `runtime`, `workload`, or `serving`;
- bottleneck class: `kernel`, `framework`, `cpu_data`, `transfer`,
  `communication`, `io`, or `environment`;
- stable component identifier;
- metric name, unit, direction, and kind.

The concrete direction key also binds the target identity. A label, mechanism
name, candidate hash, or iteration number is deliberately absent. Calling the
same work a new mechanism cannot create a new direction family.

## Comparable impact envelope

Automatic ranking is deliberately narrow. It applies only when the selected
direction and the frozen objective are in the same layer and use the same
additive, lower-is-better time metric:

```text
total_metric > 0
0 <= component_metric <= total_metric
upper_bound_absolute = component_metric
upper_bound_percent = 100 * component_metric / total_metric
```

In ratio form, the ceiling is `component_metric / total_metric`. It assumes the
component can be removed completely, so it is a full-elimination upper bound,
not a forecast. The guard does not ask an AI to estimate an eliminable fraction.

The minimum absolute and/or percentage effect is frozen in the objective. A
direction closes when even full elimination cannot meet every declared floor.
Among eligible same-layer directions, the largest ceiling is recommended.

Throughput, composite objectives, higher-is-better objectives, and cross-layer
comparisons return `unrankable`. They require a separately designed experiment;
post-hoc weights or automatic serving-state clustering are not substitutes.

## Ledger and state transitions

`init` writes `direction-lineage.json` exactly once. It freezes the objective,
environment, registered direction families, and initial portfolio digest.

`check` validates the whole decision directory, derives the next decision, and
appends `direction-decisions/decision-NNNN.json` with create-once semantics.
Every record binds the lineage and previous file digest, forming one hash chain.
Unknown files, gaps, broken hashes, symlinks, drift, or concurrent duplicate
writes fail closed. `status` validates the same chain without writing.

The actions are:

- `admit_direction`: the direction clears the frozen floor and is the largest
  comparable ceiling;
- `switch_to_higher_impact`: another comparable family has a larger ceiling;
- `close_direction`: the ceiling misses the floor or the caller explicitly
  closes the direction;
- `direction_closed`: later work tried to continue a closed family without a
  qualified reopen;
- `unrankable`: the input is outside the automatic comparison boundary.

After admission, V2.6 still owns candidate budgets, fallback paths, and per-round
stopping. V2.7 does not authorize a candidate or promotion by itself.

## Reopening a closed family

Reopen only when all of the following are true:

1. the latest decision for the same family is closed;
2. a new evidence digest is present;
3. the measurement window, target identity, or measured impact envelope changed;
4. the recomputed full-elimination ceiling still meets the original frozen floor.

A prose rewrite, new mechanism name, new iteration number, or a changed minimum
effect is not reopen evidence. Use `--request reopen`; an ordinary `check`
against a closed family returns `direction_closed`.

## Minimal sequence

```bash
python3 <skill>/scripts/direction_guard.py init \
  --portfolio direction-portfolio.json --run-dir direction-run

python3 <skill>/scripts/direction_guard.py check \
  --portfolio direction-portfolio.json --run-dir direction-run \
  --direction-id selector

python3 <skill>/scripts/direction_guard.py status --run-dir direction-run
```

Use `--request close` to record a deliberate stop and `--request reopen` only
with the qualified evidence above. The JSON contracts are
`templates/direction_portfolio.schema.json`,
`templates/direction_lineage.schema.json`, and
`templates/direction_decision.schema.json`.
