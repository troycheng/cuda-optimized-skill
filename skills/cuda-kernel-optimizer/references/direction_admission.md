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

Inputs are supplied by the user or existing measurement tools. The portfolio
references regular environment, measurement-window, target, component artifact,
and evidence files
by safe relative path and expected digest. The CLI rehashes every referenced
artifact without following symlinks before it admits or reopens anything. A
bare AI-generated digest is not evidence. The guard does not collect artifacts.

## Direction identity

A direction family is derived from:

- claim layer: `kernel`, `runtime`, `workload`, or `serving`;
- bottleneck class: `kernel`, `framework`, `cpu_data`, `transfer`,
  `communication`, `io`, or `environment`;
- stable component artifact emitted by the profiler or application evidence;
- human-readable component identifier, which is not part of family identity;
- metric name, unit, direction, and kind.

The component artifact digest, rather than a caller-chosen component name, is in
the family key. The concrete direction key also binds the target identity. A
label, mechanism name, candidate hash, or iteration number is deliberately
absent. Calling the same work a new mechanism cannot create a new family.

## Comparable impact envelope

Automatic ranking is deliberately narrow. It applies only when the selected
direction and the frozen objective are in the same layer and use the same
additive, lower-is-better time metric:

```text
total_metric > 0
0 <= component_metric <= total_metric
upper_bound_absolute = component_metric
upper_bound_percent = 100 * component_metric / baseline_total_metric
```

The baseline total is frozen per family at `init`; later work cannot shrink the
denominator to manufacture a larger percentage. The ceiling assumes the
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
The initial digest is the first snapshot anchor, not a rule that all later
measurements must be byte-identical. A later portfolio snapshot may update the
window, target, evidence, and measured component values within the same ledger;
the objective, environment, and registered family taxonomy remain frozen. Each
decision records the current portfolio digest.

`check` validates the whole decision directory, derives the next decision, and
appends `direction-decisions/decision-NNNN.json` with create-once semantics.
Every record binds the lineage and previous file digest, forming one hash chain.
Unknown files, gaps, broken hashes, symlinks, drift, or concurrent duplicate
writes fail closed. After the first decision, every `check` must pass the last
value returned by `status` as `--expected-tail-sha256`; stale or missing caller
state cannot append. `status` validates the chain and returns its current tail.

The actions are:

- `admit_direction`: the direction clears the frozen floor and is the largest
  comparable ceiling;
- `switch_to_higher_impact`: another comparable family has a larger ceiling;
- `close_direction`: the ceiling misses the floor or the caller explicitly
  closes the direction;
- `direction_closed`: later work tried to continue a closed family without a
  qualified reopen;
- `unrankable`: the input is outside the automatic comparison boundary.

Only a result with `admitted: true` may enter V2.6. Every other result is a hard
non-admission, including `switch_to_higher_impact` and `unrankable`. A mixed
portfolio is allowed: the selected direction is compared only with its
same-layer, same-metric additive subset. Selecting a throughput, composite, or
cross-layer direction returns `unrankable` and does not authorize the AI to
continue on its own.

After admission, V2.6 still owns candidate budgets, fallback paths, and
per-round stopping. V2.7 does not authorize a promotion by itself.

Whenever the AI reports that a direction should stop, it must append
`--request close` before generating, editing, compiling, or measuring another
candidate. If an earlier spoken stop was not recorded, backfill the close
decision before any other optimization work. Natural-language stop statements
alone are not machine state and must never be used as permission to continue.

## Reopening a closed family

Reopen only when all of the following are true:

1. the latest decision for the same family is closed;
2. a new evidence artifact is present and its digest is reverified by the CLI;
3. the measurement window, target identity, or measured impact envelope changed;
4. the absolute and frozen-baseline percentage ceilings each increase over the
   closure by at least their corresponding minimum effect.

A prose rewrite, new mechanism name, caller-invented component ID, unbound
digest, new iteration number, or changed minimum effect is not reopen evidence.
Use `--request reopen`; an ordinary `check` against a closed family returns
`direction_closed`.

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

## Storage trust boundary

The expected-tail handshake detects stale callers, ordinary races, and a tail
changed since the caller last observed it. The ledger is tool-level append-only,
not filesystem WORM storage. A process that can delete the entire ledger, forge
all input artifacts, or rewrite both the tail and the caller's external anchor
can defeat it. For decisions that must survive storage rollback, retain the
`ledger_tail_sha256` outside the run directory, for example in the downstream
sealed evidence or a reviewed Git commit, then supply it on the next check.
