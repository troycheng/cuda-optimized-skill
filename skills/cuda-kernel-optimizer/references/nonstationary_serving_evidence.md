# Nonstationary serving evidence

Use this gate when already-collected serving measurements may have been affected
by queue depth, load, cache state, power state, or another changing condition.
It asks whether baseline and candidate were observed under comparable paired
state. It does not run the workload and does not claim a performance gain.

## Freeze the comparison first

The design must contain at least four blocks. Each block contains one baseline
and one candidate segment, and the full plan balances AB/BA order using
`site_randomized_balanced`. The user or site planner owns that randomized plan;
the optimizer must not rearrange it after seeing results.

Before collection, create the anchor beside the design. The CLI writes it once
and refuses to replace it:

```bash
python3 <skill>/scripts/nonstationarity_guard.py init \
  --design nonstationarity-design.json \
  --anchor nonstationarity-anchor.json
```

The create-once anchor binds both the design file bytes and its canonical
semantic digest. `check` reads the design through that anchor; changing a
tolerance and recomputing the digest inside the series cannot rescue the run.

Automatic comparability requires fixed-duration time windows. Freeze the allowed
duration range, the number of burn-in rows before each timed row, the minimum
complete blocks, the primary metric, and every state dimension. Count windows
may be recorded, but this gate returns an inconclusive result for them because a
fixed request count can span materially different system states.

Each state dimension has two independent limits:

- pair tolerance compares the baseline and candidate timed rows in one block;
- phase tolerance compares the last burn-in row with its following timed row.

Both absolute and percent limits are enforced when both are present. `epsilon`
only stabilizes the percent denominator; it is not a variance allowance.

## Preserve chronology and raw evidence

The normalized series must follow the frozen block, segment, role, burn-in, and
timed-row order exactly. Missing, reordered, extra, or post-hoc excluded rows
make the input invalid. A row declared unusable remains visible and prevents its
block from satisfying the minimum.

Bind the normalized series to the canonical design digest and to a relative raw
source artifact. The CLI opens regular files without following symlinks and
rehashes the raw source before evaluation. A changed source cannot be evaluated
under the old identity.

## Read-only decision

```bash
python3 <skill>/scripts/nonstationarity_guard.py check \
  --anchor nonstationarity-anchor.json \
  --series nonstationarity-series.json
```

`comparable_paired_state` means the predeclared rows satisfy duration, pair,
phase, usability, and minimum-block checks. It permits the existing performance
gate to evaluate the metric; it is not itself evidence of a speedup.

`inconclusive_nonstationary` means the comparison must be redesigned or
recollected. The result names the failed blocks and recommends the next
experiment. It never changes host settings: `host_policy` remains
`recommend_only`.

Create-once protects against ordinary overwrite and post-hoc AI edits, not
storage rollback by an actor that can delete the directory and recreate its
history. Keep the anchor in a retained run directory or a site-owned immutable
artifact store when that threat matters.

Validate stored artifacts with:

- `templates/nonstationarity_anchor.schema.json`
- `templates/nonstationarity_design.schema.json`
- `templates/nonstationarity_series.schema.json`
- `templates/nonstationarity_verdict.schema.json`
