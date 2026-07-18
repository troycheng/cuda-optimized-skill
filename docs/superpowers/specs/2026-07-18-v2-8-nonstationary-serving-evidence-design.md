# V2.8 nonstationary serving evidence

## Purpose

V2.8 prevents an AI from treating time-varying serving conditions as a kernel
or runtime improvement. It evaluates an already collected, chronological
baseline/candidate experiment. It does not generate load, rerun a model, group
windows after seeing the result, or claim a speedup.

The result answers one question: are the baseline and candidate observations
comparable enough to enter the existing performance evidence gate?

## Why the existing stability check is not enough

Triton Performance Analyzer declares stability from the recent three windows
when both latency and throughput remain within a configured max/min band. That
is useful for one operating point, but it does not prove that baseline and
candidate saw the same cache state, queue pressure, request mix, or concurrent
load. Two internally stable periods can still belong to different regimes.

V2.8 therefore checks paired operating state across a predeclared switchback
schedule. It does not replace Perf Analyzer, Model Analyzer, V2.5 evidence, or
the user-supplied workload.

## Boundary

`nonstationarity_guard.py` is read-only. It consumes:

- one strict design JSON created before measurement;
- one strict chronological series JSON from a user or site-owned collector;
- one raw source artifact referenced by the normalized series.

The CLI rehashes the raw source and requires its path and digest to match the
series. It proves byte identity, schedule conformance, and deterministic state
comparability. It cannot prove that arbitrary normalized values were correctly
derived from an opaque profiler format. The user or site collector is the
trusted producer; an adversarial producer requires an external signature or
sealed manifest.

The tool never changes host settings. Host-level remedies remain recommendations.

## Predeclared block design

The design contains:

- a metric name, unit, and direction;
- a closed baseline role and candidate role;
- at least four planned blocks;
- a randomized-balanced assignment from the user or site planner, recorded as
  an `AB` or `BA` order for every block, with counts differing by at most one;
- the required number of burn-in observations before each timed observation;
- a time-window duration range that every row must satisfy;
- the minimum number of complete blocks;
- state dimensions with separate pair and burn-in-to-timed phase tolerances.

Each block contains one baseline segment and one candidate segment in its
declared order. Every segment contains the declared number of burn-in rows and
exactly one timed row. The chronological series must match the plan exactly.
Rows cannot be deleted, reordered, relabeled, or reassigned to another block.

The AI cannot choose or reorder assignment after measurement. Balanced `AB` and
`BA` blocks prevent a display label or a fixed early/late role from determining
the result. Burn-in is explicit because switchback experiments
can be biased by carryover when a system has not mixed into the next state.

## State comparability

For each complete block and each declared state dimension, the guard checks both
the timed baseline/candidate pair and the transition from the last burn-in row
to the timed row of each segment:

```text
absolute_delta = abs(candidate - baseline)
percent_delta = 100 * absolute_delta / max(abs(baseline), epsilon)
```

Every declared tolerance must pass. A design may use an absolute tolerance, a
percentage tolerance, or both for each check. A stable burn-in followed by a
role-specific timed-phase step therefore fails closed even when the block order
is balanced. Missing, non-finite, duplicate, or undeclared state values are
invalid input.

Automatic comparability uses fixed-duration time windows. The design freezes
minimum and maximum accepted row duration, and every row records its observed
duration. Count-window evidence or duration outside the frozen range returns
inconclusive rather than silently exposing one role to a longer time regime.

The guard does not discover regimes or thresholds from observed performance.
State dimensions and tolerances are fixed before the run. Post-hoc clustering
cannot turn an inconclusive experiment into a comparable one.

## Verdicts

The machine result is one of:

- `comparable_paired_state`: the schedule is complete, role order is balanced,
  burn-in is present, and every timed pair satisfies every state tolerance;
- `inconclusive_nonstationary`: valid evidence exists, but one or more paired
  state, phase-shift, or duration checks fail, or too few complete blocks remain;
- invalid input: the CLI exits nonzero and emits no verdict artifact.

Both verdicts set `performance_gain_claimed: false`. A comparable result only
authorizes the downstream performance gate to evaluate the metric values. It
does not promote a candidate.

An inconclusive result includes closed reasons and a deterministic next-design
recommendation. It may recommend more balanced blocks, longer burn-in, or a
new predeclared state boundary. It never edits the current evidence or excludes
an inconvenient block.

## Failure behavior

Fail input validation on:

- duplicate JSON keys, unknown fields, non-finite values, or unsafe paths;
- a raw source digest mismatch or symlink;
- a plan/series order mismatch;
- duplicate or missing block/segment/timed identities;
- an unbalanced planned role order;
- state dimensions in the series that differ from the frozen design;
- a metric taxonomy mismatch.

Return `inconclusive_nonstationary`, not an input error, when valid observed
state exceeds a frozen tolerance or the number of comparable blocks is below
the frozen minimum.

## Sources

- [NVIDIA Triton Perf Analyzer CLI](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/docs/cli.html)
- [NVIDIA Triton Model Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_analyzer.html)
- [Switchback Experiments under Geometric Mixing](https://arxiv.org/abs/2209.00197)
- [Randomization Tests in Switchback Experiments](https://arxiv.org/abs/2602.23257)
