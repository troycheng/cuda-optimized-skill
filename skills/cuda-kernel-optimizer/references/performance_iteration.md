# Performance-first iterations

Use this contract for every kernel, complete-workload, or serving optimization
round. It keeps measurement support bounded while leaving correctness,
benchmarking, and promotion to the existing workflow-specific components.

`iteration_guard.py` does not run a target, build, profiler, benchmark, or
correctness command. It does not modify project or host state. It does not promote
a candidate; it validates already produced records and derives the work class
and next action.

## Start with a falsifiable hypothesis

Before editing the target, record:

- a statement that a measurement can disprove;
- the mechanism being changed;
- target metric, direction, and minimum practical effect;
- relative paths that may change;
- frozen baseline and environment SHA-256 identities;
- one measurement path from a frozen registry;
- the round budget.

"Make the kernel faster" is not falsifiable. "Fuse class eligibility into the
mask kernel to reduce `latency_us` by at least 1%" is.

The candidate repeats the hypothesis mechanism and has different baseline and
candidate snapshot digests. Every changed path must be inside `mutation_scope`.
This catches empty or out-of-scope work, but it cannot understand every source
language. The AI must still reject comment-only, intentionally degraded, or
hypothesis-irrelevant candidates.

## Prevalidated measurement paths

Keep the registry outside the optimization round:

```json
{
  "schema_version": "cuda-optimizer/measurement-path-registry-v1",
  "paths": [
    {
      "id": "paired-kernel",
      "version": "1",
      "definition_sha256": "<64 lowercase hex characters>",
      "status": "validated"
    },
    {
      "id": "event-fallback",
      "version": "1",
      "definition_sha256": "<64 lowercase hex characters>",
      "status": "validated"
    }
  ]
}
```

The round binds `registry_sha256` plus one path's `id`, `version`, and
`definition_sha256`. Correctness and performance evidence repeat the same path,
baseline, candidate, and environment identities. Any mismatch fails closed.

A prevalidated fallback is another registry entry that existed before the
round. Do not create, edit, or validate a harness after the round exceeds its
support budget. Harness maintenance is a separate task with its own objective.

## Round record

Validate the full shape with `templates/performance_iteration.schema.json`.
The compact form below omits the repeated evidence identity fields only for
readability; the real record must include them.

```json
{
  "schema_version": "cuda-optimizer/performance-iteration-v1",
  "round_id": "iter-131-fast32",
  "lineage_id": "fast32-selector",
  "hypothesis": {
    "statement": "Fuse class eligibility into the mask kernel to lower latency.",
    "mechanism": "fuse-class-eligibility",
    "target_metric": "latency_us",
    "direction": "lower",
    "minimum_effect_pct": 1.0,
    "mutation_scope": ["kernels/fast32.py"]
  },
  "budget": {
    "round_seconds": 2700,
    "infrastructure_seconds": 120,
    "infrastructure_repairs": 0
  },
  "measurement_path": {
    "id": "paired-kernel",
    "version": "1",
    "definition_sha256": "<sha256>",
    "registry_sha256": "<sha256>"
  },
  "baseline": {
    "snapshot_sha256": "<sha256>",
    "environment_sha256": "<sha256>"
  },
  "candidate": {
    "candidate_id": "fast32-fused-mask",
    "baseline_snapshot_sha256": "<sha256>",
    "candidate_snapshot_sha256": "<different sha256>",
    "environment_sha256": "<sha256>",
    "mechanism": "fuse-class-eligibility",
    "changed_paths": ["kernels/fast32.py"]
  },
  "correctness": "<bound object or null>",
  "performance": "<bound object or null>"
}
```

Use the schemas, not the abbreviated example, when generating a record.

## Derived result

The classifier, not the AI, derives these fields:

| Work class | Required evidence | Claim |
|---|---|---|
| `candidate_evaluated` | A real candidate plus failed correctness, or passed correctness plus completed comparable timing | A candidate was tested; this is not automatically a gain |
| `measurement_blocked` | A real candidate but missing/failed correctness or timing | The selected measurement path blocked completion |
| `infrastructure_only` | Infrastructure time or repairs with no candidate | No optimization result |

The guard never emits `performance_gain`. It forwards a consistent
`confirmed_win` to the existing paired-evidence and promotion gate; that gate
still owns raw-sample validation, statistics, and the gain claim.
`confirmed_loss`, `inconclusive`, and `correctness_failed` remain completed
search results without a gain claim.

## Budget and forced stop

The infrastructure cap is:

```text
min(1200, floor(round_seconds * 0.15))
```

Allow one repair. At the exact time and repair limits, return to the candidate.
Above either limit, select a prevalidated fallback or emit `stop_direction`.

Pass previous decisions through JSON Lines with `--history`. If two consecutive
rounds in the same `lineage_id` are `measurement_blocked` or
`infrastructure_only`, select a prevalidated fallback or emit `stop_direction`.
Unrelated parallel candidates do not share the counter. Renaming or rebuilding
the current runner is not a fallback.

The output action is one of:

- `continue_candidate_search`;
- `proceed_to_existing_promotion_gate`;
- `return_to_candidate`;
- `switch_measurement_path`;
- `stop_direction`.

## Run the gate

```bash
python3 <skill>/scripts/iteration_guard.py check \
  --record iteration.json \
  --registry measurement-paths.json \
  --history iteration-decisions.jsonl \
  --out iteration-decision.json
```

The output is create-once. Append the complete decision object to the next
round's history only after the round closes.

## Report to the user

Lead with:

1. hypothesis and candidate change;
2. correctness and measured result;
3. keep/reject decision;
4. next performance action.

Mention infrastructure only when it blocks the measurement path. Never present
a runner fix, additional telemetry, an audit field, or a cleaner script as a
performance result.
