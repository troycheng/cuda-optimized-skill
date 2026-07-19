# Performance-first iterations

Use this contract for CUDA, CUTLASS, Triton, complete-workload, and serving
optimization rounds. It prevents measurement support from quietly becoming the
main task. Correctness, benchmarking, statistics, and promotion remain owned by
the existing V2.5 evidence workflow.

`iteration_guard.py` never runs a target, build, profiler, benchmark, or
correctness command. It freezes the lineage before round one, checks evidence
that already exists, classifies the round, and chooses the next action.

## 1. Freeze the lineage before editing

Prepare a registry containing measurement paths that already work. Each path
needs a different `definition_sha256`; renaming the same runner does not create
a fallback.

Also freeze an append-only list of invalid evidence IDs or artifact digests.
Invalid means unusable everywhere: later summaries, baselines, plots, direction
ranking, external review packets, and promotion inputs must reject any
intersection with that list. A corrected rerun gets a new identity.

```json
{
  "schema_version": "cuda-optimizer/measurement-path-registry-v1",
  "paths": [
    {
      "id": "paired-kernel",
      "version": "1",
      "definition_sha256": "<sha256>",
      "status": "validated"
    },
    {
      "id": "event-fallback",
      "version": "1",
      "definition_sha256": "<different sha256>",
      "status": "validated"
    }
  ]
}
```

Create the anchor once, before changing the candidate:

```bash
python3 <skill>/scripts/iteration_guard.py init \
  --registry measurement-paths.json \
  --baseline-source-sha256 <sha256> \
  --environment-sha256 <sha256> \
  --measurement-path paired-kernel@1 \
  --out iteration-anchor.json
```

The anchor embeds the baseline, environment, and validated registry. Its
canonical digest is the lineage identity; callers cannot reset the counter by
inventing another `lineage_id` inside a round.

## 2. State a falsifiable hypothesis

Before editing, record a claim a measurement can disprove:

- mechanism being changed;
- target metric and direction;
- minimum useful effect;
- source paths allowed to change;
- round and infrastructure budgets.

"Make the kernel faster" is not falsifiable. "Fuse class eligibility into the
mask kernel to lower `latency_us` by at least 1%" is.

```json
{
  "schema_version": "cuda-optimizer/performance-iteration-v1",
  "round_id": "iter-0001-fast32",
  "round_index": 1,
  "anchor_sha256": "<canonical sha256 of iteration-anchor.json>",
  "previous_decision_sha256": null,
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
    "definition_sha256": "<sha256>"
  },
  "candidate_declared": true,
  "evidence_manifest_sha256": "<V2.5 closure manifest sha256 or null>"
}
```

Validate the shape with `templates/performance_iteration.schema.json`. The
record does not contain self-reported correctness or timing fields. A completed
candidate must point to the create-once V2.5 closure produced by `evidence.py
seal`, `audit`, and `decide`.

Before sealing the V2.5 attempt, create the context binding and include it in
the attempt manifest as artifact kind `iteration_binding`:

```bash
python3 <skill>/scripts/iteration_guard.py binding \
  --anchor iteration-anchor.json \
  --record round-0001.json \
  --source-path kernels/fast32.py \
  --out evidence/iteration-binding.json
```

The binding ties the sealed source to the anchor, environment, exact
measurement-path implementation, and canonical hypothesis. Its measurement
path `definition_sha256` must equal the sealed `runner` artifact digest, and its
source path must be inside `mutation_scope`.

## 3. Close the round

Place the anchor and decisions in one run directory. Decision names are fixed
and create-once, so the chain cannot be silently reordered:

```bash
python3 <skill>/scripts/iteration_guard.py check \
  --anchor iteration-anchor.json \
  --record round-0001.json \
  --evidence-manifest evidence/manifest.json \
  --out round-0001-decision.json
```

For round two and later, set `previous_decision_sha256` to the canonical digest
of the preceding decision. The guard reads the preceding canonical file itself:
`round-0001-decision.json`, `round-0002-decision.json`, and so on. Do not supply
`--evidence-manifest` when the record contains a null evidence reference.

The guard uses no-follow file descriptors to rehash the V2.5 manifest, seal,
audit, decision, every sealed artifact, source, binding, and performance
verdict. A different 64-character string or an inline `passed` field cannot
manufacture `candidate_evaluated`.

## Derived result

| Work class | Mechanical requirement | Meaning |
|---|---|---|
| `candidate_evaluated` | A different source artifact in an integrity-passing V2.5 closure | A real candidate reached a terminal attempt; this alone is not a gain |
| `measurement_blocked` | Candidate declared, but no sealed closure | The selected path did not finish the experiment |
| `infrastructure_only` | No candidate and nonzero support work | No optimization result |

The guard never emits `performance_gain`. An integrity-passing
`confirmed_win` is forwarded to the existing promotion gate. That gate still
owns paired samples, statistics, workload constraints, and the gain claim.

## Budget and forced stop

Infrastructure time is capped at:

```text
min(1200, floor(round_seconds * 0.15))
```

One repair is allowed. Above either limit, switch to a different frozen path or
stop the direction. Two consecutive `measurement_blocked` or
`infrastructure_only` decisions in the same hash chain cause the same action.
Do not build or validate another runner inside the optimization round; registry
maintenance is a separate task.

Runner completeness has an end. Once one path produces a sealed, reproducible
correctness-plus-timing attempt, freeze it. Cosmetic output cleanup, extra
telemetry, and broader edge-case work are not optimization. Reopen the runner
only when a demonstrated defect can change correctness, sample identity,
pairing, contamination detection, cleanup safety, or the claimed metric.

The output action is one of:

- `continue_candidate_search`;
- `proceed_to_existing_promotion_gate`;
- `return_to_candidate`;
- `switch_measurement_path`;
- `stop_direction`.

## Report to the user

Lead with the hypothesis, candidate, correctness or terminal attempt, measured
result, keep/reject decision, and next performance action. Mention
infrastructure only when it blocks measurement. Never present a runner repair,
extra telemetry, an audit field, or cleaner support code as a performance result.
