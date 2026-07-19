# Single-variable software-stack audit

Use this protocol to test whether a Triton, TensorRT, CUDA, PyTorch, vLLM, or
container-stack upgrade improves the same workload. A comparison is confounded
if source lineage, backend binaries, server mode, model graph, build intent,
inputs, clocks, or measurement design also change.

## Freeze and vary

Freeze the source and model digests, build recipe, request corpus, correctness
contract, benchmark design, model configuration, custom backend, GPU, driver,
and clock policy. Vary only the declared software-stack identity.

Plugins, engines, timing caches, tactics, and compiled kernels are derived
artifacts. Rebuild them separately inside each stack, starting with empty
stack-local timing caches. A cross-version deserialization error is
compatibility evidence, not a performance result.

Validate the design before timing:

```bash
python3 <skill>/scripts/version_audit.py --input version-audit.json --out version-audit-report.json
```

## Gate order

1. Build the same source independently in both stacks.
2. Confirm each stack is self-repeat stable on representative real inputs.
3. Check both stacks against the frozen semantic and tolerance envelope.
4. Stop before timing if correctness fails.
5. Collect alternating standalone pairs before serving screens.
6. Attribute only what the design identifies. A joint image upgrade measures a
   stack effect, not a pure Triton or TensorRT effect.

## Invalid evidence

Maintain an append-only `invalid_evidence_ids` list. Anything on the list is
unusable in summaries, priors, baselines, plots, external review, and promotion.
A corrected run receives a new ID; it does not rehabilitate the invalid record.

## Stop conditions

Stop when a frozen field differs, derived artifacts are reused across stacks,
correctness or self-repeat stability fails, the candidate loses the declared
screen, or the environment is not comparable. Keep identities, build logs,
correctness artifacts, raw alternating samples, and the terminal decision.
