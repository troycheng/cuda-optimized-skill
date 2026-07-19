# Triton decode attention / GQA

Use this playbook only after metadata routing. Retrieval is not execution admission. Before execution, the Controller resolves `gate_requirements.pre_execution` against contract-bound artifacts. After execution, only `gate_requirements.promotion` can support promotion.

## Read the evidence first

- Freeze the real request distribution, Q/KV head mapping, dtype, strides, mask and scale semantics. Confirm which kernel is dispatched for every measured stratum.
- If any tail shape disagrees with the correctness oracle, stop performance work. Reproduce lengths around every block boundary and check K/V loads, invalid-logit masking, online-softmax reduction and stores.
- Separate short-context launch gaps from long-context KV traffic. Do not combine them in one candidate or choose a dispatch threshold after seeing results.
- `sm_120` is an exact target. Compile-probe every Triton feature used by the candidate. Do not infer support from Hopper, `sm_100`, or a generic “Blackwell” label.

## Admit a direction

Before grouping Q heads per KV head, inspect source and generated code to establish whether K/V are already reused. Nsys can establish kernel and launch-time shares; without NCU counters, DRAM reduction remains a hypothesis.

Choose one mechanism:

1. repair boundary handling and validate it without claiming a speedup;
2. reduce short-context launches or dispatch overhead;
3. reuse K/V work across a GQA group for long contexts;
4. change tile, warp or stage choices within one frozen shape stratum.

For grouped-GQA, preregister the expected benefit and the main counter-cost: extra accumulators can increase registers, spills and occupancy loss. Start with the smallest representative group and keep the baseline path available.

## Validate and stop

- Run correctness before timing, including boundary lengths, all head groups, NaN/Inf checks and the frozen tolerance against the user reference.
- Use randomized paired timing for the isolated path, then replay the real workload distribution. A kernel-only win is not a workload win.
- Inspect generated code and Nsys after measurement to explain the result. Treat cache effects as an alternative explanation, not proof of reduced DRAM traffic.
- Kill on any correctness or dispatch-identity failure. Close the mechanism when its removable-time upper bound is below the objective, repeated evidence rejects it, or the contract budget is exhausted.
- Never invent a time share, threshold or retry allowance. Read all three from the frozen contract or calibrated experiment design.

Sources are bound in `sources.json`: Triton fused attention and block-pointer load semantics, NVIDIA Blackwell and Nsys guides, and the fixed `kernel-skills` organization snapshot.
