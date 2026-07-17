# Systems and IR Evidence Coverage

Use this reference to choose evidence, not optimization methods. Method priority
and triggers remain in [optimization_catalog.md](optimization_catalog.md).
Architecture and toolchain routing remain in
[compatibility.md](compatibility.md). Endpoint claims follow
[serving_evidence_protocol.md](serving_evidence_protocol.md).

## System-path evidence

| Surface | First evidence | Confirmation |
|---|---|---|
| Host/device copies | Timeline with copy direction, bytes, stream, and dependency; allocator and transfer logs | End-to-end paired timing shows that removed or overlapped copies improve the target objective according to its configured direction without changing results. |
| Allocation | Allocation count, size, lifetime, pool behavior, and synchronization caused by allocation or release | The same workload shows lower allocator time or memory pressure with no leak or capacity regression. |
| Synchronization | Timeline and source correlation for device, stream, event, barrier, and implicit synchronization | The dependency is unnecessary or can be narrowed, and paired timing plus correctness confirm the change. |
| CUDA Graphs | Captured graph topology, replay eligibility, update behavior, and launch trace | Replay reduces host or launch overhead for the same operation sequence and input contract. |
| Launch density | Kernel count, duration distribution, gaps, launch rate, and CPU launch time | Fusion, batching, or replay improves the real objective without adding unacceptable latency or memory use. |

Do not infer a copy, allocation, or synchronization bottleneck from a single
kernel profile. Start with a system timeline, then bind the proposed change to
the affected operator or request path.

## CUTLASS and CuTe routing

| Question | Evidence to retain |
|---|---|
| Did dispatch select the intended kernel? | Exact CUTLASS version, target architecture, build flags, dispatch policy or selected kernel name, and generated code. |
| Does the layout match the data contract? | CuTe layout algebra or tensor shapes and strides, alignment, copy atom, boundary handling, and correctness cases. |
| Did the epilogue implement the intended fusion? | Epilogue visitor or operation graph, emitted loads/stores, numerical validation, and materialization count. |
| Is cluster execution valid and useful? | Exact cluster shape, launch support, occupancy or residency evidence, synchronization behavior, and paired timing. |
| Is the architecture route valid? | Explicit SM target and feature lookup. Never inherit WGMMA, TMA, TCGen05, TMEM, block scaling, or cluster support from numeric architecture ordering. |

Generated code confirms only the emitted mechanism. It does not establish that
the dispatch was exercised by the measured workload or that performance
improved.

## Triton autotune and IR routing

| Layer | Inspect | What it can establish |
|---|---|---|
| Autotune | Candidate configs, key fields, input domain, selected config, warm-up, timing policy, and cache key | Which configuration won for the tested key and environment. |
| TTIR | Tensor shapes, masks, reductions, program mapping, and high-level operations | Whether the source-level computation and mapping survived front-end lowering. |
| TTGIR | Layout conversions, warps, shared-memory use, barriers, and target-specific GPU operations | Whether the intended GPU layout and scheduling choices were lowered. |
| LLVM IR | Address spaces, vectorization, control flow, and target intrinsics | Whether lower-level transformations and intrinsic selection occurred. |
| PTX | Memory operations, barriers, matrix instructions, and target declarations | Which virtual-ISA mechanisms were emitted. |
| Generated code / SASS | Final instructions, spills, resource use, and source correlation | What the compiled binary contains for the exact architecture. |

Record the Triton version, source hash, compile options, device architecture,
driver/toolchain identity, and cache identity. A cached binary is evidence only
when its key and artifact hash bind it to those inputs. If an autotune cache is
reused across a changed shape distribution, runtime, or device, retune or prove
that the cache scope still matches.

## Special execution paths

Sparse, variable-length, fused, and serving paths must match the real request distribution.
Preserve the proportions and important correlations among
shapes, sequence lengths, sparsity patterns, batch sizes, cache states, and
concurrency. A dense fixed-shape benchmark cannot validate a sparse or
variable-length deployment path, and timing one fused operator cannot establish
a serving benefit.

## Decision checklist

- [ ] The system timeline identifies copies, allocation, synchronization,
      CUDA Graphs behavior, and launch density before a system-level change.
- [ ] CUTLASS/CuTe dispatch, layout, epilogue, cluster, and architecture
      evidence is bound to the measured binary.
- [ ] Triton autotune evidence includes its key and cache identity.
- [ ] TTIR, TTGIR, LLVM IR, PTX, or generated code is inspected only at the
      layer needed for the claim.
- [ ] Correctness and paired timing confirm performance; emitted code alone is
      not treated as a win.
- [ ] Sparse, variable-length, fused, and serving cases follow the real request distribution
      and the serving evidence protocol.
