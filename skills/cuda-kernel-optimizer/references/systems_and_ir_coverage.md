# Systems, CUTLASS, and Triton IR Coverage

Use this reference when kernel timing points outside the kernel body, when the
implementation uses CUTLASS or Triton, or when generated-code evidence is
needed. Treat every item as a hypothesis to benchmark, not a guaranteed win.

## Contents

- Host-device data path
- CUTLASS routing
- Triton autotune and generated code
- Sparse, variable-length, and fused paths
- Promotion evidence

## 1. Host-device data path

Measure the full path with Nsight Systems before rewriting a fast kernel. A
serving regression can be dominated by copies, synchronization, allocation,
launch gaps, runtime scheduling, or queueing even when Nsight Compute shows a
healthy device kernel.

- Minimize transfers and batch small transfers where latency permits.
- Page-locked host memory can improve transfer bandwidth and is required for
  truly asynchronous host-device copies, but it is scarce. Allocate it
  deliberately, reuse it, and measure the system-wide effect.
- Copy/compute overlap needs pinned memory, non-default streams, and hardware
  support. Verify overlap on a timeline instead of inferring it from API use.
- Zero-copy mapped host memory can help an integrated GPU or a discrete-GPU
  workload that reads each coalesced location once. Repeated PCIe reads are
  usually a poor trade for copying into device memory.
- Unified Memory prefetch and advice are hints. Verify page-fault, migration,
  and residency behavior for the actual access sequence.
- Constant memory is useful when a warp reads a small, uniform address. It can
  serialize divergent addresses.
- Texture objects are candidates for spatially local read-only sampling, not a
  generic replacement for ordinary global loads.

Reference: NVIDIA CUDA C++ Best Practices Guide, especially data transfer,
zero-copy, and memory-space guidance:
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

Unified Memory reference:
https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html

## 2. CUTLASS routing

Do not route from an old example number or assume numeric SM ordering implies
feature compatibility. CUTLASS changes quickly and Blackwell product families
can expose different instruction sets.

1. Detect the exact device architecture, CUDA toolkit, CUTLASS version, input
   types/layouts, accumulator type, alignment, shape distribution, and epilogue.
2. Use the installed CUTLASS profiler to enumerate and benchmark compatible
   operations. Build a narrow operation subset when possible.
3. Compare candidates using identical inputs, tolerance, warmup, repetitions,
   clocks, and workspace policy.
4. Explore the mechanism supported by evidence: mainloop schedule and stages,
   tile/cluster shapes, persistent or Stream-K scheduling, grouped/batched
   routing, split-K, layout/swizzle, and epilogue fusion.
5. Validate against the project reference and an appropriate library baseline.
   A profiler result is not automatically equivalent to the integrated call.

CUTLASS overview:
https://docs.nvidia.com/cutlass/latest/overview.html

CUTLASS profiler:
https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html

## 3. Triton autotune and generated code

Autotune discovery and stable profiling are separate phases.

- Make the autotune `key` include every argument whose value can change the
  best configuration.
- Use representative production shapes and a bounded search space. Preserve
  the selected config and its Triton/CUDA/device context.
- Side-effecting kernels may need `reset_to_zero` or `restore_value` during
  tuning because every candidate runs.
- `cache_results` can reduce repeated tuning cost, but a cache entry is still
  scoped evidence, not proof across versions or devices.
- Freeze the chosen config before NCU collection so multiple autotune launches
  do not contaminate the target report.

Triton autotune API:
https://triton-lang.org/main/python-api/generated/triton.autotune.html

When source intent and NCU results disagree, inspect generated artifacts. With
a current Triton checkout, use `TRITON_KERNEL_DUMP=1` and `TRITON_DUMP_DIR` to
retain compilation output. `USE_IR_LOC` can target a dumped IR file for
reproduction. Compare TTIR/TTGIR, LLVM IR, PTX, cubin metadata, and—when a
supported tool path exists—SASS. Check layout conversion, vector width,
masking, pipeline depth, MMA lowering, register pressure, spills, shared-memory
footprint, and unexpected helper kernels.

Current Triton debugging and dump controls:
https://github.com/triton-lang/triton/blob/main/README.md

## 4. Sparse, variable-length, and fused paths

- Sparse acceleration only counts if the input representation, metadata,
  preprocessing, kernel, and downstream conversion together save end-to-end
  time. Verify supported sparsity format and architecture; do not cite a
  theoretical multiplier as measured speedup.
- For variable-length or block-sparse work, bucket or schedule by actual work,
  skip fully masked blocks, and measure tail imbalance.
- Fusion can eliminate launch and global-memory round trips, but may increase
  registers, shared memory, compilation cost, and latency variance. Compare
  both the fused kernel and the full serving path.

## 5. Promotion evidence

Keep these conclusions separate:

- `benchmark winner`: fastest validated timing under the recorded benchmark
  contract; it may lack full profiler evidence.
- `fully-profiled winner`: validated candidate with a successful current NCU
  report and an auditable artifact chain.
- `deployment winner`: fully-profiled operator that also wins a clean matched
  serving A/B test under the serving evidence protocol.

Never promote a lower evidence tier by wording alone.
