# CUDA Ecosystem Compatibility

Validated on **2026-07-17**. These are validation targets, not hard minimums.
Probe the installed environment before selecting an optimization path.

| Component | Validation target | Runtime rule |
|---|---|---|
| CUDA Toolkit | CUDA Toolkit 13.3 Update 1 | Use `nvcc --version` and compile a minimal target-specific kernel. |
| Nsight Compute | Nsight Compute 2026.2.1 | This release line supports CUDA 13.3. Query metric metadata, then run a real target profile to establish counter access. |
| Triton | Triton 3.7.1 | Inspect callable signatures before using optional or experimental arguments. |
| CUTLASS | CUTLASS 4.6.1 | Read `cutlass/version.h` or `cutlass.version`; compile the selected C++ or CuTe DSL example for the exact target. |

## Observed RTX 5090 lanes

Both isolated lanes passed the same V2.2 SM120 acceptance matrix on 2026-07-17:
seven safety/helper tests and four real-GPU tests for Triton, native CUDA,
CUTLASS, randomized identical paired timing, the production outer-workload
evaluator, and target-bounded NCU.

| Lane | Immutable image ID | nvcc | PyTorch | Triton | CUTLASS headers | ncu | Tests | Counter verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Current | `sha256:a2d9d89b...8252a0e5` | 13.3.73 | 2.11.0+cu130 | 3.7.1 | 4.6.1 | 2026.2.1 | 11/11 | `ERR_NVGPUCTRPERM` |
| Compatibility | `sha256:b810841f...37188a2` | 13.0.88 | 2.11.0+cu130 | 3.6.0 | 4.6.1 | 2025.3.1 | 11/11 | `ERR_NVGPUCTRPERM` |

The counter verdict is a host-policy result, not a toolchain compatibility
failure. A successful `ncu --query-metrics` is not counter-access proof. These
runs did not add `SYS_ADMIN`, run privileged containers, or change the driver.
The acceptance runner bound each lane to the recorded immutable image ID,
required a dedicated read-only CUTLASS 4.6.1 checkout, failed closed when GPU
occupancy could not be queried, and started from fresh artifact directories.

## Observed real-workload lane

An isolated user-provided vLLM binary workload ran in full mode with a
`balanced` policy limited to one round, two branches, one outer candidate, and
10,800 seconds. It completed in 2,232.43 seconds:

| Surface | Valid pairs | Estimate | 95% CI | Threshold | Status |
|---|---:|---:|---:|---:|---|
| Kernel paired timing | 100 | +26.3287% | [22.1801%, 30.6322%] | +0.5% | `confirmed_win` |
| `latency_us` workload | 100 | -0.0097% | [-0.0390%, 0.0365%] | +2.0% | `inconclusive` |

The authoritative full-mode verdict was `kernel_only_win`, so the baseline
remained the global best. The final checkpoint was `complete`, resume was
idempotent, and raw kernel and workload JSONL reproduced the stored statistics.
Host NCU 2026.1.1 had counter access and collected 140 metrics without
degradation. The adapter compared prebuilt baseline/optimized binaries; the
captured dispatch headers were byte-identical, so the result is binary A/B
evidence rather than source-level promotion proof. No source tree was modified.

Durable evidence is under
`/data/tcheng/cuda-skill-e2e/v2.2/artifacts/{current,compatibility,real}`. The
real run directory is
`real/orchestrator/run_20260717_043610_569950525`.

## Architecture capability rules

Architecture capabilities are explicit sets. **Never infer** TMA, WGMMA,
TCGen05, TMEM, block scaling, cluster, or CLC support from numeric SM ordering.
The `arch_feature_map` in `method_registry.json` is the machine-readable source
used by validation.

| Target | Important routing notes |
|---|---|
| `sm_90` | Hopper path: TMA, WGMMA, mbarrier, clusters, and warp-specialized schedules. |
| `sm_100` | Datacenter Blackwell path: TMA, TCGen05, TMEM, block scaling, and CLC. Do not inherit Hopper WGMMA. |
| `sm_103` | B300 Blackwell path: target explicitly and probe CUTLASS builders; TCGen05/TMEM and block-scaled paths are distinct from SM120. |
| `sm_110` | Thor target name for CUDA 13.0+; older toolkits used SM101. Probe the installed toolkit and CUTLASS version before compiling. |
| `sm_120` | GeForce Blackwell path: current CUTLASS has TMA warp-specialized and native block-scaled MMA schedules, but not SM100 TCGen05/TMEM or Hopper WGMMA. |
| `sm_121` | DGX Spark path shares major CUTLASS code with SM120, including current SM120/SM121 TMA collectives; do not inherit SM100 features. |

Use the exact compiler target (`sm_100a`, `sm_103a`, `sm_110a`, `sm_120a`,
or `sm_121a`) only when the selected implementation requires
architecture-specific features. For CUTLASS builds, set `CUTLASS_NVCC_ARCHS`
to the exact target and run `cutlass_profiler` or the corresponding unit test.

## Triton 3.7 guidance

- Use `tl.dot(..., input_precision="tf32")`; `allow_tf32` is deprecated.
- Use `tl.make_tensor_descriptor` or a host tensor descriptor for descriptor
  loads/stores and TMA-backed lowering. `tl.make_block_ptr` is a block-pointer
  abstraction and is not, by itself, proof that TMA was selected.
- Upstream `tl.range(..., warp_specialize=True)` is currently constrained to
  supported Blackwell GPUs and simple matmul loops. Treat Gluon warp
  specialization as experimental and validate generated code.
- `acc_promote_cycles` is fork-specific, not part of the upstream `tl.dot`
  signature. Check the callable signature before using it and keep an upstream
  fallback.

## CUTLASS 4.6 routing

- Prefer the CUTLASS 3.x C++ API when integrating into an existing C++/CUDA
  codebase or when a stable collective builder already covers the operation.
- Prefer CuTe DSL for Python-first kernel work, rapid layout experimentation,
  and current DSL examples. Treat experimental APIs such as `cute.compile_to`
  according to the release notes.
- Do not reuse an SM100 schedule on SM120/SM121. Select the explicit SM120
  dispatch policies and compile for the exact device target.
- The legacy Python code-generation package is named `cutlass_cppgen`; it is
  different from the CuTe DSL package.

## Nsight Compute probing

1. `ncu --query-metrics` establishes only that metric metadata can be queried.
2. Build a target-bounded report and inspect its return code and log to establish
   actual performance-counter access.
3. Query or collect only metrics supported by the installed GPU and NCU build;
   newer Blackwell SKUs can expose reduced metric sets.

## Primary references

- [CUDA 13.3 compiler targets](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-compiler-driver-nvcc/index.html)
- [PTX ISA target-family feature table](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#ptx-isa-version-history)
- [Nsight Compute 2026.2.1 release notes](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html)
- [Triton 3.7.1 release](https://github.com/triton-lang/triton/releases/tag/v3.7.1)
- [`tl.dot` API](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)
- [`tl.make_tensor_descriptor` API](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html)
- [`tl.range` warp-specialization constraints](https://triton-lang.org/main/python-api/generated/triton.language.range.html)
- [CUTLASS 4.6.1 release](https://github.com/NVIDIA/cutlass/releases/tag/v4.6.1)
- [CUTLASS changelog](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)
- [CUTLASS SM120 TMA dispatch policies](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/dispatch_policy.hpp)
