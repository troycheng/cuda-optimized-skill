# CUDA Ecosystem Compatibility

Validated on **2026-07-15**. These are validation targets, not hard minimums.
Probe the installed environment before selecting an optimization path.

| Component | Validation target | Runtime rule |
|---|---|---|
| CUDA Toolkit | CUDA Toolkit 13.3 | Use `nvcc --version` and compile a minimal target-specific kernel. |
| Nsight Compute | Nsight Compute 2026.2.1 | This release line supports CUDA 13.3. Query metric metadata, then run a real target profile to establish counter access. |
| Triton | Triton 3.7.1 | Inspect callable signatures before using optional or experimental arguments. |
| CUTLASS | CUTLASS 4.6.1 | Read `cutlass/version.h` or `cutlass.version`; compile the selected C++ or CuTe DSL example for the exact target. |

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
