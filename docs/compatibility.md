# Compatibility

Kernel optimization needs Python 3.10+, a working CUDA GPU and driver, and the
toolchain required by the target implementation.

| Path | Requirement | Boundary |
|---|---|---|
| CUDA C++ | `nvcc`, compatible driver/toolkit, target architecture | Generated binary and compiler evidence must be bound to the tested source |
| CUTLASS / CuTe | Compatible CUTLASS checkout and architecture support | Public APIs and target-specific routing take precedence over version labels alone |
| Triton | Compatible Python, PyTorch, Triton, and GPU target | Autotune, IR, launch configuration, and generated binary identity may all matter |
| Nsight Compute | Compatible `ncu` for profiling or report import | Counter access is optional; unavailable access must be reported explicitly |

## RTX 5090 and SM120

The repository includes an opt-in physical RTX 5090 lane. It is not run by the
default CPU/static test command. Historical target-side profiling returned
`ERR_NVGPUCTRPERM`; the workflow recorded that degradation without changing
permissions or driver policy.

## NCU report import

Read-only report analysis needs a compatible Nsight Compute executable and an
existing report file. It does not launch the profiled program and cannot prove
that the current host can collect counters.

Exact observed versions, architecture capability rules, Triton and CUTLASS
routing, and primary upstream sources are maintained in the
[canonical compatibility reference](https://github.com/troycheng/cuda-optimized-skill/blob/main/skills/cuda-kernel-optimizer/references/compatibility.md).

The physical GPU fixture and opt-in commands are documented in the
[RTX 5090 test guide](https://github.com/troycheng/cuda-optimized-skill/blob/main/tests/gpu/sm120/README.md).
