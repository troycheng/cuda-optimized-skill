# Validation status

This page describes where the project itself has been exercised. It does not
predict the speedup of a new workload.

## Automated checks

The current suite contains 813 CPU/static tests. On 2026-07-18, 808 passed,
five physical RTX 5090 opt-in tests were skipped, and none failed. These tests cover input
validation, state recovery, evidence binding, shared-host guards, timeouts,
restoration, and decision logic. They do not run a GPU and cannot validate the
reader's CUDA environment.

## Physical GPU lane

The recorded RTX 5090 lane completed 13 of 13 checks in 34.302 seconds. Target-
side NCU collection returned `ERR_NVGPUCTRPERM`; the workflow reported the
permission boundary and did not change the driver or counter policy.

Exact commands and opt-in requirements are maintained in the
[RTX 5090 test guide](../tests/gpu/sm120/README.md). Toolchain and architecture
rules are listed in [Compatibility](compatibility.md).

## What these checks mean

They show that the project workflow, evidence files, and failure paths behaved
as recorded in those environments. They do not show that every CUDA, CUTLASS,
Triton, framework, or serving workload is supported, and they are not a general
performance guarantee.

Workload-specific results are kept separately in [Case studies](case-studies.md).
