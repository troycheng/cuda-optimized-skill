# Validation status

This page describes where the project itself has been exercised. It does not
predict the speedup of a new workload.

## Automated checks

The current suite contains 935 tests. In the local CPU/static lane on
2026-07-19, 929 passed, six physical RTX 5090 opt-in tests were skipped, and
none failed. These checks cover input validation, state recovery, evidence
binding, shared-host guards, timeouts, restoration, capability retrieval,
stability calibration, audit cadence, and deterministic decision logic. They
do not validate the reader's CUDA environment.

## Physical GPU lane

The V3 RTX 5090 lane completed 15 of 15 checks in 34.307 seconds using immutable
container image
`sha256:a2d9d89bc4394eab3fadc62c6b5b3f739b6494c1f64c56f5ba5e6c008252a0e5`.
Its new long-run test measured eight real identical-kernel pairs. The observed
noise median was 34.153%, the upper confidence bound was 36.712%, and the
minimum detectable effect was 40.193%, above the frozen 0.5% practical effect.
The Controller therefore stayed in `CALIBRATING` instead of admitting an
optimization claim. Target-side NCU collection returned `ERR_NVGPUCTRPERM`;
the workflow reported the permission boundary and did not change the driver or
counter policy.

Exact commands and opt-in requirements are maintained in the
[RTX 5090 test guide](../tests/gpu/sm120/README.md). Toolchain and architecture
rules are listed in [Compatibility](compatibility.md).

## What these checks mean

They show that the project workflow, evidence files, and failure paths behaved
as recorded in those environments. They do not show that every CUDA, CUTLASS,
Triton, framework, or serving workload is supported, and they are not a general
performance guarantee.

Workload-specific results are kept separately in [Case studies](case-studies.md).
