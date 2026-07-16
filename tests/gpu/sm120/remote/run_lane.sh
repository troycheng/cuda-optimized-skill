#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_E2E_ARTIFACTS:?set CUDA_E2E_ARTIFACTS to a writable artifact directory}"
: "${CUTLASS_PATH:?set CUTLASS_PATH to the pinned CUTLASS checkout}"

export CUDA_SM120_E2E=1
python3 -m unittest tests.gpu.sm120.test_sm120_acceptance -v
