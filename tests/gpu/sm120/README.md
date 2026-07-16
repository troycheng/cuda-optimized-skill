# RTX 5090 opt-in acceptance

These tests exercise the bundled benchmark against Triton, native CUDA, and
CUTLASS on an SM120 GPU. They are skipped during the default CPU suite.

Run in a disposable container with exactly one idle GPU:

```bash
docker run --rm --gpus device=1 \
  -e CUTLASS_PATH=/data/tcheng/cuda-skill-e2e/deps/cutlass \
  -e CUDA_E2E_ARTIFACTS=/data/tcheng/cuda-skill-e2e/artifacts/compatibility \
  -v /data:/data \
  -w /data/tcheng/cuda-skill-e2e/repo \
  lmsysorg/sglang:latest-cu130-runtime \
  bash tests/gpu/sm120/remote/run_lane.sh
```

Re-check GPU utilization immediately before the command. Do not point the
working directory at `/data/vllm-opt` or install into a running service
container.

The acceptance runner requires each backend to pass reference correctness and
to emit `samples_ms`, `median_ms`, `p95_ms`, `stddev_ms`, and `cv_pct`. Set a
different `CUDA_E2E_ARTIFACTS` directory for every toolchain lane so results are
never overwritten. Nsight Compute counter access is validated separately with
a real target-bounded profile; `ncu --query-metrics` alone is insufficient.
