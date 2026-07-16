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
