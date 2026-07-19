# RTX 5090 / SM120 opt-in acceptance

This suite runs real CUDA work and is skipped unless `CUDA_SM120_E2E=1` is
set. It covers:

- reference correctness and timing distributions for Triton, native CUDA, and
  CUTLASS;
- an identical Triton baseline/candidate measured by real randomized paired
  timing, persisted to JSONL, recomputed, and required to remain
  `inconclusive`;
- a small user-owned Python workload adapter that executes the candidate on the
  GPU through the production outer evaluator, persists its automatic
  `workload_paired_samples` evidence, and verifies that an inconclusive outer
  result cannot be promoted globally;
- a V2.4 workload-controller round that freezes a deliberately slow real Triton
  workload, collects a normalized GPU probe, registers a project-scoped
  ChangeSet, replaces redundant launches, runs paired workload evaluation, and
  requires a deterministic promotion with no host mutation;
- a V3 long-run calibration round that freezes the workload contract, measures
  eight real identical-kernel pairs, binds the calibration to source,
  environment, and contract identities, and refuses to leave `CALIBRATING`
  when noise or minimum detectable effect is too high;
- a target-bounded Nsight Compute attempt. It must either collect real metrics
  with readable counters or record exactly `ERR_NVGPUCTRPERM`; no other
  degraded result is accepted. The test never adds capabilities or changes
  driver policy.

The no-op check uses the production `run_paired`, `classify_pairs`, and
`write_paired_samples` APIs. The workload check enters through the production
`evaluate_outer_candidate` seam and recomputes the persisted pairs with
`classify_recorded_pairs`; it does not manufacture outer-loop evidence.

## Local and current-lane execution

Without opt-in, nine CPU helper regressions pass and all six GPU tests are
reported as skipped:

```bash
python3 -m unittest tests.gpu.sm120.test_sm120_acceptance -v
```

On the isolated checkout, a current host environment can run the same suite
directly:

```bash
cd /data/tcheng/cuda-skill-e2e/v2.2/repo
CUDA_VISIBLE_DEVICES=1 \
CUDA_SM120_E2E=1 \
CUDA_E2E_ARTIFACTS=/data/tcheng/cuda-skill-e2e/v2.2/artifacts/current-host \
CUTLASS_PATH=/data/tcheng/cuda-skill-e2e/deps/cutlass \
python3 -m unittest tests.gpu.sm120.test_sm120_acceptance -v
```

## Disposable current and compatibility containers

`remote/run_lane.sh` accepts `current` or `compat` (default). It requires the
exact repository path `/data/tcheng/cuda-skill-e2e/v2.2/repo`, an artifact lane
below `/data/tcheng/cuda-skill-e2e/v2.2/artifacts`, and the physical CUTLASS
checkout `/data/tcheng/cuda-skill-e2e/deps/cutlass` with both
`include/cutlass/cutlass.h` and `include/cutlass/version.h`. CUTLASS must not
overlap the repository or `/data/vllm-opt`, and the version header must report
the validated `4.6.1` release. The artifact lane must be fresh; only a regular
`run.log` created by an outer `tee` is allowed to pre-exist.

The runner checks the selected GPU for compute processes before image
inspection and again immediately before `docker run`; query failures stop the
run. It drops all Linux capabilities, disables networking, and mounts both the
repository and CUTLASS read-only. Each test copies the complete clean fixture
tree to its artifact-only `workspace/` first, so relative Python dependencies
remain available while compiler binaries and evidence stay out of the
repository.

```bash
cd /data/tcheng/cuda-skill-e2e/v2.2/repo
CUDA_E2E_GPU=1 \
CUDA_E2E_ARTIFACTS=/data/tcheng/cuda-skill-e2e/v2.2/artifacts/current \
CUTLASS_PATH=/data/tcheng/cuda-skill-e2e/deps/cutlass \
tests/gpu/sm120/remote/run_lane.sh current

CUDA_E2E_GPU=1 \
CUDA_E2E_ARTIFACTS=/data/tcheng/cuda-skill-e2e/v2.2/artifacts/compatibility \
CUTLASS_PATH=/data/tcheng/cuda-skill-e2e/deps/cutlass \
tests/gpu/sm120/remote/run_lane.sh compat
```

The defaults are the locally built
`cuda-skill-current:cuda13.3-triton3.7.1-ncu2026.2.1` image and the cached
`lmsysorg/sglang:latest-cu130-runtime` compatibility image. The runner resolves
the requested reference once to an immutable `sha256:` image ID, inspects and
runs that exact ID with `--pull never`, and saves both `requested_ref` and
`resolved_id` in `container-image.json`. Override the defaults with
`CUDA_CURRENT_IMAGE` or `CUDA_COMPAT_IMAGE` when a digest-pinned image is
available.

Use a distinct `CUDA_E2E_ARTIFACTS` directory for every lane. Expected durable
outputs include:

```text
artifacts/<lane>/
├── container-image.json
├── env.json
├── triton_vector/bench.json
├── triton_vector/workspace/...
├── cuda_reduction/bench.json
├── cuda_reduction/workspace/...
├── cutlass_gemm/bench.json
├── cutlass_gemm/workspace/...
├── paired_noop/
│   ├── paired_samples.jsonl
│   ├── statistics.json
│   └── workspace/...
├── workload_smoke/
│   └── iterv1/
│       ├── workspace/...
│       └── workload/<candidate-sha-prefix>/paired_samples.jsonl
├── workload_controller/
│   ├── workspace/...
│   └── run/
│       ├── probes/timeline.json
│       ├── diagnosis.json
│       ├── change_set.json
│       ├── review.json
│       ├── evaluation.json
│       └── decision.json
└── ncu_target/
    ├── workspace/...
    └── iterv1/
        ├── best_input.ncu.log
        └── ncu_top.json
```

Large profiler reports remain in the isolated artifact tree. `ncu
--query-metrics` is not treated as proof that hardware counters are readable.

## Recorded validation results

The V2.4 controller lane ran on a physical RTX 5090 on 2026-07-17. The current
container passed 13/13 checks in 34.302 seconds using immutable image
`sha256:a2d9d89bc4394eab3fadc62c6b5b3f739b6494c1f64c56f5ba5e6c008252a0e5`.
Its normalized probe recorded `gpu_busy_pct=0.0`; one metric was not enough to
assign a bottleneck, so diagnosis remained `inconclusive`. A fixture ChangeSet
removed two redundant Triton launches. Three paired A/B runs all passed the
checksum constraint and measured a 60.4616% latency improvement with a 95% CI
of [60.0894%, 61.4941%]. The optional reviewer was not configured, and the
deterministic controller promoted the change. NCU returned
`ERR_NVGPUCTRPERM`; the test did not change host permissions or driver policy.

The V3 lane ran on the same physical RTX 5090 on 2026-07-19. A fresh artifact
lane passed 15/15 checks in 34.307 seconds with the same immutable image. Eight
real identical-kernel pairs produced a 34.153% median symmetric-pair noise
estimate, a 36.712% upper confidence bound, and a 40.193% minimum detectable
effect against a frozen 0.5% practical effect. The calibrated state was
`yellow`, so the run remained `CALIBRATING`. This is a fail-closed result: the
controller detected that the environment could not support the requested
claim and did not begin candidate exploration. NCU again returned
`ERR_NVGPUCTRPERM`; no host permission or driver policy was changed.
