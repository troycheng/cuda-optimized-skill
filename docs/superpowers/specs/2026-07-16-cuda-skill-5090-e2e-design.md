# CUDA Kernel Optimizer 5090 End-to-End Validation Design

## Goal

Validate and improve `cuda-kernel-optimizer` on a real RTX 5090 until it is a
current, reliable, and useful iterative optimization skill rather than a
collection of plausible static guidance. The validation must cover Triton,
native CUDA, CUTLASS, and one real SM120 workload.

## Chosen approach

Use a layered workload matrix:

1. Validate environment discovery, correctness, timing, profiling, analysis,
   branch selection, ablation, SASS inspection, and reporting on controlled
   operators.
2. Cover memory-, latency-, and compute-dominated behavior across Triton,
   native CUDA, and CUTLASS.
3. Only after the controlled matrix passes, copy one vLLM SM120/FP8 operator
   into the isolated test area and use it as a realistic final workload.

This separates skill defects from application-specific build and dependency
problems while still requiring a real-project result.

## Repository and machine safety

- Modify and push only `troycheng/cuda-optimized-skill` on the existing
  `agent/update-cuda-skill-v2-1` branch or a new branch under that fork.
- Keep `KernelFlow-ops/cuda-optimized-skill` read-only. Do not create an
  upstream pull request without separate user approval.
- Do not edit `/data/vllm-opt` or its dirty working trees in place.
- Do not edit `/data/vllm-opt/third_party/cuda-optimized-skill`; that checkout
  points to the original repository.
- Create all remote artifacts under `/data/tcheng/cuda-skill-e2e/`.
- Run new disposable containers instead of installing into running service
  containers. Bind exactly one currently idle GPU; prefer GPU 1 and re-check
  utilization immediately before every test phase.
- Do not upgrade the NVIDIA driver, change Docker daemon settings, restart
  services, or change host-wide Python packages.

## Toolchain matrix

### Current validation lane

Target the current public releases as of 2026-07-16:

- CUDA Toolkit 13.3 Update 1
- Nsight Compute 2026.2.1
- Triton 3.7.1
- CUTLASS 4.6.1

Use a pinned container image digest, an isolated Python virtual environment,
and a tag-pinned CUTLASS checkout. Record the exact PyTorch build, compiler,
driver, image digest, Python packages, and Git revisions in `env.json`.

### Compatibility lane

Start a disposable container from the already-cached CUDA 13.0 runtime image
and validate against its current PyTorch 2.11 + Triton 3.6 stack. Use the host's
Nsight Compute 2026.1.1 where a container-local profiler is unavailable.

The compatibility lane must report missing or older capabilities accurately;
it must not silently claim current-lane validation.

### Installation policy

Permitted installations are limited to the test directory or disposable
container:

- CUDA development container images
- an isolated Python virtual environment and wheels
- Triton and compatible PyTorch packages
- CUTLASS source at a pinned release tag
- a local Nsight Compute package when the image lacks the required version
- small build and test utilities required by those components

If CUDA 13.3 or Nsight Compute 2026.2.1 cannot be installed without changing
the host driver or daemon, stop and report the exact incompatibility instead of
weakening host safety.

## Workload matrix

### Triton

- Vector operation: memory-bandwidth and launch-overhead path.
- Softmax or reduction: reduction, synchronization, and latency path.
- GEMM: tensor-core, tiling, pipeline, and compute path.

### Native CUDA

- A deliberately simple reduction with a correct Python reference.
- Candidate improvements should exercise coalescing, vectorized access,
  warp-level reduction, occupancy, and launch configuration decisions.

### CUTLASS

- An SM120 GEMM built from CUTLASS 4.6.1.
- Include one supported low-precision or block-scaled path when the exact
  toolkit and library build exposes it.
- Do not infer SM100 TCGen05/TMEM or SM90 WGMMA support for SM120.

### Real workload

- Copy one vLLM SM120/FP8 operator and the smallest reproducible reference or
  benchmark into the isolated directory.
- Preserve the original source and dependency tree untouched.
- Treat project build failures separately from skill workflow failures.

## Validation workflow

For every controlled workload:

1. Run environment discovery and preflight validation.
2. Establish correctness with deterministic inputs and explicit tolerances.
3. Benchmark independent CUDA event samples after warm-up.
4. Repeat the benchmark in multiple fresh processes and report distribution
   and variance, not one best sample.
5. Profile exactly one marked target invocation with Nsight Compute.
6. Confirm the report contains the intended kernel and usable SM120 metrics.
7. Run evidence-quality-aware bottleneck and roofline analysis.
8. Generate multiple candidate branches, benchmark all valid candidates, and
   select the champion only outside the measured noise band.
9. Profile the champion, run method ablation, and verify expected SASS or an
   explicit non-SASS validation rule.
10. Produce a final summary that separates method contribution,
    hyperparameter contribution, failed ideas, unsupported capabilities, and
    measurement uncertainty.

The real workload runs the same workflow after the controlled matrix passes.

## Acceptance criteria

- All existing CPU tests and structural skill validation pass.
- The environment probe identifies RTX 5090 as SM120 and records every relevant
  tool version without fabricated fallbacks.
- Every accepted candidate passes its reference correctness check; tolerances
  and precision changes are explicit.
- Timing uses independent samples. Repeated-process variance and the selected
  noise threshold are present in the artifacts.
- Nsight Compute captures only the marked target launch, produces a non-empty
  report, and either proves counter access or reports the exact permission
  failure.
- Roofline output distinguishes measured evidence from heuristics and never
  assigns a synthetic 100% gap to missing metrics.
- The full loop completes for at least one Triton, one CUDA, and one CUTLASS
  workload.
- At least two controlled workloads from different bottleneck classes show a
  repeatable improvement above the measured noise threshold. A deliberately
  weak baseline is acceptable, but the report must attribute the gain.
- The real workload either produces a repeatable improvement or an
  evidence-backed stop decision with rejected methods and reasons. A speedup is
  not fabricated as a mandatory success condition.
- No existing service, running container, source worktree, or upstream GitHub
  repository is modified.

## Skill improvement loop

Real-GPU failures become reproducible tests before skill code or guidance is
changed:

1. Preserve the failing command, environment manifest, minimal workload, and
   relevant profiler output.
2. Add a failing local or GPU integration test that demonstrates the missing
   behavior.
3. Implement the smallest skill or script correction.
4. Re-run the focused test, complete CPU suite, affected GPU workload, and
   installation validation.
5. Commit and push only to the user's fork branch.

GPU integration tests must be opt-in and clearly declare their required GPU,
toolchain, expected duration, and artifact directory.

## Artifacts

Each run stores:

- immutable source snapshot and reference implementation
- environment manifest and container image digest
- exact commands and return codes
- correctness results and timing samples
- `.ncu-rep`, normalized metrics, roofline output, and SASS evidence
- candidate and ablation results
- final summary and a machine-readable acceptance report

Large profiler reports remain on the 5090 machine. Small reproducible fixtures,
tests, scripts, and documentation are committed to the fork when they improve
the reusable skill.

## Failure handling

- If the selected GPU becomes occupied, stop the phase and select another idle
  GPU after a fresh utilization check.
- If profiling counters are blocked, first capture the real error. Do not add
  container privileges or modify driver settings without separate approval.
- If a latest package combination is incompatible, retain the failed manifest,
  test the nearest supported combination in a separate lane, and report the
  distinction.
- If a workload cannot satisfy the benchmark contract, classify that as a
  contract/usability gap and drive the same test-first correction loop.

## Official version references

- https://developer.nvidia.com/cuda-downloads
- https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html
- https://github.com/triton-lang/triton/releases/tag/v3.7.1
- https://github.com/NVIDIA/cutlass/releases/tag/v4.6.1
