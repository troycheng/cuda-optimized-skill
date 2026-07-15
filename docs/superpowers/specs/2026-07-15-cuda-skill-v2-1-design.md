# CUDA Kernel Optimizer Skill v2.1 Design

## Goal

Make the existing CUDA/CUTLASS/Triton optimization skill reliable in current
Codex environments while preserving its useful iterative workflow: validate,
profile, branch, benchmark, attribute, and verify generated code.

## Repository safety

- Develop only in `troycheng/cuda-optimized-skill`.
- Keep `KernelFlow-ops/cuda-optimized-skill` as a read-only `upstream` remote.
- Do not create an upstream pull request without separate user approval.
- Push the implementation branch only to the fork's `origin` remote.

## Scope

### Skill packaging and portability

- Fix the invalid YAML frontmatter and shorten the trigger description.
- Replace Claude-specific wording and requests for hidden chain-of-thought with
  agent-neutral, evidence-backed decision records.
- Add `agents/openai.yaml` with concise Codex UI metadata.
- Keep detailed optimization guidance in references rather than expanding
  `SKILL.md`.

### Benchmark correctness

- Move optional heavyweight imports behind CLI parsing so `--help` works before
  PyTorch is installed.
- Collect independent timing samples instead of duplicating one aggregate
  average as minimum, median, and maximum.
- Preserve current correctness checks and output schema where practical.

### Nsight Compute targeting

- Mark only the measured target-kernel region and configure Nsight Compute to
  ignore setup, allocation, random initialization, and warm-up launches.
- Prefer a CUDA profiler start/stop range because it works for raw CUDA,
  CUTLASS, library-backed kernels, and Triton without guessing kernel symbols.
- Profile one deterministic target invocation per report unless a caller
  explicitly requests more.
- Detect unsupported metrics by querying the installed Nsight Compute version
  rather than assuming every metric exists on every architecture.

### Bottleneck analysis

- Stop presenting utilization-gap heuristics as a complete Roofline model.
- Add measured or caller-provided operation counts and transferred bytes when
  available; otherwise label the result as degraded heuristic analysis.
- Do not allocate method budget to an axis whose measured gap is zero.
- Remove generic peak-value fallbacks for unknown GPUs from authoritative bound
  classification.

### Current CUDA ecosystem guidance

- Update public Triton guidance for 3.7 APIs: `input_precision`, tensor
  descriptors for TMA, and supported warp-specialization paths.
- Mark non-upstream arguments such as `acc_promote_cycles` as fork-specific and
  capability-gated.
- Extend architecture capabilities for SM103, SM110, SM120, and SM121 without
  treating numeric SM ordering as feature inheritance.
- Add CUTLASS 4.6 and CuTe DSL routing while retaining compatible C++ guidance.
- Document CUDA 13.3 and Nsight Compute 13.3 as the current validation targets,
  not hard minimum requirements.

## Test design

Tests run without a GPU unless explicitly marked as GPU integration tests.

- Frontmatter and `agents/openai.yaml` validation.
- `--help` smoke tests for every bundled script without PyTorch installed.
- Timing tests using a fake CUDA event implementation that proves samples are
  independent.
- Nsight command-construction tests that prove profiling starts only inside the
  target range.
- Budget-allocation tests covering one-axis, near-zero, tie, and cap cases.
- Metric parsing fixtures for long-form and wide-form Nsight Compute CSV.
- Architecture-capability tests that require an explicit feature set per SM:
  SM120/SM121 TMA and `tcgen05` paths are represented where supported, while
  SM100a-only features are not inherited by numeric SM ordering.
- Static reference checks for removed deprecated or fork-only API claims.

GPU end-to-end validation remains a separate check because this macOS host has
no CUDA device. The fork must clearly report that limitation rather than imply
GPU execution passed.

## Compatibility and migration

- Preserve current script names and main CLI arguments.
- Add new flags with safe defaults instead of breaking callers.
- Keep existing state JSON readable; new fields must be optional.
- Avoid rewriting the optimization catalog beyond claims that are obsolete,
  ambiguous, or unsafe.

## Non-goals

- No automatic pull request to the original repository.
- No complete rewrite of the optimization method catalog.
- No installation or mutation of a remote GPU environment.
- No claim that static macOS checks replace GPU validation.
