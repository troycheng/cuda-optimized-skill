# cuda-kernel-optimizer V2.2

**English** | [简体中文](README.zh-CN.md)

A Codex-compatible skill for trustworthy CUDA, CUTLASS, and Triton
optimization. V2.2 uses a dual-loop workflow: the inner loop proves kernel
correctness and speed with paired measurements; the optional outer loop proves
the result on a real, user-owned workload.

This repository is a skill package, not a standalone optimizer. The agent reads
`SKILL.md`; deterministic scripts freeze inputs, enforce budgets, collect
evidence, make promotion decisions, and preserve resumable artifacts.

## Install in Codex

Install the maintained fork from `main`:

```bash
python3 "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-installer/scripts/install-skill-from-github.py" \
  --repo troycheng/cuda-optimized-skill \
  --ref main \
  --path skills/cuda-kernel-optimizer
```

The installer does not overwrite an existing skill. Back up or move the old
directory first, install again, then restart the Codex session.

```bash
cd skills/cuda-kernel-optimizer
```

## What V2.2 changes

- **Dual-loop evidence**: kernel microbenchmark evidence and real-workload KPI
  evidence are reported separately.
- **User-owned workload**: the skill never discovers, downloads, or invents a
  representative workload. End-to-end claims require one of three explicit
  workload inputs.
- **Budget presets**: `balanced` is the default; wall-clock, branch, round, pair,
  candidate, case, and sanitizer limits are frozen at setup.
- **Paired verdicts**: randomized AB/BA blocks, telemetry gates, confidence
  intervals, and a minimum practical effect replace fastest-sample promotion.
- **Promotion authority**: only `decision.json` can advance the best candidate.
  An `inconclusive` result never promotes.
- **Durable evidence**: frozen manifests, checkpoints, compiler provenance, raw
  `paired_samples.jsonl`, and a two-layer summary make conclusions auditable and
  recomputable.

## Budget presets

`balanced` is used when the user does not choose a preset.

| Preset | Max seconds | Branches | Max rounds | Min pairs | Max pairs | Outer candidates | Max cases | Sanitizer |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `quick` | 2700 | 4 | 2 | 20 | 50 | 1 | 3 | targeted |
| `balanced` (default) | 10800 | 8 | 4 | 20 | 100 | 2 | 10 | targeted |
| `thorough` | 36000 | 16 | 8 | 30 | 200 | 3 | unlimited | full |

Use `--budget custom` only with all required explicit limits. The scheduler
reserves shutdown time, stops admitting new stages before the deadline, and
writes a resumable checkpoint.

## Inputs

Always provide:

1. A baseline `.cu` or Triton `.py` kernel.
2. A Python reference exposing `reference(**kwargs)`.
3. The signature dimensions as JSON.

Optionally provide exactly one real workload form:

- `--workload ./workload.py`: Python adapter. Start from
  `skills/cuda-kernel-optimizer/templates/workload.py`.
- `--workload-cmd 'command ...' --objective ./objective.json`: a command parsed
  without a shell plus an explicit objective.
- `--workload-manifest ./workload.json`: strict manifest containing the source,
  objective, and cases.

The objective schema is
[`templates/objective.schema.json`](skills/cuda-kernel-optimizer/templates/objective.schema.json).
It declares one primary metric, its direction and minimum effect, plus allowed
regression for every constraint.

Without a workload the run is kernel-only. A confirmed kernel improvement may
produce `kernel_only_win`, which does not claim application throughput or
latency. With a workload, `end_to_end_win` requires both a confirmed kernel win
and a confirmed primary-KPI win with every constraint passing.

## Quick start

Kernel-only:

```bash
python3 skills/cuda-kernel-optimizer/scripts/orchestrate.py setup \
  --baseline ./gemm.cu \
  --ref ./ref.py \
  --dims '{"M":4096,"N":4096,"K":4096}' \
  --budget balanced
```

Full mode with a Python workload:

```bash
python3 skills/cuda-kernel-optimizer/scripts/orchestrate.py setup \
  --baseline ./gemm.cu \
  --ref ./ref.py \
  --dims '{"M":4096,"N":4096,"K":4096}' \
  --budget balanced \
  --workload ./workload.py
```

The agent then creates the requested branch candidates and closes each round:

```bash
python3 skills/cuda-kernel-optimizer/scripts/orchestrate.py close-iter \
  --run-dir ./run_YYYYMMDD_HHMMSS --iter 1
```

After an interruption, validate frozen inputs and inspect the next unfinished
stage without replaying completed work:

```bash
python3 skills/cuda-kernel-optimizer/scripts/orchestrate.py resume --run-dir \
  ./run_YYYYMMDD_HHMMSS
```

Finalize after the decision stage completes:

```bash
python3 skills/cuda-kernel-optimizer/scripts/orchestrate.py finalize \
  --run-dir ./run_YYYYMMDD_HHMMSS
```

## How promotion works

The inner loop gates candidates through reference correctness, configured
sanitizers, compiler/SASS evidence, and randomized paired baseline/candidate
measurements. A candidate is shortlisted only after a `confirmed_win` whose
finite confidence interval clears the configured minimum effect.

The outer loop runs only for a user-provided workload. It collects paired
baseline/candidate observations across the frozen cases, evaluates the primary
metric and constraints, and emits the terminal decision. Loss, timeout,
malformed evidence, failed constraints, and `inconclusive` all retain the
current best.

## Artifact tree

Exact files vary by backend and outcome; optional evidence is never represented
as successful when it is missing.

```text
run_YYYYMMDD_HHMMSS/
├── manifest.json                   # frozen inputs, policy, input_hash
├── state.json                      # candidate registry and history
├── checkpoint.json                 # durable resume boundary
├── env.json                        # GPU and toolchain snapshot
├── workload/spec.json              # frozen workload snapshot or null
├── baseline/
│   ├── <baseline>
│   └── bench.json
├── iterv1/
│   ├── analysis.md
│   ├── methods.json
│   ├── branches/
│   │   └── <candidate>/
│   │       ├── kernel.{cu,py}
│   │       ├── bench.json
│   │       ├── compiler_evidence/manifest.json
│   │       └── paired_samples.jsonl
│   ├── sanitizer.json
│   ├── sanitizer/*.json
│   ├── sass_check.json
│   ├── workload/<candidate-hash>/paired_samples.jsonl
│   ├── decision.json               # authoritative promotion decision
│   └── *.ncu.log                   # successful or degraded profiler log
├── iterv2/ ...
└── summary.md                      # separate kernel/workload conclusions
```

Raw pair files include frozen candidate identity and classifier configuration,
so the confidence result can be recomputed. `summary.md` links the evidence and
states whether profiler, sanitizer, compiler, or workload coverage degraded.

## RTX 5090 validation and NCU permissions

On 2026-07-16 the opt-in SM120 matrix passed correctness and timing artifact
checks on a physical RTX 5090 for Triton, native CUDA, and CUTLASS.

| Lane | CUDA compiler | Triton | CUTLASS | Nsight Compute | Result |
|---|---:|---:|---:|---:|---|
| Compatibility | 13.0.1 | 3.6.0 | 4.6.1 | 2025.3.1 | 3/3 backends passed |
| Current | 13.3.73 | 3.7.1 | 4.6.1 | 2026.2.1 | 3/3 backends passed |

The host returned `ERR_NVGPUCTRPERM` for hardware counters in both lanes. The
skill preserves the command, return code, and log, records unavailable counter
coverage, and continues with other evidence. No privilege, container
capability, or driver policy was changed. NCU evidence augments the decision
when counter access is available; it is not required for correctness or
paired timing. `ncu --query-metrics` alone does not prove counter permission.

See [`tests/gpu/sm120/README.md`](tests/gpu/sm120/README.md) for the opt-in GPU
tests and
[`references/compatibility.md`](skills/cuda-kernel-optimizer/references/compatibility.md)
for validated versions and architecture routing.

## Runtime requirements

- CUDA GPU with a working driver and `nvidia-smi`.
- Python 3.10+ with CUDA-enabled `torch`; install `triton` for Triton kernels.
- `nvcc` for CUDA/CUTLASS and `cuobjdump` for SASS evidence.
- CUTLASS headers through `$CUTLASS_PATH` or `$CUTLASS_INCLUDE_DIR` when used.
- Optional `ncu`; lack of counter permission is a recorded degraded mode.

The generic benchmark driver is bundled. The skill does not redistribute CUDA,
CUTLASS, Triton, or Nsight Compute.

## References

- [Formal workflow](skills/cuda-kernel-optimizer/SKILL.md)
- [Annotated walkthrough](skills/cuda-kernel-optimizer/examples/walkthrough.md)
- [Compatibility](skills/cuda-kernel-optimizer/references/compatibility.md)
- [Optimization catalog](skills/cuda-kernel-optimizer/references/optimization_catalog.md)
- [NCU metrics guide](skills/cuda-kernel-optimizer/references/ncu_metrics_guide.md)
- [Sanitizer policy](skills/cuda-kernel-optimizer/references/sanitizer_policy.json)

## License / attribution

This skill is independent of and does not redistribute CUTLASS, Triton, or
Nsight Compute. Install those dependencies separately.
