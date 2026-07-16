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
cd "${CODEX_HOME:-$HOME/.codex}/skills/cuda-kernel-optimizer"
```

All commands below run from this installed skill root, so script paths start at
`scripts/`. Contributors using a repository checkout can instead run
`cd skills/cuda-kernel-optimizer` once and use the same commands.

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

A minimal Python manifest is:

```json
{
  "kind": "python",
  "source": "./workload.py",
  "objective": {
    "primary_metric": {"name": "p50_latency_ms", "direction": "lower"},
    "min_effect_pct": 1.0,
    "constraints": []
  },
  "cases": [{}]
}
```

`kind` must be `python` or `command`. A manifest requires `kind`, `source`, and
`cases`. Use one objective source: embedded objective or --objective, never both. For a Python
manifest, the embedded/external objective must match the adapter's `metrics()`.

The objective schema is
[`templates/objective.schema.json`](skills/cuda-kernel-optimizer/templates/objective.schema.json).
It declares one primary metric, its direction and minimum effect, plus allowed
regression for every constraint.

`kernel_only_win` confirms only the kernel result. It is the normal successful
outcome without a workload, but may also be the terminal outcome in full mode
after workload failure/loss/inconclusive evidence. It never advances the global
best in full mode. `end_to_end_win` requires both a confirmed kernel win and a
confirmed primary-KPI win with every constraint passing; it is the only
full-mode outcome that advances the global best.

## Quick start

Kernel-only:

```bash
python3 scripts/orchestrate.py setup \
  --baseline /path/to/gemm.cu \
  --ref /path/to/ref.py \
  --dims '{"M":4096,"N":4096,"K":4096}' \
  --budget balanced
```

Full mode with a Python workload:

```bash
python3 scripts/orchestrate.py setup \
  --baseline /path/to/gemm.cu \
  --ref /path/to/ref.py \
  --dims '{"M":4096,"N":4096,"K":4096}' \
  --budget balanced \
  --workload /path/to/workload.py
```

Setup freezes inputs, validates and seeds the baseline, and writes the initial
checkpoint. It does not profile the current best or create branch directories.
Open each iteration before the agent reads profiler evidence or writes branches:

```bash
python3 scripts/orchestrate.py open-iter \
  --run-dir /path/to/run_YYYYMMDD_HHMMSS --iter 1
```

`open-iter` attempts current-best profiling, computes Roofline evidence, and
creates the budgeted branch directories. The agent then writes candidates and
closes the round:

```bash
python3 scripts/orchestrate.py close-iter \
  --run-dir /path/to/run_YYYYMMDD_HHMMSS --iter 1
```

After an interruption, validate frozen inputs and inspect the next unfinished
stage without replaying completed work:

```bash
python3 scripts/orchestrate.py resume --run-dir \
  /path/to/run_YYYYMMDD_HHMMSS
```

Finalize after the decision stage completes:

```bash
python3 scripts/orchestrate.py finalize \
  --run-dir /path/to/run_YYYYMMDD_HHMMSS
```

## How promotion works

The real candidate order is correctness, randomized paired baseline/candidate
measurement, sanitizer processing of the confirmed shortlist, and SASS on the
final eligible candidate. Compiler provenance and SASS are evidence and method
classification, not hard promotion gates. Correctness and sanitizer remain
hard gates; changed source/artifact identity fails closed.

The outer loop runs only for a user-provided workload. It collects paired
baseline/candidate observations across the frozen cases, evaluates the primary
metric and constraints, and emits the terminal decision. A confirmed failed
hard constraint becomes `rejected_constraint`. Workload collection failure,
primary loss/inconclusive, or inconclusive constraint evidence after a confirmed
kernel win can terminate as `kernel_only_win`. All of these retain the global
best in full mode.

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
│   ├── workload/<candidate-hash-prefix>/paired_samples.jsonl
│   ├── decision.json               # authoritative promotion decision
│   └── *.ncu.log                   # successful or degraded profiler log
├── iterv2/ ...
└── summary.md                      # separate kernel/workload conclusions
```

Raw pair files include frozen candidate identity and classifier configuration,
so the confidence result can be recomputed. `summary.md` links the evidence and
states whether profiler, sanitizer, compiler, or workload coverage degraded.

## RTX 5090 validation and NCU permissions

V2.2 was validated on a physical RTX 5090 on 2026-07-17. Both isolated
containers ran the same 11-test matrix: seven safety/helper checks plus four
real-GPU checks covering Triton, native CUDA, CUTLASS, randomized identical
paired timing, the production outer-workload evaluator, and target-bounded NCU.

| Lane | Immutable image ID | nvcc | PyTorch | Triton | CUTLASS headers | NCU | Result |
|---|---|---:|---:|---:|---:|---:|---|
| Current | `sha256:a2d9d89b...8252a0e5` | 13.3.73 | 2.11.0+cu130 | 3.7.1 | 4.6.1 | 2026.2.1 | 11/11 passed |
| Compatibility | `sha256:b810841f...37188a2` | 13.0.88 | 2.11.0+cu130 | 3.6.0 | 4.6.1 | 2025.3.1 | 11/11 passed |

The capability-dropped containers returned `ERR_NVGPUCTRPERM` for hardware
counters. The tests accepted only that exact degraded result or a successful
profile with real metrics; no privilege, container capability, or driver policy
was changed. The runner used immutable image IDs, a dedicated read-only CUTLASS
4.6.1 checkout, fail-closed GPU-idle checks, and fresh artifact directories.

An additional isolated, user-provided vLLM binary workload was run in full mode
with the `balanced` budget (one round, two branches, 10,800-second cap). The
kernel paired result was a confirmed **26.3287%** improvement with a 95% CI of
**[22.1801%, 30.6322%]** over 100 valid pairs. The outer `latency_us` workload
result was **-0.0097%**, 95% CI **[-0.0390%, 0.0365%]**, also over 100 valid
pairs and below the required 2% effect. The authoritative verdict was therefore
`kernel_only_win`; the global best remained the baseline. The run completed in
2,232.43 seconds, resumed idempotently at `complete`, and its host NCU 2026.1.1
profile read 140 metrics without degradation.

The workload adapter compares the user's prebuilt baseline and optimized
binaries. The captured dispatch headers were byte-identical, so this is valid
binary A/B evidence but not source-level promotion proof. The source trees were
not modified. Durable evidence is under
`/data/tcheng/cuda-skill-e2e/v2.2/artifacts/{current,compatibility,real}`; the
real run is `real/orchestrator/run_20260717_043610_569950525`.

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
