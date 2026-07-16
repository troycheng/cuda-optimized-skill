# cuda-kernel-optimizer

**English** | [简体中文](README.zh-CN.md)

A Codex-compatible skill that iteratively optimizes a CUDA / CUTLASS / Triton kernel against a Python reference. It combines correctness checks, robust timing distributions, optional `nsight-compute` (`ncu`) profiling, branch selection, ablation, and SASS verification.

This is a **skill package**, not a standalone tool. An agent reads `SKILL.md` and drives the loop. The scripts under `scripts/` handle the deterministic parts (environment detection, profiling, benchmarking, and state).

---

![alt text](asset/v2_en_arch.png)

## Usage
```text
Use this prompt in the agent:
@cuda-kernel-optimizer use this skill to optimize "the operator you want to optimize" for N iterations.
```

## What's new in V2

V2 upgrades the loop from "try-and-log" into "try–attribute–verify–learn". Four mechanisms are added on top of V1; everything below reflects V2 behavior:

- **Roofline-driven axis budget** — instead of V1's fixed 1-method-per-axis, V2 computes per-iteration compute/memory/latency gaps (Δc, Δm, Δl) and splits the 3-method budget proportionally (per-axis cap = 2). When all three gaps fall below 0.15 the loop early-stops with `near_peak: true`.
- **Branch-and-Select exploration** — each iteration generates K branch candidates (default K=4) sharing the same methods but varying tile size, pipeline stages, warp count, and implementation variants. The fastest correct branch wins as champion; the rest are archived in `frontier`.
- **Ablation-based attribution** — after the champion is picked, each method is ablated one at a time. `attribution(m) = ms_without_m − ms_champion` gives a per-method causal contribution instead of a single packed verdict.
- **SASS instruction-level verification** — `cuobjdump --dump-sass` is grepped against a signature table (`sass_signatures.json`) to confirm each claimed optimization actually appears in the compiled machine code.
- **Noise-aware measurements** — benchmark JSON preserves independent samples, median, nearest-rank p95, population standard deviation, and median-normalized CV. Branches are ranked by median and annotated when their difference is inside the configured noise band.

These together change method classification from two buckets (effective / ineffective) to three: `effective_methods` (SASS ✓ and attribution > noise), `ineffective_methods` (SASS ✓ but attribution ≤ noise), and `implementation_failed_methods` (SASS ✗).

## RTX 5090 validation

The opt-in SM120 matrix passed on a physical RTX 5090 on 2026-07-16 for
Triton, native CUDA, and CUTLASS correctness plus timing artifacts.

| Lane | CUDA compiler | Triton | CUTLASS | Nsight Compute | Result |
|---|---:|---:|---:|---:|---|
| Compatibility | 13.0.1 | 3.6.0 | 4.6.1 | 2025.3.1 | 3/3 backends passed |
| Current | 13.3.73 | 3.7.1 | 4.6.1 | 2026.2.1 | 3/3 backends passed |

The host blocked hardware-counter access with `ERR_NVGPUCTRPERM` in both
lanes. The skill correctly records `can_read_counters: false` and retains the
failure log; no privileged capability or driver policy change was used. See
[`tests/gpu/sm120/README.md`](tests/gpu/sm120/README.md) for the opt-in command.

An additional isolated vLLM SM120 blockwise-FP8 `down_proj`
(`m=1,n=8704,k=5120`) binary A/B used five fresh processes per candidate and
200 timed launches per process. Both candidates passed correctness; medians
were 20.482 us and 20.483 us, so the run stopped inside the 2% noise band. The
captured source headers were byte-identical despite distinct extension hashes,
so this is intentionally reported as binary evidence rather than a new
source-patch validation.

## What you need

On the host where the agent runs:

- A CUDA GPU with working drivers (`nvidia-smi` works)
- `nvcc` in `$PATH` (for CUDA / CUTLASS backends)
- `ncu` in `$PATH` if profiler metrics are required. Without counter access, the skill records the concrete failure and continues with correctness, timing, source, and SASS evidence.
- `cuobjdump` in `$PATH` (ships with the CUDA toolkit) — needed for V2's SASS verification step
- Python 3.10+ with `torch` (CUDA build), `triton` if you want the Triton backend
- For CUTLASS kernels: `$CUTLASS_PATH` or `$CUTLASS_INCLUDE_DIR` pointing at a tree with both `cutlass/` and `cute/` headers

`benchmark.py` (the generic operator benchmark driver) is bundled at `scripts/benchmark.py` — no separate installation needed.

### `ncu` permission gotcha

On many cloud and container setups, profiling-counter access is disabled. You'll see it as `can_read_counters: false` in `env.json`. Do not change host policy or add container capabilities automatically. With explicit operator authorization, possible remedies include:

- Run the host as root, or
- Add `options nvidia NVreg_RestrictProfilingToAdminUsers=0` to `/etc/modprobe.d/nvidia.conf` and reboot, or
- For docker: `--cap-add=SYS_ADMIN` (Nsight docs recommend this)

## What you provide

1. **Baseline kernel file** — `gemm.cu` (CUDA/CUTLASS) or `gemm.py` (Triton)
2. **Reference file** — `ref.py` exposing `reference(**kwargs)` and optional `atol` / `rtol`
3. **Dims** — the scalar args the signature takes (e.g. `M=4096 N=4096 K=4096`)
4. **Path to `benchmark.py`** — already bundled under `scripts/benchmark.py`; `orchestrate.py` defaults to it. Pass `--benchmark <path>` only if you have a custom version.
5. Optional: iteration count `N` (default 3), `ncu_num` per-axis top-K (default 5), noise threshold (default 2%), **branches per iteration `K` (default 4, via `--branches`)**

## What you get back

A sibling directory of your baseline, `run_YYYYMMDD_HHMMSS/`, containing:

```text
run_YYYYMMDD_HHMMSS/
├── state.json                   # global state, re-readable across sessions
│                                #   V2 adds: branches, implementation_failed_methods,
│                                #            roofline_history, frontier
├── env.json                     # GPU / nvcc / ncu / CUTLASS snapshot
├── baseline/
│   ├── <baseline>               # copied verbatim
│   └── bench.json               # seed timing + correctness
├── iterv1/
│   ├── roofline.json            # Δc / Δm / Δl + per-axis budget allocation
│   ├── methods.json             # methods picked under the budget (trigger_strength included)
│   ├── analysis.md              # evidence, decisions, validation, and risks
│   ├── best_input.ncu-rep       # present when target profiling succeeds
│   ├── branches/                # K branch candidates (same methods, different hyperparams)
│   │   ├── b0/kernel.{cu,py} + bench.json
│   │   ├── b1/…
│   │   └── …
│   ├── kernel.{cu,py}           # champion kernel (fastest correct branch)
│   ├── kernel.ncu-rep           # present when champion profiling succeeds
│   ├── ncu_top.json             # available top-K metrics per axis
│   ├── *.ncu.log                # preserved success or failure logs
│   ├── sass_check.json          # per-method SASS signature verification
│   ├── ablations/               # leave-one-out ablation runs
│   │   ├── no_<method_a>/kernel.{cu,py} + bench.json
│   │   └── …
│   ├── attribution.json         # per-method causal contribution (ms)
│   └── bench.json
├── iterv2/ …
├── iterv3/ …
└── summary.md                   # headline speedup, timeline, bottleneck drift, retrospective
```

## Manual invocation

You do not need to drive the loop by hand, but these commands are useful when debugging the skill itself:

```bash
# 0 + 0b + 1 + 2 + 3a-for-iter1
python scripts/orchestrate.py setup \
  --baseline   ./gemm.cu \
  --ref        ./ref.py \
  --iterations 3 \
  --ncu-num    5 \
  --branches   4 \
  --dims       '{"M":4096,"N":4096,"K":4096}'
  # --benchmark defaults to scripts/benchmark.py (bundled)

# --- (The agent writes iterv1/kernel.cu + iterv1/methods.json + iterv1/analysis.md
#      + K branch candidates under iterv1/branches/) ---

# 3d + 3f + 3a-for-iter2 for iter 1
# close-iter now also runs: branch selection → SASS check → ablation → state update
python scripts/orchestrate.py close-iter \
  --run-dir   run_20260418_143022 \
  --iter      1
  # --benchmark defaults to scripts/benchmark.py (bundled)

# (repeat code-gen + close-iter for iter 2 and iter 3)

# 4
python scripts/orchestrate.py finalize --run-dir run_20260418_143022
```

Each script is independently invocable (`--help` on any of them); `orchestrate.py` is just a convenience wrapper.

## Repo layout

```text
cuda-kernel-optimizer/
├── SKILL.md                         # skill entry point
├── README.md                        # you are here
├── scripts/
│   ├── benchmark.py                 # bundled benchmark driver (from project)
│   ├── check_env.py                 # detect GPU / nvcc / ncu / cuobjdump / CUTLASS / libs
│   ├── preflight.py                 # validate baseline + ref contract
│   ├── state.py                     # the ONLY writer of state.json
│   ├── validate_methods.py          # priority-compliance gate (called by state.py)
│   ├── run_iteration.py             # calls benchmark.py, captures results
│   ├── profile_ncu.py               # runs ncu, extracts top-K per axis
│   ├── roofline.py                  # [V2] compute Δc/Δm/Δl, allocate axis budget, near_peak check
│   ├── branch_explore.py            # [V2] compile + benchmark K branches, elect champion, update frontier
│   ├── ablate.py                    # [V2] leave-one-out ablation, emit per-method attribution
│   ├── sass_check.py                # [V2] cuobjdump → grep signatures → per-method SASS verdict
│   ├── summarize.py                 # renders summary.md (V2: includes bottleneck drift table)
│   └── orchestrate.py               # end-to-end CLI (setup/close-iter/finalize)
├── references/
│   ├── ncu_metrics_guide.md         # bottleneck → optimization mapping
│   ├── optimization_catalog.md      # priority-ordered catalog
│   ├── method_registry.json         # machine-readable mirror (validator reads)
│   └── sass_signatures.json         # [V2] method → expected SASS instruction signatures
├── templates/
│   ├── iteration_report.md          # analysis.md skeleton
│   └── methods.schema.json          # schema for methods.json (V2: adds trigger_strength)
└── examples/
    └── walkthrough.md               # annotated example session
```

## How an agent uses this

When a user says "optimize `gemm.cu`", the agent:

1. reads `SKILL.md`
2. calls `orchestrate.py setup` (which runs env check → preflight → init → seed baseline → first profile)
3. reads `iterv1/ncu_top.json` and the current best kernel source
4. **runs `roofline.py` to get Δc / Δm / Δl and the per-axis method budget (total = 3, per-axis cap = 2); if `near_peak: true`, the loop ends here**
5. consults `references/optimization_catalog.md` + `references/ncu_metrics_guide.md`
6. picks methods **under the axis budget** (budget-aware scan: skip axis if budget=0, pick top-N by `trigger_strength` if budget=2), writes them + reasoning to `iterv1/methods.json` and `iterv1/analysis.md`
7. writes **K branch candidates** to `iterv1/branches/b{0..K-1}/kernel.<ext>` — same methods, different hyperparameters (tile / stages / warps / impl variants)
8. calls `orchestrate.py close-iter --iter 1`, which internally:
   - runs `branch_explore.py` → compiles + benchmarks all branches, elects the fastest correct one as champion (copied to `iterv1/kernel.<ext>`), archives the rest in `frontier`
   - profiles the champion with `ncu` → `iterv1/kernel.ncu-rep`
   - runs `sass_check.py` → `iterv1/sass_check.json`
   - runs `ablate.py` → `iterv1/attribution.json`
   - updates state: each method lands in one of `effective_methods` / `ineffective_methods` / `implementation_failed_methods` based on SASS ✓/✗ × attribution > noise
9. on correctness failure (all K branches fail): inspects `bench.json.correctness` + `bench.stderr.txt`, rewrites the kernel, retries (up to 3×)
10. on success: `best_file` advances if faster; `roofline_history` is appended
11. loops back to step 3 for the next iteration
12. calls `orchestrate.py finalize` and writes a retrospective into `summary.md` — including the bottleneck drift table sourced from `roofline_history`

See `examples/walkthrough.md` for a full example and `SKILL.md` for the formal procedure.

## Limits and honest caveats

- **Ceiling**: if your reference is already cuBLAS / cuDNN / cuBLASLt, meaningful wins require algorithmic changes (split-K, stream-K, fused epilogues, mixed precision) that may not fit a 3-iteration budget. Large speedups are easier when the baseline is hand-rolled.
- **Noise**: kernels running under ~50 μs are dominated by launch overhead. The skill's default 2% noise threshold helps, but if your dims are tiny, raise `--repeat` or the dimensions. Ablation attribution uses the same threshold — sub-noise contributions are classified as `ineffective_methods`.
- **Triton + `@triton.autotune`**: autotuning under `ncu` is slow and can time out. Either pre-bake a single config before profiling, or set `--launch-count 1` and increase warmup.
- **ncu CSV column names**: older `ncu` (< 2022.1) emits `"Metric Value"` with different capitalization/units; `profile_ncu.py` is tolerant but if you see all zeros check the `.ncu.log` file in the iteration directory.
- **Branch cost**: with K=4 and ablation, each iteration compiles up to K + (num_methods) kernels. On a fresh build this can be slow; lower `--branches` if wall-clock matters more than exploration.
- **SASS signatures are heuristic**: `sass_signatures.json` greps for instruction patterns, not full semantic equivalence. A method can pass the grep but still be implemented suboptimally — attribution is what catches that.
- **Retries are bounded**: after 3 correctness failures on one iteration, the skill moves on and records the attempt as failed rather than looping forever. A kernel that can't be made correct after 3 tries usually has a conceptual issue that needs human review.

## Example result

Using the Batch Normalization problem from Tensara as an example, this project demonstrates a substantial performance improvement from a baseline implementation to an optimized kernel. After submission to the A100-80GB environment, the solution passed 4/4 test cases successfully. The average runtime dropped from 82.94 ms to 439.13 μs, while throughput increased dramatically from 2.52 GFLOPS to 476.20 GFLOPS. It is worth noting that most development and tuning were carried out locally on an RTX 3060, so local measurements cannot fully reflect the upper-bound performance achievable on an A100. Therefore, the final benchmark results should be based on the platform’s A100 evaluation, which better highlights the impact of careful kernel optimization and implementation details.

![alt text](asset/Tensara_baseline.png)

![alt text](asset/Tensara_best.png)


## License / attribution

This skill is independent of and does not redistribute CUTLASS, Triton, or Nsight Compute. You need to install those separately.

## Star History

<a href="https://www.star-history.com/?repos=KernelFlow-ops%2Fcuda-optimized-skill&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=KernelFlow-ops/cuda-optimized-skill&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=KernelFlow-ops/cuda-optimized-skill&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=KernelFlow-ops/cuda-optimized-skill&type=date&legend=top-left" />
 </picture>
</a>
