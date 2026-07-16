# V2.2 dual-loop walkthrough

This example optimizes `gemm.cu` against `ref.py` and then validates the
shortlisted kernel on a user-provided inference workload. Values are illustrative;
the verdict rules and artifact names match V2.2.

## 1. User-owned inputs

```text
~/work/
├── gemm.cu
├── ref.py
└── workload.py
```

`ref.py` exposes `reference(**kwargs)`. The user copies
`templates/workload.py`, replaces every TODO, and declares the real objective:

```python
def metrics():
    return {
        "primary_metric": {"name": "p50_latency_ms", "direction": "lower"},
        "min_effect_pct": 1.0,
        "constraints": [
            {"name": "p99_latency_ms", "max_regression_pct": 0.5},
        ],
    }
```

The adapter, its declared local dependencies, objective, and cases belong to the
user. The skill does not discover or synthesize a workload.

Equivalent workload entry points are:

```bash
# Python adapter
--workload ~/work/workload.py

# Command adapter (objective is mandatory)
--workload-cmd 'python3 ~/work/run_service_bench.py --json' \
--objective ~/work/objective.json

# Strict manifest
--workload-manifest ~/work/workload.json
```

Choose only one form.

## 2. Freeze a balanced run

```bash
python3 cuda-kernel-optimizer/scripts/orchestrate.py setup \
  --baseline ~/work/gemm.cu \
  --ref ~/work/ref.py \
  --dims '{"M":4096,"N":4096,"K":4096}' \
  --backend cuda \
  --budget balanced \
  --workload ~/work/workload.py
```

`balanced` is the default and freezes a 10800-second limit, 8 branches, 4
rounds, 20-100 paired blocks, 2 outer candidates, at most 10 workload cases,
and targeted sanitizer coverage. Setup emits paths to `manifest.json`,
`state.json`, and `checkpoint.json`, plus `next_stage: candidate_correctness`.

At this point the baseline and reference have stable hashes, the baseline passed
correctness, and the workload snapshot is frozen. Changing those inputs requires
a new run.

## 3. Inner kernel loop

The agent reads the current best, available environment/profiler evidence, the
optimization catalog, and the iteration report template. Suppose it records two
methods and writes eight implementation variants:

```text
iterv1/
├── analysis.md
├── methods.json
└── branches/
    ├── b1/kernel.cu
    ├── b2/kernel.cu
    └── ... b8/kernel.cu
```

Then it runs:

```bash
python3 cuda-kernel-optimizer/scripts/orchestrate.py close-iter \
  --run-dir ~/work/run_20260717_101500 --iter 1
```

For every viable branch, the deterministic loop checks the Python reference,
runs the sanitizer policy, captures compiler/SASS provenance, and measures
randomized AB/BA baseline/candidate pairs. Telemetry-invalid blocks are retained
but excluded from the valid pair count.

Illustrative inner results:

| Candidate | Correct | Sanitizer | Estimate | 95% CI | Verdict |
|---|---|---|---:|---:|---|
| b2 | yes | passed | +1.7% | [+0.8%, +2.4%] | `confirmed_win` |
| b5 | yes | passed | +0.4% | [-0.3%, +1.2%] | `inconclusive` |
| b7 | no | not run | — | — | rejected |

Only b2 may enter the outer shortlist. b5 is preserved for audit but cannot be
promoted merely because its point estimate is positive.

If NCU target profiling returns `ERR_NVGPUCTRPERM`, the log and return code are
kept and profiler coverage is marked unavailable. The run does not request
additional capabilities or change driver policy; paired timing continues.

## 4. Outer real-workload loop

Because this is full mode, the b2 inner win is not enough to update `best_file`.
The orchestrator evaluates frozen baseline/candidate roles across the user cases
and writes raw observations under:

```text
iterv1/workload/<candidate-hash>/paired_samples.jsonl
```

Illustrative workload result:

- p50 latency estimate: +1.3%, 95% CI [+1.1%, +1.6%]; required effect: 1.0%;
- p99 regression: +0.2%; allowed regression: 0.5%;
- verdict: `confirmed_win`, all constraints pass.

`iterv1/decision.json` can therefore record `end_to_end_win`. State promotion
occurs only after this decision is durably written and bound to the candidate
hash. If the p50 interval had crossed 1.0%, or p99 had regressed more than 0.5%,
the current best would remain unchanged.

## 5. Kernel-only contrast

If setup omitted every workload option, the same confirmed b2 kernel evidence
could produce `kernel_only_win`. The summary would explicitly say that no
user-provided workload was supplied and would not claim an end-to-end result.

## 6. Resume without replay

After interruption:

```bash
python3 cuda-kernel-optimizer/scripts/orchestrate.py resume --run-dir \
  ~/work/run_20260717_101500
```

Resume verifies the manifest, state, checkpoint, candidate bytes, and frozen
workload, then reports `next_stage` and `next_iteration`. Completed stages are
not replayed. A drifted input or unsafe symlink fails closed.

## 7. Finalize and inspect evidence

```bash
python3 cuda-kernel-optimizer/scripts/orchestrate.py finalize --run-dir \
  ~/work/run_20260717_101500
```

`summary.md` starts with the terminal result and budget, then reports frozen
inputs, kernel evidence, real-workload evidence, profiler/sanitizer/compiler
coverage, candidates, raw paths, and resume status. Recompute statistics from
the linked `paired_samples.jsonl` files rather than trusting copied prose.
