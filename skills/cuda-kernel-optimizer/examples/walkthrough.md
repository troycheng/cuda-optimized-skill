# Example Walkthrough (v2 — Roofline-Driven, Branch-and-Select)

A hypothetical session optimizing `gemm.cu` against `ref.py` for 3 iterations on an H100, with 4 branches per iteration.

## Layout before the run

```
~/work/
├── gemm.cu          ← baseline (has `extern "C" void solve(float*, float*, float*, int, int, int)`)
└── ref.py           ← defines `reference(A, B, C, M, N, K)` + `atol = 1e-3`
```

## Command chain (agent-driven)

### Step 0–2 — Setup

```bash
python cuda-kernel-optimizer/scripts/orchestrate.py setup \
  --baseline ~/work/gemm.cu \
  --ref      ~/work/ref.py \
  --iterations 3 \
  --ncu-num 5 \
  --branches 4 \
  --dims '{"M":4096,"N":4096,"K":4096}'
```

Output (abridged):
```json
{
  "run_dir": "/home/user/work/run_20260420_143022",
  "state":   "/home/user/work/run_20260420_143022/state.json",
  "env":     "/home/user/work/env.json",
  "early_stop": false,
  "next_step": "The agent should now read iterv1/roofline.json ..."
}
```

This automatically:
1. Probes the environment (`check_env.py`)
2. Validates baseline + ref contract (`preflight.py`)
3. Initializes `state.json` with branches=4
4. Seeds baseline timing
5. Profiles baseline with ncu (`--set full`) → `iterv1/best_input.ncu-rep`
6. Computes roofline gaps → `iterv1/roofline.json`

### Step 3b — Agent reads roofline (iter 1)

The agent inspects `iterv1/roofline.json`:
```json
{
  "delta_compute": 0.92,
  "delta_memory": 0.57,
  "delta_latency": 0.61,
  "bound": "compute",
  "near_peak": false,
  "axis_budget": {"compute": 2, "memory": 0, "latency": 1}
}
```

Interpretation: HMMA utilization at 8% → massive compute gap (Δ_c=0.92). Long scoreboard stalls at 61% → latency gap. Memory bandwidth at 43% → moderate but not dominant. Budget allocates **2 compute + 0 memory + 1 latency**.

### Step 3c — Agent picks methods (iter 1)

| Axis | Budget | Method id | Priority |
|------|--------|-----------|----------|
| compute | 2 | `compute.tensor_core` | P1 |
| compute | — | `compute.overlap_compute_memory` | P2 |
| latency | 1 | `latency.async_pipeline` | P3 |

Note: memory budget = 0, so no memory methods this round (Δ_m = 0.57 > 0.10 but gets rounded out by proportional allocation dominated by compute+latency).

### Step 3d — Agent writes 4 branch kernels

All branches apply the same 3 methods, but with different hyperparameters:

| Branch | Tile (M×N×K) | Stages | Warps | Notes |
|--------|-------------|--------|-------|-------|
| b1 | 128×128×32 | 3 | 4 | Conservative baseline |
| b2 | 128×256×32 | 3 | 8 | Wider N tile |
| b3 | 256×128×32 | 4 | 4 | Wider M tile + deeper pipeline |
| b4 | 128×128×64 | 5 | 4 | Deeper K tile + max stages |

The agent writes:
- `iterv1/branches/b1/kernel.cu` through `b4/kernel.cu`
- `iterv1/methods.json`
- `iterv1/analysis.md`

### Step 3e–3j — Close iteration

```bash
python cuda-kernel-optimizer/scripts/orchestrate.py close-iter \
  --run-dir ~/work/run_20260420_143022 \
  --iter 1
```

This automatically:
1. **Branch explore**: compiles + benchmarks all 4 branches
   - b1: 3.21 ms (PASS)
   - b2: 2.14 ms (PASS) ← champion
   - b3: 2.89 ms (PASS)
   - b4: FAIL (validation error — K tile too large caused register spill)
2. **Selects b2** as champion → copies to `iterv1/kernel.cu`
3. **NCU profiles champion** (`--set full`) → `iterv1/kernel.ncu-rep`
4. **Ablation** (if the agent provided ablated kernels under `iterv1/ablations/`):
   - Without `compute.tensor_core`: 4.82 ms → attribution = +2.68 ms ✓
   - Without `compute.overlap`: 2.31 ms → attribution = +0.17 ms ✓
   - Without `latency.async_pipeline`: 2.19 ms → attribution = +0.05 ms (below 2% noise threshold) ✗
5. **SASS check**: greps for HMMA → found ✓; greps for LDGSTS/CP.ASYNC → found ✓
6. **State update**:
   - `compute.tensor_core` → effective (attributed + SASS verified)
   - `compute.overlap_compute_memory` → effective (attributed + SASS verified)
   - `latency.async_pipeline` → ineffective (attribution below noise threshold)
   - b1, b3 saved to `state.frontier`
7. **Opens iter 2**: profiles new best → `iterv2/best_input.ncu-rep` → roofline

Output:
```json
{
  "iter": 1,
  "status": "closed",
  "best_ms": 2.14,
  "next_iter": 2,
  "early_stop": false
}
```

### Step 3b — Agent reads roofline (iter 2)

```json
{
  "delta_compute": 0.35,
  "delta_memory": 0.62,
  "delta_latency": 0.48,
  "bound": "bandwidth",
  "near_peak": false,
  "axis_budget": {"compute": 1, "memory": 1, "latency": 1}
}
```

The bound shifted from compute to bandwidth — tensor cores now running at 65% (from 8%), but memory is the new bottleneck. Budget is now 1:1:1 — balanced.

### Iter 2 & 3 — Same loop

Each iteration:
1. The agent reads the roofline budget → picks methods accordingly
2. The agent writes K=4 branch variants
3. `close-iter` runs the full pipeline (branch → ncu → ablate → sass → update)

By iter 3, if roofline shows all Δ < 0.15, `early_stop: true` and the loop terminates.

### Step 4 — Finalize

```bash
python cuda-kernel-optimizer/scripts/orchestrate.py finalize \
  --run-dir ~/work/run_20260420_143022
```

Produces `summary.md` with:
- Roofline history table (how Δ shifted across iterations)
- Per-iteration timeline with methods + speedup + status
- Effective methods (attribution-verified)
- Ineffective and implementation-failed method lists
- Frontier candidates (unexplored branches)
- The agent appends a retrospective paragraph

## Final layout

```
run_20260420_143022/
├── state.json
├── env.json
├── summary.md
├── baseline/
│   ├── gemm.cu
│   └── bench.json
├── iterv1/
│   ├── kernel.cu               (champion = b2)
│   ├── methods.json
│   ├── analysis.md
│   ├── roofline.json
│   ├── best_input.ncu-rep      (profile of baseline)
│   ├── ncu_top.json
│   ├── kernel.ncu-rep          (profile of champion — ALWAYS present)
│   ├── attribution.json
│   ├── sass_check.json
│   ├── bench.json
│   ├── branch_results.json
│   ├── branches/
│   │   ├── b1/kernel.cu + bench.json
│   │   ├── b2/kernel.cu + bench.json
│   │   ├── b3/kernel.cu + bench.json
│   │   └── b4/kernel.cu + bench.json
│   └── ablations/
│       ├── compute_tensor_core/kernel.cu + bench.json
│       ├── compute_overlap_compute_memory/kernel.cu + bench.json
│       └── latency_async_pipeline/kernel.cu + bench.json
├── iterv2/
│   └── ... (same pattern)
└── iterv3/
    └── ...
```

## Key differences from v1

| Aspect | v1 | v2 |
|--------|----|----|
| Axis budget | Fixed 1:1:1 | Roofline-proportional, cap=2 |
| Candidates per iter | 1 | K=4 (branch-and-select) |
| Method validation | Priority compliance only | + attribution + SASS check |
| Champion ncu report | Optional | Mandatory every iteration |
| Effective classification | Overall improved → all 3 effective | Per-method attribution required |
| Early stop | None | All Δ < 0.15 → near_peak |
| Frontier / rollback | None | Non-champion branches saved |
