# CUDA Kernel Optimizer v2.2 双环优化实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（- [ ]）语法来跟踪进度。

**目标：** 将 cuda-kernel-optimizer 升级为预算可控、paired A/B 统计可信、且能用用户提供的真实 workload 做端到端验收的双环优化 skill。

**架构：** 保留现有 evidence-guided branch generation、正确性、NCU、ablation 和 SASS 流程；新增彼此独立的 budget、paired statistics、artifact/checkpoint、workload adapter、outer-loop decision 模块。内环只筛选统计确认的 kernel 候选，外环是唯一可以产生 end_to_end_win 并推进全局 best 的验收面。

**技术栈：** Python 3.10+ 标准库、PyTorch CUDA events、CUDA/CUTLASS/Triton、Nsight Compute、Compute Sanitizer、unittest、RTX 5090 SM120、Git。

---

## 执行前提

- 从当前 main 创建隔离分支和 worktree：

~~~bash
git worktree add ../cuda-optimized-skill-v2-2 -b agent/v2-2-dual-loop main
cd ../cuda-optimized-skill-v2-2
~~~

- 所有实现和测试提交到 agent/v2-2-dual-loop。
- 只允许推送 troycheng/cuda-optimized-skill；不得向 KernelFlow-ops/cuda-optimized-skill 推送。
- 5090 远端只写 /data/tcheng/cuda-skill-e2e/，不得修改 /data/vllm-opt 或其 dirty worktree。
- 每个任务完成后先运行任务内验证，再 commit。

## 文件结构决策

### 新建

- skills/cuda-kernel-optimizer/scripts/budget.py
  预算 preset、override、deadline 和 admission control。
- skills/cuda-kernel-optimizer/scripts/paired_stats.py
  paired improvement、bootstrap CI 和统一 verdict。
- skills/cuda-kernel-optimizer/scripts/artifact_store.py
  hash、atomic JSON、append-only JSONL、manifest 和 checkpoint。
- skills/cuda-kernel-optimizer/scripts/telemetry.py
  GPU telemetry 采集与整块 paired block 污染判定。
- skills/cuda-kernel-optimizer/scripts/paired_benchmark.py
  同进程准备 baseline/candidate 并执行随机 AB/BA kernel 测量。
- skills/cuda-kernel-optimizer/scripts/workload_adapter.py
  Python adapter、command、manifest 三种用户 workload 的规范化接口。
- skills/cuda-kernel-optimizer/scripts/workload_evaluate.py
  用户真实 workload 的 paired 外环运行器。
- skills/cuda-kernel-optimizer/scripts/decision.py
  primary KPI、constraints、Pareto 和 terminal status 判定。
- skills/cuda-kernel-optimizer/scripts/sanitize.py
  targeted/full Compute Sanitizer 调度与证据保存。
- skills/cuda-kernel-optimizer/scripts/compiler_evidence.py
  source/IR/PTX/SASS/binary 的可用性和 hash manifest。
- skills/cuda-kernel-optimizer/references/sanitizer_policy.json
  优化方法到 sanitizer tools 的显式映射。
- skills/cuda-kernel-optimizer/templates/objective.schema.json
  workload objective 的机器可读 schema。
- skills/cuda-kernel-optimizer/templates/workload.py
  用户 workload adapter 示例模板。
- tests/test_budget.py
- tests/test_paired_stats.py
- tests/test_artifact_store.py
- tests/test_telemetry.py
- tests/test_paired_benchmark.py
- tests/test_workload_adapter.py
- tests/test_workload_evaluate.py
- tests/test_decision.py
- tests/test_orchestrate.py
- tests/test_sanitize.py
- tests/test_compiler_evidence.py
- tests/test_state_schema.py
- tests/test_summarize.py
- tests/gpu/sm120/fixtures/workload_smoke.py
- tests/gpu/sm120/fixtures/objective.json

### 修改

- skills/cuda-kernel-optimizer/scripts/benchmark.py
  暴露 prepare_solution、warm_solution 和 measure_once，不改变现有 CLI。
- skills/cuda-kernel-optimizer/scripts/preflight.py
  校验 workload 输入互斥性和 objective contract。
- skills/cuda-kernel-optimizer/scripts/state.py
  schema_version=2、候选状态、kernel/workload best、checkpoint 兼容检查。
- skills/cuda-kernel-optimizer/scripts/branch_explore.py
  使用 paired verdict 取代 median + 固定噪声阈值。
- skills/cuda-kernel-optimizer/scripts/run_iteration.py
  输出统一 statistic 字段，不再回传 average_ms 作为 promotion metric。
- skills/cuda-kernel-optimizer/scripts/orchestrate.py
  接入预算、workload、outer loop、checkpoint 和 resume。
- skills/cuda-kernel-optimizer/scripts/sass_check.py
  将 SASS 文件和 hash 注册到 compiler evidence。
- skills/cuda-kernel-optimizer/scripts/summarize.py
  分开 kernel 和 end-to-end 结论。
- skills/cuda-kernel-optimizer/SKILL.md
- skills/cuda-kernel-optimizer/agents/openai.yaml
- skills/cuda-kernel-optimizer/examples/walkthrough.md
- skills/cuda-kernel-optimizer/templates/iteration_report.md
- README.md
- README.zh-CN.md
- tests/test_benchmark.py
- tests/test_branch_explore.py
- tests/test_readme_sync.py
- tests/test_skill_metadata.py
- tests/gpu/sm120/test_sm120_acceptance.py
- tests/gpu/sm120/remote/run_lane.sh
- tests/gpu/sm120/README.md

## 任务 1：预算策略与 deadline admission

**文件：**

- 创建：skills/cuda-kernel-optimizer/scripts/budget.py
- 创建：tests/test_budget.py

- [ ] **步骤 1：编写 preset、override 和硬 deadline 的失败测试**

~~~python
class BudgetTests(unittest.TestCase):
    def test_balanced_is_three_hour_default(self):
        policy = budget.resolve_budget("balanced")
        self.assertEqual(policy.max_seconds, 3 * 60 * 60)
        self.assertEqual(policy.branches, 8)
        self.assertEqual(policy.max_rounds, 4)
        self.assertEqual(policy.min_pairs, 20)
        self.assertEqual(policy.max_pairs, 100)
        self.assertEqual(policy.outer_candidates, 2)
        self.assertEqual(policy.reserve_seconds, 300)

    def test_stage_is_rejected_when_it_cannot_finish_before_reserve(self):
        policy = budget.resolve_budget("quick")
        clock = budget.BudgetClock(policy, started_at=100.0)
        self.assertFalse(
            clock.can_start(now=100.0 + 2400.0, estimated_seconds=10.0)
        )

    def test_custom_requires_positive_wall_time(self):
        with self.assertRaisesRegex(ValueError, "max_seconds"):
            budget.resolve_budget("custom", max_seconds=0)
~~~

- [ ] **步骤 2：运行测试确认失败**

运行：

~~~bash
python -m unittest tests.test_budget -v
~~~

预期：FAIL，原因是 budget.py 不存在。

- [ ] **步骤 3：实现不可变 BudgetPolicy 和 BudgetClock**

~~~python
@dataclass(frozen=True)
class BudgetPolicy:
    name: str
    max_seconds: int
    branches: int
    max_rounds: int
    min_pairs: int
    max_pairs: int
    outer_candidates: int
    max_cases: int | None
    sanitizer_mode: str
    reserve_seconds: int = 300


PRESETS = {
    "quick": BudgetPolicy("quick", 2700, 4, 2, 20, 50, 1, 3, "targeted"),
    "balanced": BudgetPolicy("balanced", 10800, 8, 4, 20, 100, 2, 10, "targeted"),
    "thorough": BudgetPolicy("thorough", 36000, 16, 8, 30, 200, 3, None, "full"),
}


class BudgetClock:
    def can_start(self, *, now: float, estimated_seconds: float) -> bool:
        execution_deadline = self.started_at + self.policy.max_seconds - self.policy.reserve_seconds
        return now + max(0.0, estimated_seconds) <= execution_deadline

    def remaining_seconds(self, *, now: float) -> float:
        return max(0.0, self.started_at + self.policy.max_seconds - now)
~~~

resolve_budget 必须复制 preset 后应用显式 override，不能修改 PRESETS；custom 必须要求 max_seconds、branches、max_rounds、min_pairs、max_pairs 和 outer_candidates 都为正。

- [ ] **步骤 4：运行预算测试**

运行：

~~~bash
python -m unittest tests.test_budget -v
~~~

预期：全部 PASS。

- [ ] **步骤 5：提交**

~~~bash
git add skills/cuda-kernel-optimizer/scripts/budget.py tests/test_budget.py
git commit -m "feat: add budget presets and deadline admission"
~~~

## 任务 2：统一 paired statistics 判定

**文件：**

- 创建：skills/cuda-kernel-optimizer/scripts/paired_stats.py
- 创建：tests/test_paired_stats.py

- [ ] **步骤 1：编写 lower/higher、win/loss/inconclusive 和污染样本测试**

~~~python
class PairedStatsTests(unittest.TestCase):
    def test_lower_is_better_confirmed_win(self):
        pairs = [
            {"baseline": 100.0, "candidate": 98.0, "valid": True}
            for _ in range(24)
        ]
        result = paired_stats.classify_pairs(
            pairs, direction="lower", min_effect_pct=0.5,
            confidence=0.95, bootstrap_samples=1000, seed=7,
        )
        self.assertEqual(result["status"], "confirmed_win")
        self.assertGreaterEqual(result["ci_low_pct"], 0.5)
        self.assertEqual(result["statistic"], "median_paired_improvement_pct")

    def test_higher_is_better_confirmed_loss(self):
        pairs = [
            {"baseline": 100.0, "candidate": 98.0, "valid": True}
            for _ in range(24)
        ]
        result = paired_stats.classify_pairs(
            pairs, direction="higher", min_effect_pct=0.5,
            bootstrap_samples=1000, seed=7,
        )
        self.assertEqual(result["status"], "confirmed_loss")

    def test_mixed_noise_is_inconclusive(self):
        pairs = [
            {"baseline": 100.0, "candidate": value, "valid": True}
            for value in (99.0, 101.0) * 12
        ]
        result = paired_stats.classify_pairs(
            pairs, direction="lower", min_effect_pct=0.5,
            bootstrap_samples=1000, seed=7,
        )
        self.assertEqual(result["status"], "inconclusive")

    def test_invalid_blocks_are_excluded_but_preserved_in_counts(self):
        pairs = [
            {"baseline": 100.0, "candidate": 98.0, "valid": True},
            {"baseline": 100.0, "candidate": 50.0, "valid": False},
        ] * 20
        result = paired_stats.classify_pairs(
            pairs, direction="lower", min_effect_pct=0.5,
            bootstrap_samples=1000, seed=7,
        )
        self.assertEqual(result["valid_pairs"], 20)
        self.assertEqual(result["invalid_pairs"], 20)
~~~

- [ ] **步骤 2：运行测试确认失败**

运行：

~~~bash
python -m unittest tests.test_paired_stats -v
~~~

预期：FAIL，原因是 paired_stats.py 不存在。

- [ ] **步骤 3：实现方向归一化、median bootstrap 和 verdict**

~~~python
def improvement_pct(baseline: float, candidate: float, direction: str) -> float:
    if baseline == 0:
        raise ValueError("baseline must be non-zero")
    if direction == "lower":
        return (baseline - candidate) / abs(baseline) * 100.0
    if direction == "higher":
        return (candidate - baseline) / abs(baseline) * 100.0
    raise ValueError("direction must be lower or higher")


def classify_pairs(
    pairs, *, direction, min_effect_pct, confidence=0.95,
    bootstrap_samples=10000, seed=0,
):
    valid = [p for p in pairs if p.get("valid", True)]
    values = [
        improvement_pct(float(p["baseline"]), float(p["candidate"]), direction)
        for p in valid
    ]
    if not values:
        return _result("inconclusive", [], pairs, None, None, None)
    estimate = statistics.median(values)
    low, high = bootstrap_median_ci(
        values, confidence=confidence, samples=bootstrap_samples, seed=seed
    )
    if low >= min_effect_pct:
        status = "confirmed_win"
    elif high <= -min_effect_pct:
        status = "confirmed_loss"
    else:
        status = "inconclusive"
    return _result(status, values, pairs, estimate, low, high)
~~~

bootstrap_median_ci 使用 random.Random(seed)，采用线性 percentile；结果包含 raw improvement、valid/invalid counts、confidence、min_effect_pct 和 statistic 名称。

- [ ] **步骤 4：运行统计测试**

运行：

~~~bash
python -m unittest tests.test_paired_stats -v
~~~

预期：全部 PASS。

- [ ] **步骤 5：提交**

~~~bash
git add skills/cuda-kernel-optimizer/scripts/paired_stats.py tests/test_paired_stats.py
git commit -m "feat: add paired bootstrap decision engine"
~~~

## 任务 3：版本化 artifact、manifest 和 checkpoint

**文件：**

- 创建：skills/cuda-kernel-optimizer/scripts/artifact_store.py
- 创建：tests/test_artifact_store.py
- 创建：tests/test_state_schema.py
- 修改：skills/cuda-kernel-optimizer/scripts/state.py

- [ ] **步骤 1：编写 schema、hash、append-only 样本和旧 state 拒绝测试**

~~~python
class ArtifactStoreTests(unittest.TestCase):
    def test_manifest_contains_schema_and_input_hashes(self):
        store = artifact_store.ArtifactStore(self.root)
        manifest = store.initialize(
            inputs={"baseline": str(self.baseline), "ref": str(self.reference)},
            budget={"name": "balanced", "max_seconds": 10800},
        )
        self.assertEqual(manifest["schema_version"], 2)
        self.assertRegex(manifest["inputs"]["baseline"]["sha256"], r"^[0-9a-f]{64}$")

    def test_pairs_are_append_only_jsonl(self):
        store = artifact_store.ArtifactStore(self.root)
        store.append_jsonl("candidates/c1/paired_samples.jsonl", {"pair": 1})
        store.append_jsonl("candidates/c1/paired_samples.jsonl", {"pair": 2})
        rows = store.read_jsonl("candidates/c1/paired_samples.jsonl")
        self.assertEqual([row["pair"] for row in rows], [1, 2])

    def test_checkpoint_rejects_changed_frozen_input(self):
        store = artifact_store.ArtifactStore(self.root)
        store.write_checkpoint({"input_hash": "aaa", "stage": "paired"})
        with self.assertRaisesRegex(ValueError, "frozen input"):
            store.load_checkpoint(expected_input_hash="bbb")


class StateSchemaTests(unittest.TestCase):
    def test_legacy_state_is_rejected_with_actionable_message(self):
        with self.assertRaisesRegex(ValueError, "schema_version.*new run"):
            state.validate_state({"run_dir": "/tmp/legacy"})
~~~

- [ ] **步骤 2：运行测试确认失败**

运行：

~~~bash
python -m unittest tests.test_artifact_store tests.test_state_schema -v
~~~

预期：FAIL，缺少 ArtifactStore 和 state.validate_state。

- [ ] **步骤 3：实现 atomic JSON、JSONL、hash 和 v2 state**

ArtifactStore 公开以下固定接口：

~~~python
CURRENT_SCHEMA_VERSION = 2

class ArtifactStore:
    def initialize(self, *, inputs: dict, budget: dict, environment: dict | None = None) -> dict: ...
    def candidate_dir(self, candidate_id: str) -> Path: ...
    def write_json(self, relative_path: str, payload: dict) -> Path: ...
    def append_jsonl(self, relative_path: str, payload: dict) -> Path: ...
    def read_jsonl(self, relative_path: str) -> list[dict]: ...
    def write_checkpoint(self, payload: dict) -> Path: ...
    def load_checkpoint(self, *, expected_input_hash: str) -> dict: ...
~~~

state.py 新 state 至少包含：

~~~python
{
    "schema_version": 2,
    "run_dir": run_dir,
    "input_hash": manifest["input_hash"],
    "budget": budget,
    "workload": workload_or_none,
    "best_file": baseline_copy,
    "best_kernel_statistics": None,
    "best_workload_statistics": None,
    "candidates": {},
    "frontier": [],
    "history": [],
}
~~~

validate_state 对缺少 schema_version 或非 2 的 state 给出“start a new
v2.2 run; legacy resume is not supported”错误；state 的读取入口必须调用
validate_state。新 run 的 checkpoint 必须原子写入。

- [ ] **步骤 4：运行 artifact/state 测试和旧 state 测试**

运行：

~~~bash
python -m unittest tests.test_artifact_store tests.test_state_schema -v
python -m unittest tests.test_branch_explore tests.test_profile_ncu -v
~~~

预期：全部 PASS；现有 NCU state 写入测试不回归。

- [ ] **步骤 5：提交**

~~~bash
git add skills/cuda-kernel-optimizer/scripts/artifact_store.py \
  skills/cuda-kernel-optimizer/scripts/state.py \
  tests/test_artifact_store.py tests/test_state_schema.py
git commit -m "feat: add versioned run artifacts and checkpoints"
~~~

## 任务 4：telemetry gate 与同进程 paired kernel runner

**文件：**

- 创建：skills/cuda-kernel-optimizer/scripts/telemetry.py
- 创建：skills/cuda-kernel-optimizer/scripts/paired_benchmark.py
- 创建：tests/test_telemetry.py
- 创建：tests/test_paired_benchmark.py
- 修改：skills/cuda-kernel-optimizer/scripts/benchmark.py
- 修改：tests/test_benchmark.py

- [ ] **步骤 1：编写整块污染、AB/BA 顺序和 input reset 测试**

~~~python
class TelemetryTests(unittest.TestCase):
    def test_temperature_or_clock_drift_invalidates_whole_block(self):
        verdict = telemetry.validate_block(
            before={"temperature_c": 60, "sm_clock_mhz": 2500},
            after={"temperature_c": 67, "sm_clock_mhz": 2250},
            max_temperature_delta_c=5,
            max_clock_delta_pct=5,
        )
        self.assertFalse(verdict["valid"])
        self.assertIn("temperature_delta", verdict["reasons"])
        self.assertIn("clock_delta", verdict["reasons"])


class PairedBenchmarkTests(unittest.TestCase):
    def test_orders_are_randomized_and_both_sides_are_reset(self):
        runner = paired_benchmark.PairedKernelRunner(
            baseline_state=self.baseline,
            candidate_state=self.candidate,
            seed=3,
            telemetry_reader=lambda: {"temperature_c": 60, "sm_clock_mhz": 2500},
        )
        pairs = runner.run(blocks=4)
        self.assertEqual({pair["order"] for pair in pairs}, {"AB", "BA"})
        self.assertEqual(self.baseline.reset_count, 4)
        self.assertEqual(self.candidate.reset_count, 4)
~~~

- [ ] **步骤 2：运行测试确认失败**

运行：

~~~bash
python -m unittest tests.test_telemetry tests.test_paired_benchmark -v
~~~

预期：FAIL，两个模块不存在。

- [ ] **步骤 3：从 benchmark.py 暴露稳定的小接口**

~~~python
def prepare_solution(
    solution_file, *, backend, dims, ptr_size, arch, nvcc_bin, seed
):
    return _setup_backend(
        solution_file, backend, dims, ptr_size, arch, nvcc_bin, seed=seed
    )


def warm_solution(state, warmup: int) -> None:
    for _ in range(warmup):
        _reset_tensor_inputs(state)
        state["callable"]()
    torch.cuda.synchronize()


def measure_once(state, *, cuda=None) -> float:
    _reset_tensor_inputs(state)
    return _time_iterations(
        state["callable"], warmup=0, repeat=1, cuda=cuda
    )[0]
~~~

保留 benchmark.py 原 CLI 行为和 JSON schema；新增测试证明现有 --help 与 samples_ms 不变。

- [ ] **步骤 4：实现 telemetry reader 和 paired runner**

telemetry.py 使用一次 nvidia-smi CSV 查询读取 temperature.gpu、clocks.sm、power.draw、memory.used 和 utilization.gpu。命令不可用时返回 available=false，此时只记录缺失，不伪造污染。

paired runner 的每个 block：

~~~python
order = rng.choice(("AB", "BA"))
before = telemetry_reader()
if order == "AB":
    baseline_ms = measure_once(baseline_state)
    candidate_ms = measure_once(candidate_state)
else:
    candidate_ms = measure_once(candidate_state)
    baseline_ms = measure_once(baseline_state)
after = telemetry_reader()
gate = validate_block(before=before, after=after)
pair = {
    "order": order,
    "baseline": baseline_ms,
    "candidate": candidate_ms,
    "valid": gate["valid"],
    "invalid_reasons": gate["reasons"],
    "telemetry": {"before": before, "after": after},
}
~~~

- [ ] **步骤 5：运行新增和现有 benchmark 测试**

运行：

~~~bash
python -m unittest tests.test_telemetry tests.test_paired_benchmark tests.test_benchmark -v
~~~

预期：全部 PASS。

- [ ] **步骤 6：提交**

~~~bash
git add skills/cuda-kernel-optimizer/scripts/benchmark.py \
  skills/cuda-kernel-optimizer/scripts/telemetry.py \
  skills/cuda-kernel-optimizer/scripts/paired_benchmark.py \
  tests/test_benchmark.py tests/test_telemetry.py tests/test_paired_benchmark.py
git commit -m "feat: add telemetry-gated paired kernel timing"
~~~

## 任务 5：统一 branch ranking 与 state promotion statistic

**文件：**

- 修改：skills/cuda-kernel-optimizer/scripts/branch_explore.py
- 修改：skills/cuda-kernel-optimizer/scripts/run_iteration.py
- 修改：skills/cuda-kernel-optimizer/scripts/state.py
- 修改：tests/test_branch_explore.py
- 修改：tests/test_state_schema.py

- [ ] **步骤 1：把旧 median/noise 测试改成 paired verdict 测试**

~~~python
def test_only_confirmed_winner_is_shortlisted(self):
    with mock.patch.object(
        branch_explore,
        "_paired_candidate",
        side_effect=[
            {"status": "inconclusive", "estimate_pct": 0.3, "ci_low_pct": -0.2},
            {"status": "confirmed_win", "estimate_pct": 2.0, "ci_low_pct": 1.4},
        ],
    ):
        output = branch_explore.run(str(self._state(Path(tmp))), iteration=1)
    self.assertEqual(output["status"], "shortlist_ready")
    self.assertEqual(output["shortlist"][0]["branch_index"], 2)


def test_no_confirmed_winner_keeps_current_best(self):
    with mock.patch.object(
        branch_explore,
        "_paired_candidate",
        return_value={"status": "inconclusive", "estimate_pct": 0.2},
    ):
        output = branch_explore.run(str(self._state(Path(tmp))), iteration=1)
    self.assertEqual(output["status"], "no_confirmed_kernel_win")
    self.assertIsNone(output["champion"])
~~~

增加 state 测试：full 模式下 inner win 不能更新 best_file；kernel-only 模式下 confirmed kernel win 可以更新 best_file，但 status 必须是 kernel_only_win。

- [ ] **步骤 2：运行测试确认失败**

运行：

~~~bash
python -m unittest tests.test_branch_explore tests.test_state_schema -v
~~~

预期：FAIL，因为 branch_explore 仍按 median 选最快分支。

- [ ] **步骤 3：改造 branch_explore 输出**

每个正确候选调用 paired_benchmark 和 paired_stats，结果写入：

~~~python
{
    "branch_index": index,
    "kernel": kernel,
    "correctness": "passed",
    "statistics": statistics,
    "status": statistics["status"],
}
~~~

shortlist 只包含 confirmed_win，并按 estimate_pct 降序排列。无 confirmed_win 时不复制候选覆盖当前 best。

- [ ] **步骤 4：删除 average_ms promotion 路径**

run_iteration.py 的 summary 使用：

~~~python
{
    "statistic": statistics["statistic"],
    "estimate_pct": statistics["estimate_pct"],
    "ci_low_pct": statistics["ci_low_pct"],
    "ci_high_pct": statistics["ci_high_pct"],
    "status": statistics["status"],
}
~~~

state.py 只接受 decision.json 的 terminal status 推进 best；不再从 bench.kernel.average_ms 推断 improved。

- [ ] **步骤 5：运行 branch、state 和 benchmark 测试**

运行：

~~~bash
python -m unittest tests.test_branch_explore tests.test_state_schema tests.test_benchmark -v
~~~

预期：全部 PASS，且不再存在 median ranking/global average promotion 混用。

- [ ] **步骤 6：提交**

~~~bash
git add skills/cuda-kernel-optimizer/scripts/branch_explore.py \
  skills/cuda-kernel-optimizer/scripts/run_iteration.py \
  skills/cuda-kernel-optimizer/scripts/state.py \
  tests/test_branch_explore.py tests/test_state_schema.py
git commit -m "fix: unify branch and global performance decisions"
~~~

## 任务 6：用户 workload contract 与 preflight

**文件：**

- 创建：skills/cuda-kernel-optimizer/scripts/workload_adapter.py
- 创建：skills/cuda-kernel-optimizer/templates/objective.schema.json
- 创建：skills/cuda-kernel-optimizer/templates/workload.py
- 创建：tests/test_workload_adapter.py
- 修改：skills/cuda-kernel-optimizer/scripts/preflight.py

- [ ] **步骤 1：编写三种输入、互斥、objective 和 cleanup 测试**

~~~python
class WorkloadAdapterTests(unittest.TestCase):
    def test_python_adapter_requires_complete_contract(self):
        path = self.write_adapter("def prepare(candidate): pass\n")
        with self.assertRaisesRegex(ValueError, "validate.*benchmark.*metrics.*cleanup"):
            workload_adapter.load_python_adapter(path)

    def test_command_requires_external_objective(self):
        with self.assertRaisesRegex(ValueError, "--objective"):
            workload_adapter.normalize_workload(command=["python", "run.py"])

    def test_manifest_embedded_and_external_objective_conflict(self):
        with self.assertRaisesRegex(ValueError, "conflicting objective"):
            workload_adapter.normalize_workload(
                manifest=self.manifest_with_objective,
                objective=self.external_objective,
            )

    def test_cleanup_runs_after_benchmark_failure(self):
        adapter = self.fake_adapter(benchmark_error=RuntimeError("boom"))
        with self.assertRaisesRegex(RuntimeError, "boom"):
            workload_adapter.run_once(adapter, candidate="candidate.so")
        self.assertEqual(adapter.calls[-1], "cleanup")
~~~

- [ ] **步骤 2：运行测试确认失败**

运行：

~~~bash
python -m unittest tests.test_workload_adapter -v
~~~

预期：FAIL，缺少 workload_adapter.py。

- [ ] **步骤 3：实现规范化 WorkloadSpec**

~~~python
@dataclass(frozen=True)
class WorkloadSpec:
    kind: str
    source: str | list[str]
    objective: dict
    cases: tuple[dict, ...]
    source_hash: str


REQUIRED_ADAPTER_CALLS = (
    "prepare", "validate", "benchmark", "metrics", "cleanup"
)
~~~

command 使用 shlex.split 后以 shell=False 执行，并通过环境变量传递：

- CUDA_OPTIMIZER_CANDIDATE
- CUDA_OPTIMIZER_ROLE
- CUDA_OPTIMIZER_OUTPUT
- CUDA_OPTIMIZER_CASE

command 必须向 CUDA_OPTIMIZER_OUTPUT 写 JSON，不从 stdout 猜测结构。

- [ ] **步骤 4：在 preflight.py 增加 workload 检查**

preflight 新参数：

~~~text
--workload
--workload-cmd
--workload-manifest
--objective
~~~

多于一种 workload form 直接失败。未提供 workload 时输出 mode=kernel-only；提供 workload 时 objective 和 lifecycle 全部通过才输出 mode=full。

- [ ] **步骤 5：运行 workload 和原 preflight 测试**

运行：

~~~bash
python -m unittest tests.test_workload_adapter -v
python skills/cuda-kernel-optimizer/scripts/preflight.py --help
~~~

预期：测试 PASS，help 返回 0 并显示四个新参数。

- [ ] **步骤 6：提交**

~~~bash
git add skills/cuda-kernel-optimizer/scripts/workload_adapter.py \
  skills/cuda-kernel-optimizer/scripts/preflight.py \
  skills/cuda-kernel-optimizer/templates/objective.schema.json \
  skills/cuda-kernel-optimizer/templates/workload.py \
  tests/test_workload_adapter.py
git commit -m "feat: add user-owned workload contract"
~~~

## 任务 7：真实 workload paired 外环与决策引擎

**文件：**

- 创建：skills/cuda-kernel-optimizer/scripts/workload_evaluate.py
- 创建：skills/cuda-kernel-optimizer/scripts/decision.py
- 创建：tests/test_workload_evaluate.py
- 创建：tests/test_decision.py

- [ ] **步骤 1：编写 retry、primary KPI、constraint 和 Pareto 测试**

~~~python
class WorkloadEvaluateTests(unittest.TestCase):
    def test_transient_failure_retries_twice_then_succeeds(self):
        adapter = FakeAdapter(outcomes=[RuntimeError("x"), RuntimeError("x"), {"latency": 9.0}])
        result = workload_evaluate.measure_candidate(adapter, "candidate", retries=2)
        self.assertEqual(result["metrics"]["latency"], 9.0)
        self.assertEqual(result["attempts"], 3)

    def test_persistent_failure_has_no_partial_win(self):
        adapter = FakeAdapter(outcomes=[{"latency": 9.0}, RuntimeError("x"), RuntimeError("x")])
        result = workload_evaluate.evaluate_pairs(adapter, blocks=3, retries=1)
        self.assertEqual(result["status"], "workload_failed")
        self.assertNotIn("confirmed_win", json.dumps(result))


class DecisionTests(unittest.TestCase):
    def test_end_to_end_win_requires_primary_win_and_all_constraints(self):
        result = decision.decide(
            mode="full",
            kernel={"status": "confirmed_win"},
            workload={"primary": {"status": "confirmed_win"}},
            constraints=[{"name": "p99", "status": "passed"}],
        )
        self.assertEqual(result["status"], "end_to_end_win")

    def test_constraint_regression_rejects_candidate(self):
        result = decision.decide(
            mode="full",
            kernel={"status": "confirmed_win"},
            workload={"primary": {"status": "confirmed_win"}},
            constraints=[{"name": "memory", "status": "failed"}],
        )
        self.assertEqual(result["status"], "rejected_constraint")

    def test_full_mode_without_workload_win_is_kernel_only_win(self):
        result = decision.decide(
            mode="full",
            kernel={"status": "confirmed_win"},
            workload={"primary": {"status": "inconclusive"}},
            constraints=[],
        )
        self.assertEqual(result["status"], "kernel_only_win")
~~~

- [ ] **步骤 2：运行测试确认失败**

运行：

~~~bash
python -m unittest tests.test_workload_evaluate tests.test_decision -v
~~~

预期：FAIL，两个模块不存在。

- [ ] **步骤 3：实现 workload pair collection**

每个外环 block 随机执行 best/candidate 或 candidate/best。每次调用 run_once，保留完整 metrics 和 retry 记录。primary metric 和每个 constraint 分别构造 paired_stats 输入。

~~~python
{
    "order": "AB",
    "baseline_metrics": {"p50_latency_ms": 10.0, "p99_latency_ms": 12.0},
    "candidate_metrics": {"p50_latency_ms": 9.7, "p99_latency_ms": 12.1},
    "valid": True,
    "attempts": {"baseline": 1, "candidate": 1},
}
~~~

约束 pass 条件：其 regression 置信区间上界不得超过 max_regression_pct；不充分证据视为 inconclusive，不能晋级。

- [ ] **步骤 4：实现 terminal status**

decision.py 只允许以下状态：

~~~python
TERMINAL_STATUSES = {
    "rejected_compile",
    "rejected_correctness",
    "rejected_constraint",
    "confirmed_loss",
    "inconclusive",
    "kernel_only_win",
    "end_to_end_win",
    "pareto_frontier",
}
~~~

full 模式只有 end_to_end_win 能推进全局 best。kernel-only 模式可将 confirmed kernel win 记为 kernel_only_win 并推进 kernel best，但 summary 必须保留 kernel-only 标签。

- [ ] **步骤 5：运行外环与决策测试**

运行：

~~~bash
python -m unittest tests.test_workload_evaluate tests.test_decision tests.test_paired_stats -v
~~~

预期：全部 PASS。

- [ ] **步骤 6：提交**

~~~bash
git add skills/cuda-kernel-optimizer/scripts/workload_evaluate.py \
  skills/cuda-kernel-optimizer/scripts/decision.py \
  tests/test_workload_evaluate.py tests/test_decision.py
git commit -m "feat: add real-workload outer-loop decisions"
~~~

## 任务 8：预算感知 orchestrator、checkpoint 与 resume

**文件：**

- 修改：skills/cuda-kernel-optimizer/scripts/orchestrate.py
- 修改：skills/cuda-kernel-optimizer/scripts/state.py
- 创建：tests/test_orchestrate.py

- [ ] **步骤 1：编写默认 balanced、deadline、outer shortlist 和 resume 测试**

~~~python
class OrchestrateTests(unittest.TestCase):
    def test_setup_defaults_to_balanced(self):
        args = orchestrate.build_parser().parse_args([
            "setup", "--baseline", "a.py", "--ref", "r.py", "--dims", "{}"
        ])
        self.assertEqual(args.budget, "balanced")

    def test_deadline_stops_new_candidate_and_checkpoints(self):
        result = orchestrate.schedule_next(
            state=self.state,
            clock=FakeClock(can_start=False),
            estimated_seconds=60,
        )
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["candidate_status"], "inconclusive")
        self.assertTrue(result["checkpoint_written"])

    def test_full_mode_sends_only_confirmed_kernel_wins_to_outer_loop(self):
        shortlisted = orchestrate.select_outer_candidates(
            [
                {"id": "c1", "status": "inconclusive", "estimate_pct": 2.0},
                {"id": "c2", "status": "confirmed_win", "estimate_pct": 1.5},
            ],
            limit=2,
        )
        self.assertEqual([item["id"] for item in shortlisted], ["c2"])

    def test_resume_rejects_changed_input_hash(self):
        with self.assertRaisesRegex(ValueError, "frozen input"):
            orchestrate.resume(self.checkpoint, input_hash="changed")
~~~

- [ ] **步骤 2：运行测试确认失败**

运行：

~~~bash
python -m unittest tests.test_orchestrate -v
~~~

预期：FAIL，缺少 build_parser、schedule_next、select_outer_candidates 和 resume。

- [ ] **步骤 3：拆出 build_parser 并增加 CLI**

setup 增加：

~~~text
--budget {quick,balanced,thorough,custom}
--max-seconds
--max-rounds
--branches
--min-pairs
--max-pairs
--confidence
--min-effect-pct
--outer-candidates
--output-root
--workload
--workload-cmd
--workload-manifest
--objective
~~~

新增：

~~~text
orchestrate.py resume --run-dir <run>
~~~

默认 budget=balanced；旧 --noise-threshold-pct 退出公开 CLI，并在使用时给出迁移错误，不能静默映射。
--output-root 默认为 baseline 所在目录；显式提供时，manifest、candidate、
checkpoint 和 summary 全部写入该目录下的新 run_<timestamp>。

- [ ] **步骤 4：接入双环 stage machine**

checkpoint.stage 只允许：

~~~python
STAGES = (
    "baseline",
    "candidate_correctness",
    "candidate_paired",
    "candidate_profile",
    "candidate_sanitizer",
    "workload_paired",
    "decision",
    "complete",
)
~~~

每个 stage 完成后原子保存 checkpoint。deadline 前五分钟进入 cleanup reserve；未完成 stage 在安全取消点终止并标记 inconclusive。

- [ ] **步骤 5：接入 state promotion**

orchestrator 将 decision.json 传给 state.py：

- end_to_end_win：full 模式更新 best_file 和 best_workload_statistics。
- kernel_only_win：只在 kernel-only 模式更新 best_file；full 模式仅保存 candidate/frontier。
- 其他状态：不推进 best。

- [ ] **步骤 6：运行 orchestrator 与所有相关 CPU 测试**

运行：

~~~bash
python -m unittest tests.test_orchestrate tests.test_budget \
  tests.test_workload_adapter tests.test_workload_evaluate \
  tests.test_decision tests.test_state_schema -v
python skills/cuda-kernel-optimizer/scripts/orchestrate.py --help
python skills/cuda-kernel-optimizer/scripts/orchestrate.py setup --help
python skills/cuda-kernel-optimizer/scripts/orchestrate.py resume --help
~~~

预期：全部 PASS，三个 help 命令返回 0。

- [ ] **步骤 7：提交**

~~~bash
git add skills/cuda-kernel-optimizer/scripts/orchestrate.py \
  skills/cuda-kernel-optimizer/scripts/state.py tests/test_orchestrate.py
git commit -m "feat: orchestrate budgeted dual-loop runs"
~~~

## 任务 9：Compute Sanitizer 正确性闸门

**文件：**

- 创建：skills/cuda-kernel-optimizer/scripts/sanitize.py
- 创建：skills/cuda-kernel-optimizer/references/sanitizer_policy.json
- 创建：tests/test_sanitize.py
- 修改：skills/cuda-kernel-optimizer/scripts/orchestrate.py

- [ ] **步骤 1：编写 targeted/full、缺失工具和返回码测试**

~~~python
class SanitizeTests(unittest.TestCase):
    def test_targeted_async_method_selects_memcheck_racecheck_synccheck(self):
        tools = sanitize.select_tools(
            method_ids=["latency.async_pipeline"], mode="targeted",
            policy=self.policy,
        )
        self.assertEqual(tools, ["memcheck", "racecheck", "synccheck"])

    def test_full_always_runs_all_tools(self):
        self.assertEqual(
            sanitize.select_tools([], mode="full", policy=self.policy),
            ["memcheck", "racecheck", "initcheck", "synccheck"],
        )

    def test_missing_compute_sanitizer_is_explicit_not_passed(self):
        result = sanitize.run_tools(
            executable=None, tools=["memcheck"], command=["python", "bench.py"]
        )
        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["passed"])
~~~

- [ ] **步骤 2：运行测试确认失败**

运行：

~~~bash
python -m unittest tests.test_sanitize -v
~~~

预期：FAIL，sanitize.py 不存在。

- [ ] **步骤 3：实现显式 policy 和命令**

sanitizer_policy.json 为每个需要闸门的方法列出 tools。sanitize.py 命令必须使用：

~~~python
[
    compute_sanitizer,
    "--tool", tool,
    "--error-exitcode", "86",
    *benchmark_command,
]
~~~

每个 tool 保存 command、returncode、stdout、stderr 和 status。不得加入 sudo、SYS_ADMIN、privileged 或驱动修改。

- [ ] **步骤 4：接入 orchestrator**

- quick/balanced：只对 policy 命中的方法执行 targeted。
- thorough：对进入外环的 finalist 执行 full。
- failed：rejected_correctness。
- unavailable：保存缺失证据并继续，但 summary 标记 sanitizer coverage degraded。

- [ ] **步骤 5：运行 sanitizer、orchestrator 和 NCU 降级测试**

运行：

~~~bash
python -m unittest tests.test_sanitize tests.test_orchestrate tests.test_profile_ncu -v
~~~

预期：全部 PASS。

- [ ] **步骤 6：提交**

~~~bash
git add skills/cuda-kernel-optimizer/scripts/sanitize.py \
  skills/cuda-kernel-optimizer/scripts/orchestrate.py \
  skills/cuda-kernel-optimizer/references/sanitizer_policy.json \
  tests/test_sanitize.py
git commit -m "feat: add sanitizer correctness gates"
~~~

## 任务 10：compiler evidence 与 provenance

**文件：**

- 创建：skills/cuda-kernel-optimizer/scripts/compiler_evidence.py
- 创建：tests/test_compiler_evidence.py
- 修改：skills/cuda-kernel-optimizer/scripts/benchmark.py
- 修改：skills/cuda-kernel-optimizer/scripts/sass_check.py

- [ ] **步骤 1：编写 available/unavailable stage、hash 和相同二进制测试**

~~~python
class CompilerEvidenceTests(unittest.TestCase):
    def test_records_hashes_and_missing_stages_without_fabrication(self):
        result = compiler_evidence.collect(
            source=self.source,
            binary=self.binary,
            discovered={"ptx": self.ptx},
            compile_command=["nvcc", "kernel.cu"],
        )
        self.assertEqual(result["source"]["status"], "available")
        self.assertEqual(result["ptx"]["status"], "available")
        self.assertEqual(result["ttgir"]["status"], "unavailable")
        self.assertRegex(result["binary"]["sha256"], r"^[0-9a-f]{64}$")

    def test_identical_binary_is_reported(self):
        self.assertTrue(
            compiler_evidence.same_artifact(self.binary_a, self.binary_b)
        )
~~~

- [ ] **步骤 2：运行测试确认失败**

运行：

~~~bash
python -m unittest tests.test_compiler_evidence -v
~~~

预期：FAIL，compiler_evidence.py 不存在。

- [ ] **步骤 3：实现 evidence manifest**

固定 stage keys：

~~~python
STAGES = ("source", "ttir", "ttgir", "llvm_ir", "ptx", "sass", "binary")
~~~

每项输出 status、path、sha256、size_bytes。collect 只登记实际存在的文件；不存在时 status=unavailable。不得为了取 line info 改变 production compile flags。

- [ ] **步骤 4：接入编译和 SASS**

benchmark.compile_cu 记录 nvcc command、backend、arch 和输出 binary。Triton 只收集其 cache 中实际发现的 IR/PTX；找不到则 unavailable。sass_check 将 dump 保存为 compiler_evidence/sass.txt 并注册 hash。

- [ ] **步骤 5：运行 compiler、benchmark、SASS 相关测试**

运行：

~~~bash
python -m unittest tests.test_compiler_evidence tests.test_benchmark \
  tests.test_profile_ncu -v
~~~

预期：全部 PASS。

- [ ] **步骤 6：提交**

~~~bash
git add skills/cuda-kernel-optimizer/scripts/compiler_evidence.py \
  skills/cuda-kernel-optimizer/scripts/benchmark.py \
  skills/cuda-kernel-optimizer/scripts/sass_check.py \
  tests/test_compiler_evidence.py
git commit -m "feat: preserve compiler provenance evidence"
~~~

## 任务 11：双层 summary 与可复算结论

**文件：**

- 创建：tests/test_summarize.py
- 修改：skills/cuda-kernel-optimizer/scripts/summarize.py

- [ ] **步骤 1：编写 headline、kernel-only、degraded 和 raw sample 链接测试**

~~~python
class SummarizeTests(unittest.TestCase):
    def test_full_win_has_separate_kernel_and_workload_sections(self):
        text = summarize.render_text(self.full_win_state)
        self.assertIn("# Result: end_to_end_win", text)
        self.assertIn("## Kernel evidence", text)
        self.assertIn("## Real workload evidence", text)
        self.assertIn("paired_samples.jsonl", text)

    def test_missing_workload_cannot_render_end_to_end_win(self):
        text = summarize.render_text(self.kernel_only_state)
        self.assertIn("# Result: kernel_only_win", text)
        self.assertIn("No user workload was supplied", text)
        self.assertNotIn("# Result: end_to_end_win", text)

    def test_ncu_or_sanitizer_degradation_is_prominent(self):
        text = summarize.render_text(self.degraded_state)
        self.assertIn("ERR_NVGPUCTRPERM", text)
        self.assertIn("sanitizer coverage: unavailable", text)
~~~

- [ ] **步骤 2：运行测试确认失败**

运行：

~~~bash
python -m unittest tests.test_summarize -v
~~~

预期：FAIL，缺少 render_text 和新章节。

- [ ] **步骤 3：实现 answer-first summary**

固定顺序：

1. terminal result、budget、mode。
2. frozen inputs 和 environment。
3. kernel evidence：estimate、CI、pairs、correctness、SASS。
4. real workload evidence：primary KPI、constraints、CI。
5. profiler/sanitizer/compiler coverage。
6. best、frontier、rejected 和 inconclusive。
7. raw artifact 路径和 resume 状态。

render(state_path, out_path) 调用纯函数 render_text(state)，便于测试。

- [ ] **步骤 4：运行 summary 和 state 测试**

运行：

~~~bash
python -m unittest tests.test_summarize tests.test_state_schema tests.test_decision -v
~~~

预期：全部 PASS。

- [ ] **步骤 5：提交**

~~~bash
git add skills/cuda-kernel-optimizer/scripts/summarize.py tests/test_summarize.py
git commit -m "feat: report kernel and workload evidence separately"
~~~

## 任务 12：更新 skill 指令、metadata 和中英文 README

**文件：**

- 修改：skills/cuda-kernel-optimizer/SKILL.md
- 修改：skills/cuda-kernel-optimizer/agents/openai.yaml
- 修改：skills/cuda-kernel-optimizer/examples/walkthrough.md
- 修改：skills/cuda-kernel-optimizer/templates/iteration_report.md
- 修改：README.md
- 修改：README.zh-CN.md
- 修改：tests/test_readme_sync.py
- 修改：tests/test_skill_metadata.py

- [ ] **步骤 1：先把文档测试改成 v2.2 契约**

~~~python
def test_readmes_identify_v2_2_dual_loop_and_balanced_default(self):
    for text in (self.english, self.chinese):
        self.assertIn("V2.2", text)
        self.assertIn("balanced", text)
        self.assertIn("3", text)
        self.assertIn("kernel_only_win", text)
        self.assertIn("end_to_end_win", text)

def test_skill_requires_user_owned_workload_for_end_to_end_claims(self):
    text = SKILL_MD.read_text(encoding="utf-8")
    self.assertIn("user-provided workload", text)
    self.assertIn("paired", text.lower())
    self.assertIn("inconclusive", text)
    self.assertIn("ERR_NVGPUCTRPERM", text)
~~~

- [ ] **步骤 2：运行测试确认失败**

运行：

~~~bash
python -m unittest tests.test_readme_sync tests.test_skill_metadata -v
~~~

预期：FAIL，文档仍标记 V2.1。

- [ ] **步骤 3：更新 SKILL.md**

SKILL.md 保持 500 行以内，只保留：

- 输入和 user-owned workload 规则。
- budget preset 和 balanced 默认值。
- 双环操作顺序。
- paired verdict 和 promotion gate。
- failure/degraded 行为。
- 输出 contract。

详细 objective schema、sanitizer policy 和 compatibility 通过直接链接按需读取，不复制进 SKILL.md。

- [ ] **步骤 4：更新 README、walkthrough、template 和 metadata**

两份 README 必须同步说明：

- V2.2 双环和预算表。
- 三种 workload 输入。
- kernel-only 与 end-to-end 的区别。
- 新 artifact tree。
- resume 命令。
- 5090 测试和 NCU 权限事实。

openai.yaml 的 short_description 改为：

~~~yaml
short_description: "Trustworthy CUDA kernel and workload optimization"
~~~

default_prompt 同时提及 reference 和可选 user-provided workload。

- [ ] **步骤 5：运行文档与 skill 校验**

运行：

~~~bash
python -m unittest tests.test_readme_sync tests.test_skill_metadata -v
python /Users/tcheng/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  skills/cuda-kernel-optimizer
git diff --check
~~~

预期：unittest 全部 PASS；quick_validate 输出 Skill is valid；git diff --check 无输出。

- [ ] **步骤 6：提交**

~~~bash
git add README.md README.zh-CN.md \
  skills/cuda-kernel-optimizer/SKILL.md \
  skills/cuda-kernel-optimizer/agents/openai.yaml \
  skills/cuda-kernel-optimizer/examples/walkthrough.md \
  skills/cuda-kernel-optimizer/templates/iteration_report.md \
  tests/test_readme_sync.py tests/test_skill_metadata.py
git commit -m "docs: document v2.2 dual-loop workflow"
~~~

## 任务 13：完整 CPU 回归与本地 skill 同步

**文件：**

- 不修改仓库实现文件；只验证并更新本机已安装副本。

- [ ] **步骤 1：运行完整 CPU 测试**

运行：

~~~bash
python -m unittest discover -s tests -p 'test_*.py' -v
~~~

预期：全部非 GPU opt-in 测试 PASS，SM120 测试显示 skipped。

- [ ] **步骤 2：运行脚本 help、skill 和格式校验**

运行：

~~~bash
for script in skills/cuda-kernel-optimizer/scripts/*.py; do
  python "$script" --help >/dev/null
done
python /Users/tcheng/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  skills/cuda-kernel-optimizer
git diff --check
~~~

预期：所有 help 返回 0；Skill is valid；git diff --check 无输出。

- [ ] **步骤 3：安全更新本机安装副本**

运行：

~~~bash
stamp="$(date +%Y%m%d-%H%M%S)"
staging="/Users/tcheng/.codex/skills/cuda-kernel-optimizer.staging-$stamp"
backup="/Users/tcheng/.codex/skills/cuda-kernel-optimizer.backup-$stamp"
cp -a skills/cuda-kernel-optimizer "$staging"
python /Users/tcheng/.codex/skills/.system/skill-creator/scripts/quick_validate.py "$staging"
mv /Users/tcheng/.codex/skills/cuda-kernel-optimizer "$backup"
mv "$staging" /Users/tcheng/.codex/skills/cuda-kernel-optimizer
~~~

预期：staging 校验通过后才切换；旧安装保留为带时间戳 backup。

- [ ] **步骤 4：验证安装内容与仓库一致**

运行：

~~~bash
diff -qr skills/cuda-kernel-optimizer \
  /Users/tcheng/.codex/skills/cuda-kernel-optimizer
~~~

预期：无输出。

## 任务 14：RTX 5090 受控双环和真实 workload 验收

**文件：**

- 创建：tests/gpu/sm120/fixtures/workload_smoke.py
- 创建：tests/gpu/sm120/fixtures/objective.json
- 修改：tests/gpu/sm120/test_sm120_acceptance.py
- 修改：tests/gpu/sm120/remote/run_lane.sh
- 修改：tests/gpu/sm120/README.md
- 远端创建：/data/tcheng/cuda-skill-e2e/v2.2/
- 远端写入：/data/tcheng/cuda-skill-e2e/v2.2/artifacts/**

- [ ] **步骤 1：先增加 opt-in GPU acceptance 测试**

~~~python
def test_paired_noop_is_not_reported_as_win(self):
    result = self.run_paired(
        baseline=FIXTURES / "triton_vector.py",
        candidate=FIXTURES / "triton_vector.py",
        max_pairs=30,
    )
    self.assertEqual(result["statistics"]["status"], "inconclusive")

def test_user_workload_smoke_produces_separate_outer_evidence(self):
    result = self.run_orchestrator(
        workload=FIXTURES / "workload_smoke.py",
        objective=FIXTURES / "objective.json",
        budget="custom",
        max_seconds=900,
    )
    self.assertIn(result["decision"]["status"], {
        "end_to_end_win", "kernel_only_win", "inconclusive"
    })
    self.assertTrue(result["workload_result"]["raw_pairs"])
~~~

- [ ] **步骤 2：运行本地测试确认 opt-in skip**

运行：

~~~bash
python -m unittest tests.gpu.sm120.test_sm120_acceptance -v
~~~

预期：没有 CUDA_SM120_E2E=1 时全部 skipped。

- [ ] **步骤 3：同步隔离 repo 到 5090**

运行：

~~~bash
ssh 5090 'mkdir -p /data/tcheng/cuda-skill-e2e/v2.2/repo \
  /data/tcheng/cuda-skill-e2e/v2.2/artifacts'
rsync -a --delete --exclude .git/ \
  ./ 5090:/data/tcheng/cuda-skill-e2e/v2.2/repo/
~~~

预期：只更新 /data/tcheng/cuda-skill-e2e/v2.2/repo，不触碰 /data/vllm-opt。

- [ ] **步骤 4：运行 current lane 受控矩阵**

运行：

~~~bash
ssh 5090 'cd /data/tcheng/cuda-skill-e2e/v2.2/repo && \
  CUDA_SM120_E2E=1 \
  CUDA_E2E_ARTIFACTS=/data/tcheng/cuda-skill-e2e/v2.2/artifacts/current \
  CUTLASS_PATH=/data/tcheng/cuda-skill-e2e/deps/cutlass \
  python3 -m unittest tests.gpu.sm120.test_sm120_acceptance -v'
~~~

预期：CUDA、CUTLASS、Triton correctness/timing、noop inconclusive、workload smoke 全部 PASS。

- [ ] **步骤 5：运行 compatibility lane**

使用 tests/gpu/sm120/remote/run_lane.sh 和现有 pinned compatibility image：

~~~bash
ssh 5090 'cd /data/tcheng/cuda-skill-e2e/v2.2/repo && \
  CUDA_E2E_ARTIFACTS=/data/tcheng/cuda-skill-e2e/v2.2/artifacts/compatibility \
  CUTLASS_PATH=/data/tcheng/cuda-skill-e2e/deps/cutlass \
  tests/gpu/sm120/remote/run_lane.sh'
~~~

预期：三 backend 和新 paired/noop 测试 PASS；NCU 若返回 ERR_NVGPUCTRPERM，结果必须 degraded 且测试通过。

- [ ] **步骤 6：运行用户提供的隔离 vLLM workload**

把用户已提供、已隔离的 workload adapter 放在：

~~~text
/data/tcheng/cuda-skill-e2e/real-workload/source/workload.py
~~~

运行 balanced 模式，并将所有产物写到 v2.2 目录：

~~~bash
ssh 5090 'cd /data/tcheng/cuda-skill-e2e/v2.2/repo && \
  CUDA_VISIBLE_DEVICES=1 \
  python3 skills/cuda-kernel-optimizer/scripts/orchestrate.py setup \
    --baseline /data/tcheng/cuda-skill-e2e/real-workload/source/baseline.py \
    --ref /data/tcheng/cuda-skill-e2e/real-workload/source/reference.py \
    --dims '"'"'{"M":1,"N":8704,"K":5120}'"'"' \
    --workload /data/tcheng/cuda-skill-e2e/real-workload/source/workload.py \
    --budget balanced \
    --output-root /data/tcheng/cuda-skill-e2e/v2.2/artifacts/real \
    --env-out /data/tcheng/cuda-skill-e2e/v2.2/artifacts/real/env.json'
~~~

setup 返回 run_dir 后，执行者按 next_step 生成每轮候选、调用 close-iter，
并在结束时调用 finalize。预期：baseline、paired kernel、outer workload、
decision、checkpoint 和 summary 都存在。相同源码/二进制不得报 win；真实候选
只能按保存的 paired CI 得出结论。

- [ ] **步骤 7：验证预算、resume 和原始样本可复算**

运行：

~~~bash
run_dir="$(ssh 5090 'ls -dt /data/tcheng/cuda-skill-e2e/v2.2/artifacts/real/run_* | head -1')"
ssh 5090 "python3 /data/tcheng/cuda-skill-e2e/v2.2/repo/skills/cuda-kernel-optimizer/scripts/orchestrate.py \
  resume --run-dir '$run_dir'"
~~~

预期：已完成 stage 不重复；input hash 匹配；summary 中统计值可由 paired_samples.jsonl 复算；实际 elapsed 不超过预算 manifest 的 max_seconds。

- [ ] **步骤 8：提交 GPU 测试和证据说明**

~~~bash
git add tests/gpu/sm120
git commit -m "test: validate v2.2 dual-loop on RTX 5090"
~~~

## 任务 15：最终回归、分支集成和仅向 fork 推送

**文件：**

- 修改：README.md
- 修改：README.zh-CN.md
- 修改：skills/cuda-kernel-optimizer/references/compatibility.md
- 修改：docs/superpowers/plans/2026-07-16-cuda-skill-v2-2-dual-loop.md

- [ ] **步骤 1：记录实际 5090 版本与结果**

将 current/compatibility 工具版本、测试数、真实 workload verdict、elapsed、NCU verdict 和 artifact 路径写入两份 README、compatibility.md 和本计划的执行记录。不得写入 token、cookie、私有 host 凭据或 /data/vllm-opt 的未提交内容。

- [ ] **步骤 2：运行最终完整验证**

运行：

~~~bash
python -m unittest discover -s tests -p 'test_*.py' -v
python /Users/tcheng/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  skills/cuda-kernel-optimizer
git diff --check
git status --short
~~~

预期：CPU 测试全部 PASS，GPU opt-in 本地 skipped，Skill is valid，diff check 无输出，status 只包含预期文档更新。

- [ ] **步骤 3：提交 release evidence**

~~~bash
git add README.md README.zh-CN.md \
  skills/cuda-kernel-optimizer/references/compatibility.md \
  docs/superpowers/plans/2026-07-16-cuda-skill-v2-2-dual-loop.md
git commit -m "docs: record v2.2 validation evidence"
~~~

- [ ] **步骤 4：确认远端保护**

运行：

~~~bash
git remote -v
git remote get-url --push origin
git remote get-url --push upstream
~~~

预期：

- origin push 指向 troycheng/cuda-optimized-skill。
- upstream push 为 DISABLED 或其他不可推送地址。

- [ ] **步骤 5：合并到 main**

在主 worktree：

~~~bash
git checkout main
git merge --ff-only agent/v2-2-dual-loop
~~~

预期：fast-forward 成功，无 merge commit。

- [ ] **步骤 6：重新验证 main**

运行：

~~~bash
python -m unittest discover -s tests -p 'test_*.py' -v
python /Users/tcheng/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  skills/cuda-kernel-optimizer
git diff --check
git status --short --branch
~~~

预期：全部 PASS；main 工作区干净，只领先 origin/main。

- [ ] **步骤 7：只推送 fork main**

运行：

~~~bash
git push origin main
git ls-remote origin refs/heads/main
~~~

预期：origin/main 指向本地 HEAD。不得运行 git push upstream。

- [ ] **步骤 8：最终同步本机已安装 skill**

重复任务 13 的 staging、quick_validate、backup、atomic move 和 diff -qr。最终安装内容必须与已推送 main 的 skills/cuda-kernel-optimizer 完全一致。

## 总体验收命令

~~~bash
python -m unittest discover -s tests -p 'test_*.py' -v
python /Users/tcheng/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  skills/cuda-kernel-optimizer
git diff --check
git status --short --branch
~~~

全部成功后才能声明 v2.2 完成。RTX 5090 和真实 workload 证据必须单独列出，不能用 CPU 测试替代。
