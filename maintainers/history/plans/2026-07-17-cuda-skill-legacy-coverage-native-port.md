# CUDA Kernel Optimizer 旧版覆盖能力原生迁移实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用
> `superpowers:subagent-driven-development`（推荐）或
> `superpowers:executing-plans` 逐任务实现此计划；每个生产改动先执行
> `superpowers:test-driven-development`，宣称完成前执行
> `superpowers:verification-before-completion`。用下列复选框跟踪进度。

**目标：** 在不合并旧分支、不改变 v2.2 双环决策语义的前提下，补齐
现有 `.ncu-rep` 的安全分析、显式选择的跨 run 策略记忆，以及 serving、
systems、CUTLASS 和 Triton IR 的证据指南，并用任务优先、自然双语的 README
让 fork 易懂、易用。

**架构：** 两个新增 CLI 都是核心 orchestrator 之外的独立工具。
`analyze_ncu_rep.py` 只读取既有 report，复用当前 NCU 指标分类，并以
`analysis.json` 作为最后发布的完成标记；`strategy_memory.py` 只从已经完成且
重新验证过的 v2.2 run 中采集证据，建议保持 advisory，唯一 promotion 权限仍
属于 run 内的 `decision.json`。新增文档通过 `SKILL.md` 渐进披露，不扩展核心
状态机。

**技术栈：** Python 3.10+ 标准库、`unittest`、POSIX process groups、
`fcntl.flock`、安全 no-follow artifact helpers、Nsight Compute CLI、RTX 5090
只读 report 验证、Git。

---

## 执行边界

- 工作树：
  `/Users/tcheng/Documents/Codex/2026-07-15-triton-skill/cuda-optimized-skill/.worktrees/legacy-coverage-v2-2`
- 分支：`agent/legacy-coverage-v2-2`
- 当前基线：`main` 的 v2.2 实现，加已批准规格提交 `9f93d09` 与规格闭环提交
  `da31cbb`。
- 不 merge/cherry-pick `origin/agent/complete-legacy-skill-coverage`。
- 不修改 `orchestrate.py` 的 CLI、stage、budget、workload outer loop 或
  promotion 语义。
- 不把 strategy memory 设置为默认路径，也不从 orchestrator 自动调用。
- 远端 5090 只允许写独立测试根目录，只复制既有 `.ncu-rep`；不修改驱动、
  NCU counter 权限、用户源码或其他 worktree。
- 所有变更先保留在本地分支；用户确认前不合并到 fork `main`、不 push。
- README 只能在实现和 5090 验证完成后重写，必须描述实测行为而非计划行为。
- GitHub About 只能在用户批准、fork `main` 推送成功后更新；必须先后验证目标
  是 `troycheng/cuda-optimized-skill`，且 parent 仍为
  `KernelFlow-ops/cuda-optimized-skill`。
- 每个任务遵循 RED → GREEN → focused regression → commit；禁止先写生产代码
  再补测试。

## 文件职责

### 新建

- `skills/cuda-kernel-optimizer/scripts/analyze_ncu_rep.py`：安全导入一个既有
  report，生成有身份绑定的分析 bundle。
- `tests/test_analyze_ncu_rep.py`：路径、进程、解析、发布和 CLI 退出码测试。
- `skills/cuda-kernel-optimizer/scripts/strategy_memory.py`：scope、完成 run
  采集、安全 memory store 与 advisory suggestion。
- `tests/test_strategy_memory.py`：scope、证据绑定、并发、容量和 advisory
  行为测试。
- `skills/cuda-kernel-optimizer/references/serving_evidence_protocol.md`：从
  generated code 到 serving endpoint 的 claim ladder。
- `skills/cuda-kernel-optimizer/references/systems_and_ir_coverage.md`：系统、
  CUTLASS/CuTe 和 Triton IR 的证据路由。

### 修改

- `skills/cuda-kernel-optimizer/SKILL.md`：新增两个按需入口及两份 reference
  链接；继续保持 500 行以内。
- `skills/cuda-kernel-optimizer/agents/openai.yaml`：仅补充触发面，不改变
  workload 和 reference 的现有约束。
- `README.md`、`README.zh-CN.md`：任务优先的独立自然写作版本；事实、命令、
  Mermaid 拓扑、advisory 限制和 reference 保持一致。
- `tests/test_skill_metadata.py`、`tests/test_readme_sync.py`：锁定文档发现性与
  中英文同步。

### 明确不修改

- `skills/cuda-kernel-optimizer/scripts/decision.py`
- `skills/cuda-kernel-optimizer/scripts/state.py`
- `skills/cuda-kernel-optimizer/scripts/orchestrate.py`
- `skills/cuda-kernel-optimizer/scripts/paired_stats.py`
- `skills/cuda-kernel-optimizer/scripts/workload_evaluate.py`

新增 strategy recorder 必须直接调用这些模块已经存在的 validator，不能复制
一套宽松版本。

## 固定数据契约

### Standalone analysis

`analysis.json` 使用 `cuda-kernel-optimizer/ncu-analysis-v1`，至少包含：

```json
{
  "schema_version": "cuda-kernel-optimizer/ncu-analysis-v1",
  "status": "success",
  "counter_access": "not_probed",
  "report": {"path": "/abs/report.ncu-rep", "sha256": "..."},
  "source": null,
  "ncu": {"requested": "ncu", "resolved": "/abs/ncu", "version": "..."},
  "commands": {},
  "metric_count": 1,
  "kernels": ["target"],
  "rankings": {"compute": [], "memory": [], "latency": []},
  "primary_axis": {"axis": "memory", "quality": "heuristic"},
  "limits": [],
  "artifacts": {"raw.csv": {"sha256": "..."}}
}
```

`analysis.json` 不自哈希。所有 supporting files 逐文件原子替换；
`analysis.json` 最后写入并作为唯一完成标记。hard failure 先安全删除旧 marker；
partial 可以发布带 `status: partial` 的 marker 并返回 2。

### Strategy memory

Memory 使用 `cuda-kernel-optimizer/strategy-memory-v1`：

```json
{
  "schema_version": "cuda-kernel-optimizer/strategy-memory-v1",
  "scopes": {
    "<scope_sha256>": {
      "scope": {},
      "runs": [],
      "methods": {},
      "bundles": {}
    }
  }
}
```

`record` 输出 `cuda-kernel-optimizer/strategy-record-v1`，`suggest` 输出
`cuda-kernel-optimizer/strategy-suggestion-v1`。`suggest` 只读 manifest 与
memory，不得写 run 文件；`record` 只读 run，只在显式 `--memory` 和 `--out`
处写入。

---

### 任务 1：建立 analyzer 的安全输入与有界进程基础

**文件：**

- 新建：`tests/test_analyze_ncu_rep.py`
- 新建：`skills/cuda-kernel-optimizer/scripts/analyze_ncu_rep.py`

- [ ] **步骤 1：先写模块与 CLI 的 RED 测试。** 使用与其他测试一致的
  `importlib.util.spec_from_file_location` loader。测试：
  `REPORT`/`SOURCE` 缺失、目录、FIFO、叶子 symlink、父目录 symlink；
  `OUTPUT` 文件或 symlink；`--ncu-num` 的 `0/-1/true`；`--timeout` 的
  `0/-1/nan/inf/true`。所有失败都必须在执行 NCU 前发生。

```python
def test_report_symlink_is_rejected_before_ncu(self):
    report = self.root / "real.ncu-rep"
    report.write_bytes(b"report")
    link = self.root / "link.ncu-rep"
    link.symlink_to(report)
    with self.assertRaisesRegex(ValueError, "report.*symlink|unsafe"):
        self.module.capture_regular_file(link, field="report")
```

- [ ] **步骤 2：运行 RED。**

```bash
python3 -m unittest tests.test_analyze_ncu_rep -v
```

预期：因 `analyze_ncu_rep.py` 尚不存在而 import 失败；确认失败点是新模块缺失，
不是测试 fixture 错误。

- [ ] **步骤 3：实现最小安全输入层。** 添加：

  - `_strict_positive_int` 与 `_strict_positive_float`，显式拒绝 bool 和
    non-finite；
  - `capture_regular_file(path, field)`，通过 `artifact_store.read_regular_bytes`
    获取 bytes、absolute physical path、size、SHA-256，不使用 `Path.resolve()`
    掩盖用户路径 symlink；
  - `validate_output_directory`，只允许新建或既有真实目录，逐级 no-follow；
  - `resolve_executable`，允许请求名由 PATH 或普通工具链 symlink 解析，但最终
    target 必须是 physical regular executable，同时保存 requested/resolved；
  - `build_parser()` 和 `main()`，但此任务不实现 report import。

- [ ] **步骤 4：补有界 process-group RED 测试。** fake executable 分别：
  输出超过 1 MiB、忽略 TERM 并 fork 同样忽略 TERM 的 child、正常输出无效
  UTF-8。断言 runner 持续 drain pipe 但只保留固定字节，timeout 后 parent 与
  child 都消失，返回 `timed_out/truncated/returncode/stdout/stderr`。

- [ ] **步骤 5：实现 `_run_bounded(argv, timeout, output_limit)`。** 只接受
  argv list；使用 `Popen(start_new_session=True)`、两个 reader thread、TERM、
  有界 grace、KILL、wait 和 thread join。任何异常也必须清理进程组。禁止
  `shell=True`，禁止打印环境。

- [ ] **步骤 6：运行 GREEN 与静态检查。**

```bash
python3 -m unittest tests.test_analyze_ncu_rep -v
python3 -m py_compile skills/cuda-kernel-optimizer/scripts/analyze_ncu_rep.py
python3 skills/cuda-kernel-optimizer/scripts/analyze_ncu_rep.py --help
git diff --check
```

- [ ] **步骤 7：提交。**

```bash
git add tests/test_analyze_ncu_rep.py \
  skills/cuda-kernel-optimizer/scripts/analyze_ncu_rep.py
git commit -m "feat: add bounded ncu report importer foundation"
```

### 任务 2：实现 report import、指标排名与证据发布

**文件：**

- 修改：`tests/test_analyze_ncu_rep.py`
- 修改：`skills/cuda-kernel-optimizer/scripts/analyze_ncu_rep.py`

- [ ] **步骤 1：为命令顺序写 RED 测试。** fake NCU 必须只收到以下 argv：

```text
<resolved-ncu> --version
<resolved-ncu> --import <report> --page summary
<resolved-ncu> --import <report> --page details
<resolved-ncu> --import <report> --csv --page raw
```

断言没有 target launch、没有 shell、每条命令使用相同 resolved executable 和
captured report identity。

- [ ] **步骤 2：为解析与 primary axis 写 RED 测试。** fake raw CSV 同时覆盖
long/wide form、多个 kernel、逗号数字、无效值和三类 metric。预期结果直接由
`profile_ncu._parse_ncu_csv`、`_aggregate_across_kernels`、`_rank_by_axis`
产生；`primary_axis.quality` 固定为 `heuristic`，无可分类 metric 时 axis 为
`unknown`。

- [ ] **步骤 3：为身份漂移与退出码写 RED 测试。** 在 summary 后替换 report
为“同大小、同 mtime、不同 bytes”；另测 source 漂移、summary-only partial、
raw import 失败、timeout。要求：

  - success 返回 0；
  - 至少一个可解释 import 成功但覆盖不完整时返回 2；
  - identity drift、timeout、全部 import 失败返回 1；
  - hard failure 后不存在旧 `analysis.json`。

- [ ] **步骤 4：为发布顺序和 hostile Markdown 写 RED 测试。** 预置旧 marker，
注入 kernel 名 ``![x](file:///tmp/leak)|\n# heading``，patch 发布器在某 supporting
file 后失败。断言：Markdown 中 `[ ] ( ) ! # | < >` 等不形成链接、图片、标题
或表格；supporting artifact 的 hash 与 bytes 一致；失败时不出现新 marker；
`analysis.json` 永远最后发布且不包含自哈希。

- [ ] **步骤 5：实现 import pipeline。** 复用现有 `profile_ncu.py` 分类函数，
不复制 metric regex。每条命令保存 return code、timeout、truncated 和 bounded
stderr；固定生成：`summary.txt`、`summary.stderr.txt`、`details.txt`、
`details.stderr.txt`、`raw.csv`、`analysis.md`。缺失的 partial 输出写空文件并
在 JSON 中标注 unavailable，不制造 metric。

- [ ] **步骤 6：实现稳定 re-open 与 marker-last 发布。** report/source 在所有
imports 后重新通过安全 reader 读取并比较 SHA-256。先
`remove_regular_file(out/analysis.json, missing_ok=True)`，再用
`publish_regular_bundle` 发布 supporting files，最后
`atomic_write_json(out/analysis.json, payload)`。分析 JSON 记录
`counter_access: not_probed` 及三条限制：不证明当前 counter 权限、不证明当前
source 与 report 一致、不证明端到端收益。

- [ ] **步骤 7：运行 focused GREEN。**

```bash
python3 -m unittest tests.test_analyze_ncu_rep -v
python3 -m unittest tests.test_profile_ncu tests.test_artifact_store -v
python3 -m py_compile skills/cuda-kernel-optimizer/scripts/analyze_ncu_rep.py
git diff --check
```

- [ ] **步骤 8：提交。**

```bash
git add tests/test_analyze_ncu_rep.py \
  skills/cuda-kernel-optimizer/scripts/analyze_ncu_rep.py
git commit -m "feat: analyze existing ncu reports with bound evidence"
```

### 任务 3：建立 strategy scope 与安全 memory store

**文件：**

- 新建：`tests/test_strategy_memory.py`
- 新建：`skills/cuda-kernel-optimizer/scripts/strategy_memory.py`

- [ ] **步骤 1：写 scope RED 测试。** 构造最小合法 manifest，覆盖：
  kernel-only/full、不同 key 顺序、相同文件名不同 bytes、dims/backend/arch/
  ptr-size/workload source hash 任一变化、缺失 `environment.primary_sm_arch`、
  malformed SHA、non-finite number、manifest/inputs symlink。

```python
self.assertEqual(scope_key(manifest_a), scope_key(reordered_manifest_a))
self.assertNotEqual(scope_key(manifest_a), scope_key(same_name_new_bytes))
```

scope document 必须包含 manifest schema/input hash、backend、primary SM arch、
canonical dims、ptr size、baseline/ref SHA-256，以及完整 workload identity 或
`{"mode": "kernel-only"}`。

- [ ] **步骤 2：运行 RED。**

```bash
python3 -m unittest tests.test_strategy_memory -v
```

预期：新模块缺失。

- [ ] **步骤 3：实现 strict JSON 与 scope。** 使用 duplicate-key 和
`parse_constant` 拒绝宽松 JSON；调用 artifact reader 验证输入当前 bytes 与
manifest SHA 一致；full workload 要求 `source_hash`、objective、cases、kind；
scope 以 sort-keys、compact separators、`allow_nan=False` canonical JSON 哈希。

- [ ] **步骤 4：写 memory store RED 测试。** 覆盖：新文件 mode `0600`、邻接
`.lock`、既有 memory/lock symlink、父目录 symlink、非法 schema、unknown
field、non-finite、scope key 与 scope document 不匹配、更新中异常保持旧
store 完整、memory path 被替换时 fail closed。

- [ ] **步骤 5：实现 `_locked_memory_update`。** 锁与 memory 都从稳定的
no-follow parent directory fd 打开；Unix 使用 `fcntl.flock(LOCK_EX)`；在锁内
read/validate/update；唯一 temp regular file 用 `0600`，写完 `fsync`，
`os.replace` 后 directory `fsync`；finally 清理 temp 并 unlock。不得静默修复
损坏 store。

- [ ] **步骤 6：写并发与容量 RED 测试。** 使用 multiprocessing 同时添加两条
不同 record，最终两条都存在；相同 identity 并发只保留一条；第 257 个 scope
与一个 scope 的第 129 条 unique run 都拒绝，已有记录不被逐出。

- [ ] **步骤 7：实现去重与容量。** 去重 key 固定为
`input_hash/candidate_sha256/decision_sha256/checkpoint_identity` canonical hash；
达到容量时允许 exact duplicate no-op，拒绝新 unique entry。

- [ ] **步骤 8：运行 GREEN 与提交。**

```bash
python3 -m unittest tests.test_strategy_memory -v
python3 -m unittest tests.test_artifact_store -v
python3 -m py_compile skills/cuda-kernel-optimizer/scripts/strategy_memory.py
git diff --check
git add tests/test_strategy_memory.py \
  skills/cuda-kernel-optimizer/scripts/strategy_memory.py
git commit -m "feat: add workload scoped strategy memory store"
```

### 任务 4：只从完成且重新验证的 v2.2 run 采集记录

**文件：**

- 修改：`tests/test_strategy_memory.py`
- 修改：`skills/cuda-kernel-optimizer/scripts/strategy_memory.py`

- [ ] **步骤 1：建立真实 v2.2 run fixture。** 不手写宽松的 terminal shortcut；
  复用 `tests/test_state_schema.py` 的 artifact 形状，生成 manifest、state、
  checkpoint、decision、candidate、kernel/workload paired JSONL、methods 和可选
  attribution/SASS。fixture 要能分别产生 `kernel_only_win`、`end_to_end_win`、
  `confirmed_loss` 与 `inconclusive`。

- [ ] **步骤 2：写完成状态 RED 测试。** `record` 必须调用并通过：

  - `orchestrate._load_and_verify_manifest(run_dir)`；
  - `state.validate_state(state)`，它会重新读取 paired JSONL 并重算统计；
  - `orchestrate._validate_checkpoint(checkpoint, input_hash=...)`；
  - `orchestrate._verify_state_candidates(state)`；
  - checkpoint `stage/status == complete`；
  - state/run/manifest/terminal/candidate/checkpoint identity 全部一致。

任一 artifact 缺失、symlink、legacy schema、candidate 或 decision bytes drift、
paired sample 被改、raw statistic contradiction、checkpoint 未 complete 都 hard
fail，且 memory/out 均不改变。

- [ ] **步骤 3：写 decision 重放 RED 测试。** 从 terminal 绑定的
`decision.json.evidence` 调用 `decision.decide(...)`，要求重放结果与原 decision
在 status、mode、reason、statistics、workload fields、constraints、Pareto 上
strict 相等。缺少 evidence 的旧式 decision 可以继续作为 optimizer 自身的
历史 artifact，但不能进入 strategy memory。

- [ ] **步骤 4：实现 `load_completed_run(run_dir)`。** 所有读操作使用既有安全
reader；先 capture run root identity，验证完成后再次验证 root identity 和
关键 artifact SHA。返回 detached strict JSON record，不暴露任意环境变量或
源码内容，只保存身份、状态、统计、method IDs 和 evidence paths/hashes。

- [ ] **步骤 5：写 attribution 边界 RED 测试。** 只在 terminal iteration 的
`attribution.json` 同时满足以下条件时写 method performance evidence：method 在
该 iteration 的 `methods.json`；真实 ablated kernel 与 bench 位于该 run 的
`itervN/ablations/<method>/`；均为 non-symlink regular file；bench correctness
literal true；champion/ablated finite positive ms 与文件内容重算一致；
attribution 差值和百分比重算一致。`no_ablated_kernel`、ablated correctness
failure、缺 hash binding、tampered number 均不得生成 positive/negative，最多
记录 `unavailable`/`implementation_dependency`。

- [ ] **步骤 6：实现 bundle、method 和 SASS 分层。** terminal decision 产生
bundle outcome；合法 ablation 才产生 method performance outcome，并明确标注
`evidence_quality: diagnostic_unpaired_ablation`，不能成为 promotion；SASS
check 只写 `implementation_status`，绝不映射为 performance positive。没有合法
ablation 时 method performance map 为空。

- [ ] **步骤 7：实现 record CLI 的 write ordering。** 先完全构造/验证 record，
再在 memory lock 内去重并写 memory；成功后原子写 `--out`。如果 output 写失败，
CLI 报错但 memory 中的 deduplicated record 保持可重试；再次调用必须 no-op
dedupe，不能复制记录。

- [ ] **步骤 8：运行 focused GREEN 与提交。**

```bash
python3 -m unittest tests.test_strategy_memory -v
python3 -m unittest tests.test_state_schema tests.test_decision \
  tests.test_orchestrate tests.test_paired_stats -v
git diff --check
git add tests/test_strategy_memory.py \
  skills/cuda-kernel-optimizer/scripts/strategy_memory.py
git commit -m "feat: record verified v2.2 optimization evidence"
```

### 任务 5：生成可追溯但不干预 promotion 的建议

**文件：**

- 修改：`tests/test_strategy_memory.py`
- 修改：`skills/cuda-kernel-optimizer/scripts/strategy_memory.py`

- [ ] **步骤 1：写 suggestion RED 测试。** 同 scope 下构造 repeated positive、
negative、inconclusive、conflicting 和 bundle-only records；另构造其他 scope。
断言 suggestion 只使用 exact scope：

  - `preferred_method_ids` 仅来自至少两次、无反例的合法 positive ablation；
  - `caution_method_ids` 包含合法 negative 或 conflicting method；
  - inconclusive 不升级为 negative；
  - bundle-only 只进入 `prior_bundles`；
  - 每一项都有 record identity、decision/ablation evidence link/hash 和 count。

- [ ] **步骤 2：写只读/advisory RED 测试。** 调用前后对 manifest、state、
  checkpoint、decision、candidate 做 SHA-256 快照；调用后完全一致。输出必须
  带 `advisory: true` 和固定声明：不能删 branch、不能覆盖 profiler/budget/
  correctness/sanitizer/paired/workload/decision gate。

- [ ] **步骤 3：实现 deterministic suggestion。** 所有 ID 排序稳定；prior
  bundle 按完成时间和 identity 稳定排序；不读取当前 state，不解析源码，不
  生成新的方法，不调用 orchestrator。memory 不存在时返回
  `status: unavailable` 的 advisory output 并以非零退出，调用方可继续无 memory
  优化。

- [ ] **步骤 4：实现 CLI parser。** 固定支持且只支持：

```bash
python3 scripts/strategy_memory.py suggest \
  --memory MEMORY.json --manifest RUN/manifest.json --out SUGGESTION.json
python3 scripts/strategy_memory.py record \
  --memory MEMORY.json --run-dir RUN --out RECORD.json
```

所有路径必填，不提供 default memory；`--help` 不暗示 orchestrator integration。

- [ ] **步骤 5：运行 GREEN 与提交。**

```bash
python3 -m unittest tests.test_strategy_memory -v
python3 skills/cuda-kernel-optimizer/scripts/strategy_memory.py --help
python3 skills/cuda-kernel-optimizer/scripts/strategy_memory.py suggest --help
python3 skills/cuda-kernel-optimizer/scripts/strategy_memory.py record --help
git diff --check
git add tests/test_strategy_memory.py \
  skills/cuda-kernel-optimizer/scripts/strategy_memory.py
git commit -m "feat: suggest advisory cuda optimization strategies"
```

### 任务 6：补充 serving 与 systems/IR 证据 reference

**文件：**

- 新建：`skills/cuda-kernel-optimizer/references/serving_evidence_protocol.md`
- 新建：`skills/cuda-kernel-optimizer/references/systems_and_ir_coverage.md`
- 修改：`tests/test_skill_metadata.py`

- [ ] **步骤 1：写文档契约 RED 测试。** 要求 serving reference 同时出现四层
  claim ladder、`kernel_only_win`、`end_to_end_win`、user-provided workload、
  paired A/B、clean window、shared-host contamination、raw request/environment
  evidence；systems reference 覆盖 copies/allocation/sync/graphs/launch density、
  CUTLASS/CuTe dispatch/layout/epilogue/cluster/arch、Triton autotune 与
  TTIR/TTGIR/LLVM/PTX/cache/generated code，并链接三个既有/新增 reference。

- [ ] **步骤 2：运行 RED。**

```bash
python3 -m unittest tests.test_skill_metadata -v
```

预期：两个文件缺失。

- [ ] **步骤 3：写最小 reference。** 使用证据表和决策清单，避免复制
  `optimization_catalog.md` 的 API 大表。明确 generated code 只能证明机制已
  emit，operator timing 不能直接推出 serving 收益；稀疏/变长/fused/serving
  path 必须匹配真实请求分布。

- [ ] **步骤 4：运行 GREEN 与链接检查。**

```bash
python3 -m unittest tests.test_skill_metadata -v
rg -n "optimization_catalog|compatibility|serving_evidence_protocol" \
  skills/cuda-kernel-optimizer/references/systems_and_ir_coverage.md
git diff --check
```

- [ ] **步骤 5：提交。**

```bash
git add tests/test_skill_metadata.py \
  skills/cuda-kernel-optimizer/references/serving_evidence_protocol.md \
  skills/cuda-kernel-optimizer/references/systems_and_ir_coverage.md
git commit -m "docs: add serving and ir evidence protocols"
```

### 任务 7：接入 skill 与 agent 发现入口

**文件：**

- 修改：`skills/cuda-kernel-optimizer/SKILL.md`
- 修改：`skills/cuda-kernel-optimizer/agents/openai.yaml`
- 修改：`tests/test_skill_metadata.py`

- [ ] **步骤 1：写发现性 RED 测试。** `SKILL.md` 必须出现两个 CLI 的准确命令
  和两个 reference；仍不超过 500 行；必须同时出现 `advisory`、
  `decision.json` owns promotion、explicit `--memory`。agent default prompt 需要
  能触发“分析既有 ncu report”以及“runtime/serving evidence”，但继续要求
  user-provided workload 才能作端到端结论。

- [ ] **步骤 2：运行 RED。**

```bash
python3 -m unittest tests.test_skill_metadata -v
```

预期：新增 analyzer、memory 和 reference 的发现性断言失败；原有 v2.2 约束
继续通过。

- [ ] **步骤 3：最小修改 skill 文档。** 在 profiler 入口旁放 standalone
  analyzer；在 finalize 后放 opt-in record/suggest；systems/IR 或 serving 任务
  才指向新 reference。参数细节交给 `--help`，不把 `SKILL.md` 写成长 README。

- [ ] **步骤 4：运行 GREEN。**

```bash
python3 -m unittest tests.test_skill_metadata -v
python3 skills/cuda-kernel-optimizer/scripts/analyze_ncu_rep.py --help
python3 skills/cuda-kernel-optimizer/scripts/strategy_memory.py --help
git diff --check
```

- [ ] **步骤 5：提交。**

```bash
git add skills/cuda-kernel-optimizer/SKILL.md \
  skills/cuda-kernel-optimizer/agents/openai.yaml tests/test_skill_metadata.py
git commit -m "docs: expose report analysis and advisory memory workflows"
```

### 任务 8：完整 CPU 回归与安全反向审查

**文件：**

- 必要时仅修改本计划已列出的实现或测试文件。

- [ ] **步骤 1：完整测试。** 从干净 shell 执行并保存最新输出：

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

预期：至少原有 514 项加新增测试全部通过；原有 4 个环境相关 skip 可接受，
不得新增非明确平台原因的 skip。

- [ ] **步骤 2：验证所有脚本编译和 help。**

```bash
python3 -m py_compile skills/cuda-kernel-optimizer/scripts/*.py
for script in skills/cuda-kernel-optimizer/scripts/*.py; do
  python3 "$script" --help >/dev/null
done
```

如果某个既有脚本没有普通 `--help` 约定，只记录该既有例外；新增两个脚本必须
返回 0。

- [ ] **步骤 3：运行 skill validator 与仓库卫生检查。**

```bash
python3 /Users/tcheng/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  skills/cuda-kernel-optimizer
git diff --check
git status --short
git log --oneline --decorate -12
```

- [ ] **步骤 4：安全反向审查。** 逐项确认：无 `shell=True`；无默认 memory；
  无 orchestrator import/call strategy memory；无 `chmod` driver/counter；无
  unbounded diagnostic；无 analysis self-hash；无 SASS→performance 推断；无
  memory→promotion 路径；无 upstream remote 更改。

```bash
rg -n "shell=True|default.*memory|strategy_memory|chmod|counter_access" \
  skills/cuda-kernel-optimizer/scripts skills/cuda-kernel-optimizer/SKILL.md
git remote -v
git config --get remote.upstream.pushurl || true
```

- [ ] **步骤 5：如修复回归，遵循单独 RED/GREEN 并提交。** 不把多个无关修复
  挤进一个 commit。最后一次 CPU suite 必须在所有修复之后重新运行。

### 任务 9：RTX 5090 上只读验证真实 `.ncu-rep`

**文件：**

- 不修改仓库源码；远端只生成隔离证据目录。

- [ ] **步骤 1：执行前检查远端边界。** 确认目标为既有 5090 host，识别一个
  已有 `.ncu-rep`，记录其 SHA-256；创建新的 isolated root，例如
  `/data/tcheng/cuda-skill-e2e/v2.2-legacy-coverage-port/`。不得直接在原 report
  目录运行 analyzer。

- [ ] **步骤 2：复制而不是移动 report。** 复制后比较原件和副本 SHA-256；
  仅将当前 branch 的 skill 复制到 isolated root，不执行安装覆盖。

- [ ] **步骤 3：运行真实 analyzer。** 显式传入副本、独立 output、真实 NCU
  binary、合理 timeout。保存 stdout/stderr/exit code。接受 success 或因 report
  page 覆盖不足产生的 documented partial；不接受 timeout、identity drift、
  malformed JSON、missing marker 或 artifact hash mismatch。

```bash
python3 scripts/analyze_ncu_rep.py copied-report.ncu-rep \
  --out-dir analysis --ncu-bin /path/to/ncu --ncu-num 5 --timeout 120
```

- [ ] **步骤 4：验证只读事实。** 原 report SHA-256 未变；无 GPU workload 被
  launch；`analysis.json.counter_access == not_probed`；supporting hashes 全部
  重算一致；Markdown 可安全渲染；原 NCU/driver/counter 配置未改变。

- [ ] **步骤 5：strategy memory 只做 CPU fixture 验收。** 不把用户真实 run
  写进长期 memory；在远端 isolated root 的复制 fixture 上 record 两次验证
  dedupe，再 suggest，最后对 run 目录做前后 hash 清单比较，证明 advisory
  工具未修改 run。

### 任务 10：按最终实现重写英文与中文 README

**文件：**

- 修改：`README.md`
- 修改：`README.zh-CN.md`
- 修改：`tests/test_readme_sync.py`

- [ ] **步骤 1：从最终证据冻结 README 事实表。** 先采集但不改文档：两个新增
  CLI 的 `--help`、预算 preset、terminal status、当前 CPU test 总数和 skip、
  5090 analyzer exit status、NCU 版本、report/artifact SHA-256、
  `counter_access: not_probed`，以及既有 v2.2 真实 workload 结论。任何没有最新
  证据的数字不写入 README。

```bash
python3 skills/cuda-kernel-optimizer/scripts/analyze_ncu_rep.py --help
python3 skills/cuda-kernel-optimizer/scripts/strategy_memory.py --help
python3 skills/cuda-kernel-optimizer/scripts/orchestrate.py setup --help
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

- [ ] **步骤 2：写信息架构 RED 测试。** 在 `tests/test_readme_sync.py` 固定两个
  README 都使用以下语义顺序：项目定位 → 按任务开始 → 安装 → 5 分钟首跑 →
  可信晋级路径 → 各任务命令 → 独立工具边界 → 输入/预算/状态 → 产物/恢复 →
  兼容性/验证 → 参考/许可证。标题可以自然本地化，但索引顺序必须严格递增。

```python
def assert_in_order(testcase, text, markers):
    positions = [text.index(marker) for marker in markers]
    testcase.assertEqual(positions, sorted(positions))
```

- [ ] **步骤 3：写 Mermaid 拓扑 RED 测试。** 每份 README 必须恰好有两个
  `mermaid` block。节点 ID 和 edge operator 保持一致，label 可以本地化；第一
  张图只有 `decision.json` 能连到 promotion，compiler/SASS 只用虚线连到
  evidence；第二张图中 memory suggestion 只使用虚线且没有到 `decision.json`
  的边。

```python
EDGE = re.compile(
    r"^\s*([a-z][a-z0-9_]*)[^-\n]*?\s*(-->|-\.->)\s*"
    r"([a-z][a-z0-9_]*)",
    re.MULTILINE,
)

def mermaid_topology(block):
    return sorted(EDGE.findall(block))
```

- [ ] **步骤 4：写事实同步与文风 RED 测试。** 两份文档必须同时出现 analyzer
  的全部公开 options、record/suggest 的全部公开 options、四种用户任务、
  `balanced` default、`kernel_only_win`、`end_to_end_win`、
  `counter_access: not_probed`、两个新增 reference 和相同验证数字。禁止遗留
  planned/pending V2.1 文案、自动 memory、自动 serving claim，以及下列空泛
  词语：中文“旨在/赋能/无缝/强大/全面/通过……从而……”，英文
  “powerful/seamless/revolutionary/comprehensive”。

- [ ] **步骤 5：运行 RED。**

```bash
python3 -m unittest tests.test_readme_sync -v
```

预期：当前 release-first README 缺少 task-first 顺序、两个新 CLI、两张 Mermaid
图和新 reference；确认失败不来自既有预算或 workload 断言。

- [ ] **步骤 6：先独立重写英文 README。** 标题只写项目名，不把易过期版本号
  放进 H1。开头一句直接说明项目解决什么问题；紧接四个任务入口和可执行安装/
  首跑。用第一张 Mermaid 解释权威晋级路径，再给各任务的最短命令。保留简短
  artifact tree、预算表和可核验的 5090 摘要；长测试叙述放入 `<details>` 或
  链接到 `tests/gpu/sm120/README.md`。

- [ ] **步骤 7：基于同一事实独立写中文 README。** 不逐句翻译英文。保留
  CUDA、CUTLASS、Triton、kernel、workload、paired A/B、NCU、SASS、manifest、
  checkpoint 和 decision 等业界术语；中文使用短句、主动语态和自然标点。命令、
  默认值、状态语义、限制、hash 与英文完全一致。

- [ ] **步骤 8：加入两张同拓扑 Mermaid。** 第一张的实线主路径是 candidate →
  correctness → paired kernel evidence → sanitizer hard gate → optional
  workload → `decision.json` → conditional promotion；compiler/SASS 仅用虚线
  汇入 evidence side path。明确 full mode 的 `kernel_only_win` 不推进 global
  best。第二张展示 existing `.ncu-rep` → analysis bundle，以及 completed run →
  memory → advisory suggestion；不画 memory 到 `decision.json` 的边。

- [ ] **步骤 9：运行 GREEN 和逐行编辑审查。**

```bash
python3 -m unittest tests.test_readme_sync tests.test_skill_metadata -v
python3 skills/cuda-kernel-optimizer/scripts/analyze_ncu_rep.py --help
python3 skills/cuda-kernel-optimizer/scripts/strategy_memory.py --help
git diff --check
```

逐段检查：每段只讲一个意思；命令先于长解释；删除重复；没有营销语；中文不是
英文句法换词；英文没有中文式省略；术语、数字、链接和结论逐项对照事实表。

- [ ] **步骤 10：提交 README。**

```bash
git add README.md README.zh-CN.md tests/test_readme_sync.py
git commit -m "docs: rewrite task first bilingual project guide"
```

### 任务 11：最终证据、版本建议与集成确认点

**文件：**

- 必要时修改：本计划已列出的测试、实现或 README 文件（只修复实测问题）

- [ ] **步骤 1：若 5090 暴露问题，先本地添加 RED 测试，再最小修复并重复任务
  8、9、10。** 不把机器特例写成通用成功判断。

- [ ] **步骤 2：做最后一次 fresh verification。**

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 /Users/tcheng/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  skills/cuda-kernel-optimizer
python3 -m py_compile skills/cuda-kernel-optimizer/scripts/*.py
git diff --check
git status --short
```

- [ ] **步骤 3：检查分支相对 main 的确切内容。**

```bash
git log --oneline main..HEAD
git diff --stat main...HEAD
git diff --name-status main...HEAD
git status --short
```

- [ ] **步骤 4：交付给用户审查。** 汇报：新增测试/总测试数、5090 analyzer
  exit status 与 report/artifact hashes、README 结构和 Mermaid 数、已知限制、
  commit 列表、未 push/未 merge/未改 About 的事实。建议版本号为 `v2.3` 还是
  保持 v2.2 patch，由用户确认；未经确认不改 tag、不 merge `main`、不 push
  fork、不更新 GitHub About。

### 任务 12：获批后合并 fork、推送并更新 GitHub About

**文件：**

- 不再修改源码或文档；只执行获批的 Git/GitHub 集成操作。

- [ ] **步骤 1：重新验证目标与权限边界。** 必须同时满足：主 checkout 干净、
  `main == origin/main`、当前 feature branch 是预期 HEAD、origin 是用户 fork、
  upstream push URL 为 `DISABLED`、GitHub 仓库是 fork 且 parent 精确匹配。

```bash
git -C /Users/tcheng/Documents/Codex/2026-07-15-triton-skill/cuda-optimized-skill status --short
git fetch origin
git rev-parse main
git rev-parse origin/main
git remote -v
git config --get remote.upstream.pushurl
gh api repos/troycheng/cuda-optimized-skill \
  --jq '{full_name, fork, parent: .parent.full_name, default_branch}'
```

预期：`full_name == troycheng/cuda-optimized-skill`、`fork == true`、
`parent == KernelFlow-ops/cuda-optimized-skill`、`default_branch == main`。任一
不符立即停止，不执行 merge、push 或 About mutation。

- [ ] **步骤 2：在用户明确批准的版本策略下 fast-forward main。** 如果
  `origin/main` 已前进，先停下重新审查差异；禁止强推。

```bash
git -C /Users/tcheng/Documents/Codex/2026-07-15-triton-skill/cuda-optimized-skill \
  merge --ff-only agent/legacy-coverage-v2-2
git -C /Users/tcheng/Documents/Codex/2026-07-15-triton-skill/cuda-optimized-skill \
  push origin main
```

- [ ] **步骤 3：读回 fork main。** 本地 main、origin/main 和 GitHub API 的
  default branch commit 必须等于 feature HEAD，再允许更新 About。

```bash
git rev-parse agent/legacy-coverage-v2-2
git rev-parse main
git rev-parse origin/main
gh api repos/troycheng/cuda-optimized-skill/commits/main --jq '.sha'
```

- [ ] **步骤 4：更新双语 About description 和精确 topics。** 不触碰 upstream，
  homepage 保持空字符串。

```bash
ABOUT='Evidence-driven CUDA, CUTLASS and Triton kernel optimization with paired benchmarks, real-workload validation and NCU analysis. / 用成对基准、真实负载验证与 NCU 分析优化 CUDA、CUTLASS 和 Triton kernel。'
gh api --method PATCH repos/troycheng/cuda-optimized-skill \
  -f description="$ABOUT" -f homepage=''
gh api --method PUT repos/troycheng/cuda-optimized-skill/topics \
  -H 'Accept: application/vnd.github+json' \
  -f 'names[]=cuda' -f 'names[]=triton' -f 'names[]=cutlass' \
  -f 'names[]=gpu' -f 'names[]=kernel-optimization' \
  -f 'names[]=nsight-compute' -f 'names[]=performance' \
  -f 'names[]=codex-skills'
```

- [ ] **步骤 5：读回并逐字段验证。** description 必须逐字匹配；homepage 必须
  为空；topics 集合必须恰好是 8 项；fork/parent/default branch 不变。然后再次
  确认 upstream 无 push 能力。

```bash
gh api repos/troycheng/cuda-optimized-skill \
  --jq '{full_name, fork, parent: .parent.full_name, default_branch, description, homepage}'
gh api repos/troycheng/cuda-optimized-skill/topics --jq '.names | sort'
git config --get remote.upstream.pushurl
git status --short
```

- [ ] **步骤 6：最终交付。** 报告 fork main commit、GitHub API read-back、
  topics、README/测试结果和 installed skill 是否同步；明确 upstream 未修改。

## 完成判据

- 新增两个 CLI 可通过 `--help` 发现且全路径显式。
- Analyzer 对 symlink/identity drift/timeout/输出爆量 fail closed；成功或 partial
  输出均能由 marker 和 hashes 独立验证。
- Strategy recorder 拒绝未完成、旧 schema、tampered 或统计矛盾 run；并发与
  容量边界有测试。
- Strategy suggestion 不修改 run，不影响 branch/budget/gate/promotion；无合法
  ablation 时不伪造 method performance。
- SASS 只表达 implementation status。
- 两份 reference 与四个任务入口可从 `SKILL.md` 和中英文 README 找到。
- 两份 README 使用任务优先顺序，各包含恰好两张同拓扑 Mermaid；命令、事实、
  数字和边界一致，中文自然、英文直接，没有翻译腔或营销式套话。
- 完整 CPU suite、skill validator、py_compile、CLI help、diff check 全绿。
- RTX 5090 只读 report 验证完成，未修改权限、驱动或用户源码。
- 用户批准前分支保持本地；批准后仅 fast-forward 并推送用户 fork。
- GitHub About 只在 fork main 推送后更新，description/topics/homepage/fork/parent
  已读回验证，upstream 未修改。
