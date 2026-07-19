# CUDA Kernel and Workload Optimizer 3.0 开发计划

状态：执行中，2026-07-19

本计划以 `v2.9.0`（`8aa4a50`）为对照基线。基线测试为 813 项通过、5 项因缺少物理 GPU 跳过。开发在 `codex/v3` 隔离 worktree 中进行，不向原始上游仓库推送。

## 开发原则

- 每项行为变更先写失败测试，再写最小实现；
- 优先扩展现有 `artifact_store.py`、`workload_controller.py`、`iteration_guard.py`、`budget.py` 和 `knowledge_query.py`，不另造重叠框架；
- 每个阶段都能单独回退，文档不能先于可执行行为；
- 宿主机保持 `recommend_only`；
- 外部模型只参与设计质证和候选建议，测试和本地证据拥有最终裁决权。

## 阶段 0：固定评测与运行合同

目标：先保证后续开发能被 2.9 对照和反证，而不是最后才挑有利案例。

### 0.1 Workload Contract

新增：

- `skills/cuda-kernel-optimizer/templates/workload_contract.schema.json`
- `skills/cuda-kernel-optimizer/scripts/workload_contract.py`
- `tests/test_workload_contract.py`

实现：严格 JSON、重复键拒绝、规范化摘要、文件身份、目标、约束、预算、修改根、证据时效和宿主机策略冻结。合同变化必须生成新 run，不能覆盖旧合同。

测试：缺字段、未知字段、软链接、身份变化、目标漂移、非有限数值和 `host_policy != recommend_only` 全部拒绝；同一输入生成稳定摘要。

### 0.2 评测清单和 2.9 基线

新增：

- `tests/evals/v3/scenarios.json`
- `tests/evals/v3/README.md`
- `tools/run_skill_eval.py`
- `tests/test_skill_eval.py`

首批 fixture：错误 kernel 瓶颈、边界正确性、无 NCU 权限、噪声升高、中断恢复、证据过期、完整 workload 回归。Runner 输出统一 JSON，不把模型主观评分当主要指标。

实验矩阵固定为 `no_skill`、`v2.9`、`v3_random_planner`、
`v3_shuffled_registry` 和 `v3_full`。每次运行绑定模型、prompt、skill、合同、
环境、种子和重复编号；required event 必须由账本和带哈希产物派生，不能由模型
自报。另设一个组合故障注入长跑场景，覆盖预算单调性、旧证据、改名重试和新
合同必须新建 run。

先记录 2.9 的完成率、错误方向实验数、耗时、候选数和证据违规，再冻结 3.0 发布门槛。

验证：

```bash
python3 -m unittest tests.test_workload_contract tests.test_skill_eval -v
python3 tools/run_skill_eval.py --suite tests/evals/v3/scenarios.json --mode v2.9
```

## 阶段 1：确定性长期控制和证据账本

目标：AI 可以持续提出候选，但不能改变运行规则，也不能在重启后忘记失败。

新增或扩展：

- `skills/cuda-kernel-optimizer/scripts/run_control.py`
- `skills/cuda-kernel-optimizer/scripts/evidence_ledger.py`
- `skills/cuda-kernel-optimizer/templates/candidate_proposal.schema.json`
- `skills/cuda-kernel-optimizer/templates/run_event.schema.json`
- `tests/test_run_control.py`
- `tests/test_evidence_ledger.py`

实现顺序：

1. 写状态迁移失败测试；
2. 实现纯函数状态机和非法迁移拒绝；
3. 写候选预登记、预算耗尽、证据过期和合同漂移测试；
4. 实现 `PASS/KILL/INCONCLUSIVE/DEFERRED` 生命周期；
5. 写账本追加、哈希链、截断、覆盖、并发和恢复测试；
6. 复用安全原子写入，以 create-once 事件文件加链式摘要形成账本；
7. 集成 baseline/champion 定期重放和停止快照。

在阶段 3 的证据适配器接通前，Controller 对 `PASS` 保持 fail-closed；两个
调用方布尔值不能生成 champion。

验证：

```bash
python3 -m unittest tests.test_run_control tests.test_evidence_ledger -v
python3 -m unittest tests.test_state_schema tests.test_iteration_guard tests.test_artifact_store -v
```

## 阶段 2：能力注册表和按需 playbook

目标：把现有 63 项方法索引升级为“信号命中后只加载少量可执行知识”，同时控制上下文。

新增：

- `skills/cuda-kernel-optimizer/references/capabilities/registry.json`
- `skills/cuda-kernel-optimizer/references/capabilities/sources.json`
- `skills/cuda-kernel-optimizer/references/capabilities/*.md`
- `skills/cuda-kernel-optimizer/templates/capability.schema.json`
- `skills/cuda-kernel-optimizer/scripts/capability_query.py`
- `tests/test_capability_query.py`

首批六个 playbook 对应架构文档中的六类场景。每个 playbook 控制篇幅，只包含适用信号、反例、步骤、验证、停止条件和来源；完整参考资料继续放在 `references/`，不复制进 `SKILL.md`。

必须测试：精确架构、版本范围、完整信号组、反向信号、冲突能力、最大返回数、UTF-8 字节硬预算、成本错报、过期与未来知识降级、来源完整性、单次快照和各级目录符号链接逃逸。加入 `real` 与 `shuffled` registry 对照。检索层没有执行或晋级权，只返回分阶段的封闭 `gate_requirements` 供阶段 3 的 Controller 解析。

验证：

```bash
python3 -m unittest tests.test_capability_query tests.test_knowledge_query -v
python3 skills/cuda-kernel-optimizer/scripts/self_check.py
```

## 阶段 3：统一观察摘要和 Planner 边界

目标：让 Planner 看到足够做决定的事实，不让原始日志挤满上下文，也不让它越过晋级门禁。

新增或扩展：

- `skills/cuda-kernel-optimizer/scripts/evidence_summary.py`
- `skills/cuda-kernel-optimizer/templates/observation_summary.schema.json`
- `skills/cuda-kernel-optimizer/scripts/workload_controller.py`
- `skills/cuda-kernel-optimizer/scripts/analyze_ncu_rep.py`
- `skills/cuda-kernel-optimizer/scripts/compiler_evidence.py`
- `tests/test_evidence_summary.py`
- `tests/test_planner_boundary.py`

先接入现有 benchmark、PyTorch profiler、Nsys/NCU、compiler 和 serving 产物，不一次性重写采集器。摘要只能引用已封存产物；每个结论带来源摘要、时间、层级和新鲜度。Planner 输出严格候选 schema，任何预算、合同、账本或晋级字段都视为越权并拒绝。

本阶段还要把能力查询返回的证据类别和 `gate_requirements` 解析为合同绑定的产物
引用。`target_compile_probe`、autotune/dispatch identity、正确性、配对测量和
workload replay 均由 Controller 校验；不能把字符串类别或知识卡自身状态当作已
通过门禁。

当前已完成开发切片：能力查询结果可重放校验；观察摘要有数量和精确 UTF-8 字节
硬上限；六类门禁产物使用封闭格式并绑定合同、环境、源码/候选/架构身份；门禁在
Controller 当前时间重算新鲜度；候选 schema 引用观察摘要和能力查询摘要。Controller
只运行 allowlist 中哈希捕获的 adapter 快照，从无 producer/status/time 的原始测量
重算 PASS；封印绑定 run、账本和 adapter 实现摘要，密钥不传入 adapter。Nsys 与
PyTorch 诊断观察已经进入同一封存链；Planner 只能从当前目标的诊断观察推导信号和
证据类别，再重放能力查询。候选的内存登记、持久化登记和 run replay 都要求带
Controller HMAC 的 admission，登记时间必须等于摘要时间，旧公开入口不能再接受
调用者自报的 evidence age。

本阶段剩余：

1. 接通现有 benchmark、PyTorch、Nsys、NCU、compiler 和 serving adapter，不用测试
   fixture 代替真实产物；
2. 为高保证部署提供独立账号、容器或密钥服务配置；默认本地模式明确不抵抗同账号
   主机攻击者。该配置不阻塞默认本地 Controller，但发布文档必须明确边界。

验证：

```bash
python3 -m unittest tests.test_evidence_summary tests.test_planner_boundary -v
python3 -m unittest tests.test_workload_controller tests.test_analyze_ncu_rep tests.test_compiler_evidence -v
```

## 阶段 4：稳定性校准和反偏航闭环

目标：从当前环境估计噪声和最小可测效应，不依赖通用硬编码百分比。

扩展：

- `skills/cuda-kernel-optimizer/scripts/paired_stats.py`
- `skills/cuda-kernel-optimizer/scripts/experiment_design.py`
- `skills/cuda-kernel-optimizer/scripts/run_control.py`
- `tests/test_stability_calibration.py`
- `tests/test_long_run_recovery.py`

实现 `green/yellow/red` 状态、配对重放、MDE/MPE 计算、证据时效、周期审计、漂移后重冻结、预算断路器和跨进程恢复。所有时间与次数由合同或校准结果决定，不把开发样机数值写成通用规则。

验证包括：杀进程后恢复、修改 workload 后拒绝旧基线、账本尾部损坏、持续噪声、候选结论不足、重复失败机制和预算耗尽。

## 阶段 5：完整 workload 与 RTX 5090 实测

目标：证明系统能发现瓶颈不在 kernel，也能在 kernel 优化后回到端到端 workload 验证。

物理测试范围：

- RTX 5090 / `sm_120` 精确架构；
- 一个 Triton kernel 正向案例；
- 一个故意把瓶颈放到 CPU、I/O 或框架调度的反向案例；
- 一个完整推理 workload 的局部优化与端到端回归；
- NCU 正常路径与 `ERR_NVGPUCTRPERM` 降级路径；
- 中断恢复和预算停止。

不修改宿主机驱动、计数器权限、频率、功耗和服务配置。需要修改时只生成建议。

验证：

```bash
python3 -m unittest tests.gpu.sm120.test_sm120_acceptance -v
python3 tools/run_skill_eval.py --suite tests/evals/v3/scenarios.json --mode v3 --target sm_120
```

## 阶段 6：文档、独立前向测试和发布

更新：

- `SKILL.md`：只保留路由、边界和主循环；
- `README.md`、`README.zh-CN.md`：讲清项目是什么、能做什么、需要用户提供什么、AI 如何执行、能证明到什么程度；
- `docs/`：按使用问题组织，不放开发计划；
- `maintainers/history/`：发布时归档本设计和计划；
- Release notes：从 2.9 到 3.0 的行为变化、兼容性和迁移说明。

独立前向测试必须让未参与实现的模型仅凭发布版 skill 完成新任务，并记录它是否读到正确资料、是否越权、是否在错误瓶颈上浪费预算。完成本地静态、CPU、GPU、文档和双远端 dry-run 验证后，才合并到主干。

发布只推送：

- 用户 fork：`troycheng/cuda-optimized-skill`；
- 已授权内网镜像：`git.yukework.com/mlsys/cuda-optimized-skill`。

原始 `KernelFlow-ops/cuda-optimized-skill` 保持 push disabled。发布后再同步本机安装的 skill，并复跑 self-check。

## 首个执行切片

本轮从阶段 0 开始，严格按以下顺序：

1. 为 Workload Contract 写失败测试；
2. 实现最小合同校验与冻结；
3. 为 eval manifest 和 deterministic runner 写失败测试；
4. 记录 2.9 对照数据；
5. 通过审查后进入长期控制器和证据账本。
