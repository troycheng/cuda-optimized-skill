# V3.1 主动诊断纵向切片实现计划

> 状态：执行中  
> 基线：`2d14688`  
> 范围：结构化执行路径、竞争假设、下一证据选择

## 1. 目标

这一切片解决 readiness 之后的第一个问题：AI 如何用尽量少的采集，先判断性能损失发生在
哪一层，再选择真正能区分竞争解释的下一条证据。

完成后，系统应能：

- 把一次全局扫描压缩成带来源、覆盖范围和缺口的执行图，而不是把 profiler 文本塞给模型；
- 维护有限、可反驳、绑定当前 epoch 的竞争假设；
- 从 Controller 固定的行动目录中选择下一条证据，不让模型自行声明成本、风险或“信息量”；
- 在没有可行区分证据时输出 `evidence_gap` 并停止，不重复 profile，也不假装已经找到根因。

本切片不实现知识卡、方向实验执行、候选代码生成、概率模型、跨 workload 迁移或自动修改
宿主机。它们分别属于后续纵向切片。

## 2. 已确认的设计决策

### 2.1 保留旧 execution-path 合同

现有 `execution_path.schema.json` 是 V2.5 的分支覆盖证明，不能改名复用。V3.1 新建
`execution_map.schema.json`，避免旧证据在升级后获得不同语义。

### 2.2 执行图必须表达“没有看到什么”

执行图记录 CPU、GPU、framework、transfer、communication、I/O、synchronization 和 idle
层，但每一层都有 `observed`、`not_observed` 或 `unavailable` 覆盖状态。缺失的 profiler
表、没有权限和采集窗口未覆盖不能被解释成零耗时。

节点记录 lane、持续时间、出现次数和证据引用；边只表达观察到的 `calls`、`waits_for`、
`transfers_to`、`synchronizes`、`precedes` 或 `unknown_dependency`。CPU 与 GPU 可重叠，
因此禁止把所有节点耗时相加后称为 workload wall time。

未解释空闲、未知依赖和未覆盖区间必须保留。只要存在这类节点，假设集合就必须包含一个
`unmodeled` 假设，防止 schema 之外的机制从诊断空间消失。

### 2.3 Epoch 由 Controller 拥有

每个 epoch 绑定：

- workload contract、environment、source 和 analysis policy digest；
- profiler 类型、版本、导出 schema 和 adapter implementation digest；
- shape distribution、dynamic branch 和 execution regime digest；
- 采集窗口及边界是否可能落在窗口内部。

模型不能创建、合并或回退 epoch。机制修改、身份变化、主要执行 regime 变化或 Controller
确认的瓶颈迁移创建新 epoch。可能跨边界的窗口只能保守归入新 epoch 或标成
`boundary_ambiguous`，不能继续支持高等级结论。旧证据只留作审计。

第一版不写死“连续 N 次、变化 X%”的自动退化阈值；这要等真实 workload 回放校准后再定。

### 2.4 假设使用证据等级，不使用概率

假设记录 scope、机制陈述、支持证据、反对证据、缺少的区分证据、反驳问题和关系。
关系只有：

- `exclusive`：两项不能同时解释同一 scope；
- `depends_on`：有方向、无环；
- `coexists_with`：两项可以同时成立。

`exclusive` 和 `coexists_with` 使用规范顺序的无向 pair；同一 pair 不能出现两种关系。
`depends_on` 必须是 DAG。孤立假设仍须有可执行的反驳问题，不能靠没有反对证据自动升级。

结论等级只有 `inconclusive`、`plausible`、`direction_supported`。单一 metric 最多达到
`plausible`；`direction_supported` 至少需要两类独立观察证据，或观察证据加一个后续切片
登记的单变量实验。本切片不会制造实验身份。

### 2.5 模型不控制排序输入

AI 提出的 evidence request 只包含：问题、目标假设、行动目录 ID 和不同结果会怎样改变假设。
行动的采集类型、required capability、成本、扰动、风险和预算适配来自 Controller 冻结目录。

Controller 先拒绝 identity/epoch 不匹配、能力不可用、预算不适配、结果不能改变假设状态、
或当前 epoch 已执行过等价签名的请求，再按以下顺序稳定排序：

1. 能区分的 `exclusive` pair 数量；
2. 能反驳的孤立或 `unmodeled` 假设数量；
3. 能增加的独立 evidence kind；
4. 更低 perturbation、risk 和 cost；
5. `request_id` 字典序。

不存在可行请求时返回 `evidence_gap`，列出缺少的 capability 或授权，但不修改宿主机，
不消耗 profile 次数，也不换一个名称重试同类采集。

## 3. 外部资料和质证如何影响实现

NVIDIA 最新 Nsight Systems 文档表明，SQLite 导出 schema 会变化、表按采集内容惰性创建，
统计报告中的百分比也不等于应用 wall time。因此 execution map 必须绑定导出 schema 和
adapter digest，并把缺表视为覆盖状态，而不是零值；不能用 `cuda_*_sum` 的百分比直接拼出
关键路径。

PyTorch profiler 的 Execution Trace 提供图结构表示，但官方不保证相关实验接口向后兼容。
因此 PyTorch graph 只能作为一种有版本的输入来源，不能成为 V3.1 自定义合同的隐式 schema。

DeepSeek 的敌意评审指出三类死角：未建模 idle 归因逃逸、渐进退化跨 epoch、无权限时的
不可区分假设循环。采纳显式 `unmodeled`、`boundary_ambiguous`、请求签名去重和
`evidence_gap`；拒绝未经真实回放校准的固定退化阈值、动态节点插件和 epoch 合并。

## 4. 实现任务

### 任务 1：冻结 Epoch 与 execution map 合同

**文件：**

- 新建：`skills/cuda-kernel-optimizer/templates/analysis_epoch.schema.json`
- 新建：`skills/cuda-kernel-optimizer/templates/execution_map.schema.json`
- 新建：`skills/cuda-kernel-optimizer/scripts/analysis_epoch.py`
- 新建：`skills/cuda-kernel-optimizer/scripts/execution_map.py`
- 新建：`tests/test_analysis_epoch.py`
- 新建：`tests/test_execution_map.py`

- [x] 先写失败测试：closed schema、identity mismatch、source schema 缺失、覆盖缺口误作零值、
  跨 epoch evidence、未知边类型、CPU/GPU 重叠、unmodeled gap。
- [x] 实现 strict JSON、规范 digest、epoch admission 和 compact map validation。
- [x] 证明旧 V2.5 `execution_path.schema.json` 未改变。
- [x] 提交：`feat(v3.1): define epoch-bound execution maps`

### 任务 2：实现竞争假设合同

**文件：**

- 新建：`skills/cuda-kernel-optimizer/templates/hypothesis_set.schema.json`
- 新建：`skills/cuda-kernel-optimizer/scripts/hypothesis_space.py`
- 新建：`tests/test_hypothesis_space.py`

- [x] 先写失败测试：伪造 evidence ID、旧 epoch 引用、关系冲突、depends cycle、单 metric
  升级、遗漏 unmodeled、无反驳问题、重复 hypothesis。
- [x] Controller 使用 evidence catalog 重放引用和 evidence kind，不信任模型摘要。
- [x] 输出有限、规范排序的 active set 和稳定 digest。
- [x] 提交：`feat(v3.1): admit competing diagnosis hypotheses`

### 任务 3：实现确定性下一证据选择

**文件：**

- 新建：`skills/cuda-kernel-optimizer/templates/evidence_request.schema.json`
- 新建：`skills/cuda-kernel-optimizer/templates/evidence_selection.schema.json`
- 新建：`skills/cuda-kernel-optimizer/references/evidence_action_catalog.json`
- 新建：`skills/cuda-kernel-optimizer/scripts/evidence_selector.py`
- 新建：`tests/test_evidence_selector.py`

- [ ] 先写失败测试：模型自报低成本、不可改变假设的请求、越权 capability、预算不适配、
  改名重采、排序输入篡改、并列非确定、无可行区分项。
- [ ] Controller 重放行动目录，生成等价请求签名并检查 epoch 历史。
- [ ] 返回 `selected`、`sufficient` 或 `evidence_gap`；不输出概率或 entropy。
- [ ] 提交：`feat(v3.1): select discriminating evidence deterministically`

### 任务 4：接入 Controller 的 AI 往返边界

**文件：**

- 修改：`skills/cuda-kernel-optimizer/scripts/workload_controller.py`
- 修改：`tests/test_workload_controller.py`
- 新建：`tests/fixtures/active_diagnosis/`

- [ ] 新路径只在 V3.1 analysis contract 明确启用时生效，旧 control-v1/v2 行为零差异。
- [ ] 全局扫描后写出 hash-bound `diagnosis_context.json`，状态进入
  `next_action=propose_hypotheses`。
- [ ] AI 返回假设与 evidence request 后，Controller 重放校验并写入追加证据链。
- [ ] resume 不重复全局扫描；identity 或 epoch 变化时旧 proposal fail closed。
- [ ] 提交：`feat(v3.1): route controller through active diagnosis`

### 任务 5：本地纵向和故障注入

- [ ] CPU fixture 覆盖 kernel hot path、framework gap、transfer overlap、unknown idle 和 mixed
  bottleneck。
- [ ] 做 analysis-only 消融：移除 execution map、移除关系、移除请求去重，确认错误方向或
  重复 profile 能被测试观测。
- [ ] 记录首个有效方向前的 profile 次数、重复采集数和上下文字节数。
- [ ] 全量测试、自检、`git diff --check`。
- [ ] 提交：`test(v3.1): exercise active diagnosis locally`

### 任务 6：真实 Triton workload 与 RTX 5090 验证

- [ ] 使用用户提供的真实 workload；没有 workload 时只验证机制，不声称方向准确率或优化收益。
- [ ] 5090 先运行 readiness；NCU 权限不足时保留 degraded/evidence_gap，不调整驱动策略。
- [ ] 与 3.0 对比：首个成立方向时间、GPU 时间、profile 轮次、重复采集、错误层级预算和
  上下文字节数。
- [ ] 做前向测试、epoch 漂移、缺表、窗口跨边界、权限变化和 resume 故障注入。
- [ ] 文档只报告实际测得结果，不把 micro fixture 的通过写成“更快找到方向”。

## 5. 拒绝条件

出现任一项，本切片不得进入 Controller 集成或真机验证：

- profiler 缺表或未覆盖 layer 被写成零耗时；
- execution map 没有绑定 workload、environment、epoch、source schema 和 adapter digest；
- 旧 epoch evidence 能支持当前假设；
- 模型能修改 cost、risk、perturbation、budget fit 或排序权重；
- 单一 metric 可以产生 `direction_supported`；
- `depends_on` 有环，或同一 pair 同时 exclusive/coexists；
- unknown idle 没有进入 `unmodeled` 假设；
- 没有可行区分证据时仍继续 profile；
- 同类失败请求改名后可以再次消耗预算；
- 为获取证据自动修改宿主机权限、驱动、频率、功耗或服务。
