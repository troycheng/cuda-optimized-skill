# CUDA Skill V2.7 Direction Admission Implementation Plan

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:executing-plans 或 superpowers:subagent-driven-development 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 V2.6 单轮迭代门禁之前增加只读的方向级准入与停止账本，防止 AI 通过更换机制名称反复消耗预算，并把精力转向同一结论层中可验证上限更高的方向。

**架构：** 新的标准库工具冻结目标、方向分类和最小有效提升，按 create-once 哈希链记录方向决策。它只读取用户或既有工具提供的测量快照，使用保守的全消除上限进行同层比较；跨层、吞吐量或复合目标明确标为不可自动排序。V2.6 继续负责候选级预算、证据和停止，不改变 V2.5 证据闭环，也不执行 workload、benchmark、profiler 或宿主机修改。

**技术栈：** Python 3 标准库、JSON Schema、`unittest`、Markdown。

---

### 任务 1：用失败测试锁定方向身份、收益上限与排序边界

**文件：**
- 创建：`tests/test_direction_guard.py`
- 创建：`skills/cuda-kernel-optimizer/templates/direction_portfolio.schema.json`
- 创建：`skills/cuda-kernel-optimizer/templates/direction_lineage.schema.json`
- 创建：`skills/cuda-kernel-optimizer/templates/direction_decision.schema.json`

- [ ] **步骤 1：编写方向身份与输入验证测试。** 覆盖闭合字段、重复键、有限数值、SHA-256、固定分类、目标/单位/方向一致性，以及机制描述不参与方向身份。
- [ ] **步骤 2：编写收益上限和同层排序测试。** 对 lower-is-better 的可加时间指标验证 `component / total` 全消除上限；上限低于冻结阈值时关闭方向，更高上限的同层方向触发切换。
- [ ] **步骤 3：编写不可排序测试。** 跨 kernel/runtime/workload/serving 结论层、throughput 和 composite 指标必须返回 `unrankable`，不得伪造权重或敏感性分析。
- [ ] **步骤 4：运行 RED。** 执行 `python3 -m unittest tests.test_direction_guard -v`，确认因 `direction_guard.py` 尚不存在而失败。
- [ ] **步骤 5：添加严格 schema。** 所有对象禁止未知字段；目标、测量窗口、环境、artifact 与 target 身份均用摘要绑定。

### 任务 2：实现 create-once 方向账本和确定性 CLI

**文件：**
- 创建：`skills/cuda-kernel-optimizer/scripts/direction_guard.py`
- 修改：`tests/test_direction_guard.py`

- [ ] **步骤 1：实现严格 JSON 与规范化身份。** 方向 family 由结论层、瓶颈类、component 与 metric 构成；具体 direction 再绑定 target identity，名称和机制说明不影响身份。
- [ ] **步骤 2：实现冻结 lineage。** `init` 固定 objective、minimum effect、environment、方向 family 集合和初始 portfolio 摘要，输出 canonical `direction-lineage.json`。
- [ ] **步骤 3：实现收益上限和准入决策。** 仅对同层可加 lower-is-better 指标自动比较；输出 `admit_direction`、`switch_to_higher_impact`、`close_direction` 或 `unrankable`，不声称真实性能收益。
- [ ] **步骤 4：实现停止、重开和防洗白。** 账本扫描完整 create-once 决策链；已关闭 family 默认拒绝。仅当新 evidence、measurement window 或 target identity 与旧记录不同，且重新计算的上限仍达到原冻结阈值时允许 `reopen`。更换 mechanism 名称不能重开。
- [ ] **步骤 5：实现安全 CLI。** `init`、`check`、`status` 只读目标并通过 no-follow/create-once artifact API 写账本；canonical run directory 中的 symlink、缺口、链断裂、并发重复写均 fail closed。
- [ ] **步骤 6：运行 GREEN。** 执行 `python3 -m unittest tests.test_direction_guard -v`。

### 任务 3：接入 V2.7 skill 协议并做压力测试

**文件：**
- 修改：`skills/cuda-kernel-optimizer/SKILL.md`
- 创建：`skills/cuda-kernel-optimizer/references/direction_admission.md`
- 修改：`skills/cuda-kernel-optimizer/scripts/self_check.py`
- 修改：`tests/test_skill_metadata.py`
- 修改：`tests/test_evidence_cli.py`

- [ ] **步骤 1：先写 RED 元数据和安装自检测试。** 要求 V2.7 标题、方向准入在 V2.6 之前、三个 CLI、三个 schema、reference、只读边界和 host recommend-only 边界。
- [ ] **步骤 2：更新 skill 路由和 reference。** `SKILL.md` 只保留何时调用和硬规则；完整输入、状态机、CLI、示例、失败语义放到按需 reference。
- [ ] **步骤 3：扩展 self-check。** 确认脚本、schema、reference 可安装且 Python 可编译，不需要 GPU。
- [ ] **步骤 4：运行 GREEN。** 执行 direction guard、metadata、evidence CLI 和 self-check focused tests。
- [ ] **步骤 5：复测原压力场景。** 让独立 reviewer 处理 selector 只占 5.5%、连续微调收益约 0.19%、已有多次自然语言停止、endpoint 双状态的场景；期望先调用方向账本，不再用新机制名绕过停止。

### 任务 4：更新双语 README 与 V2.2 起始 release notes

**文件：**
- 修改：`README.md`
- 修改：`README.zh-CN.md`
- 修改：`docs/workflows.md`
- 修改：`tests/test_readme_sync.py`
- 修改：`tests/test_public_docs.py`

- [ ] **步骤 1：先写 RED 文档测试。** 两份 README 必须新增对应的 Release notes/版本记录，并按 V2.7、V2.6、V2.5、V2.4、V2.3、V2.2 顺序完整列出；内容结构一致而非逐句硬译。
- [ ] **步骤 2：写 V2.2—V2.7 版本摘要。** 从本项目开始维护的 V2.2 写起，说明每版解决的问题、增加的能力和边界；不暗示 V2.2/V2.5 存在并未创建的 Git tag。
- [ ] **步骤 3：补充 V2.7 用户说明。** 用清晰自然的语言说明“先决定方向值不值得做，再进入候选迭代”，避免把内部脚本命令写成人工操作教程。
- [ ] **步骤 4：运行 focused 文档测试。** 执行 readme sync、public docs 与 metadata tests。

### 任务 5：审查、回归、双端发布和本地安装

**文件：**
- 修改：仅限审查发现涉及的 V2.7 文件

- [ ] **步骤 1：运行完整验证。** 执行 `self_check.py`、`python3 -m unittest discover -s tests -v`、`git diff --check`，并确认测试数量和跳过项。
- [ ] **步骤 2：进行独立代码审查与外部 AI 二次质证。** 重点检查 ledger 绕过、方向改名洗白、跨层错误排序、上限夸大、symlink/TOCTOU、文档误导；修复重要问题后重跑验证。
- [ ] **步骤 3：提交并合并到 main。** 在 `codex/v2.7-direction-admission` 提交，合并到主干，再在合并后的 main 上跑完整验证。
- [ ] **步骤 4：发布 V2.7。** 只向 fork `origin` 和内网 `internal` 推送相同 main SHA 与 V2.7 tag；绝不向 `upstream` 推送，并读回核验两个远端 SHA。
- [ ] **步骤 5：同步本地 skill。** 从已发布 main 安装到 `~/.codex/skills/cuda-kernel-optimizer`，执行安装目录 self-check，并核对源码与安装副本摘要。
