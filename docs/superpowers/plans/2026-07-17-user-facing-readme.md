# 面向使用者的 README 重写实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将中英文 README 从内部 CLI 手册重写为清晰、专业的项目说明，让读者理解项目能力、输入、AI 执行过程、结果判定、交付物和限制。

**架构：** README 采用相同的十段式信息架构，中文和英文分别按母语习惯写作。主阅读路径只保留一张高层流程图和少量自然语言示例；内部命令、schema、状态机和详细复现信息改为链接到现有专业文档。

**技术栈：** Markdown、Mermaid、Python `unittest`

---

## 文件结构

- 修改：`tests/test_readme_sync.py`——验证新的使用者信息架构、双语能力一致性和内部命令下沉。
- 修改：`README.zh-CN.md`——中文主文档，使用自然、直接的中文技术写作。
- 修改：`README.md`——英文主文档，与中文版信息对等但不逐句直译。
- 保留：`skills/cuda-kernel-optimizer/SKILL.md`——AI 执行协议，不在本任务中改写。
- 保留：`skills/cuda-kernel-optimizer/examples/workload-controller.md`——完整 workload controller 契约。
- 保留：`tests/gpu/sm120/README.md`——RTX 5090 详细验收和复现入口。

### 任务 1：用测试固定新的 README 信息架构

**文件：**
- 修改：`tests/test_readme_sync.py`
- 测试：`tests/test_readme_sync.py`

- [ ] **步骤 1：删除旧的 CLI 导向断言，写入新的失败测试**

将旧测试中对以下内容的强制要求删除：

```text
python3 scripts/orchestrate.py setup
python3 scripts/orchestrate.py open-iter
python3 scripts/workload_controller.py run
python3 scripts/strategy_memory.py record
三张 Mermaid 图
artifact 目录树
镜像 SHA 与逐项验收 hash
```

新增以下测试：

```python
def test_readmes_answer_the_same_user_questions(self) -> None:
    english = (
        "## What this project is",
        "## Problems it can solve",
        "## What you need to provide",
        "## How the AI works",
        "## How results are accepted",
        "## What you receive",
        "## Modification scope and safety limits",
        "## Usage examples",
        "## Tested environments and compatibility",
        "## Installation and further documentation",
    )
    chinese = (
        "## 项目是什么",
        "## 可以解决哪些问题",
        "## 需要提供什么",
        "## AI 会如何执行",
        "## 如何确认优化结果",
        "## 最终会得到什么",
        "## 修改范围与安全限制",
        "## 使用示例",
        "## 测试情况与兼容性",
        "## 安装与进一步文档",
    )
    assert_in_order(self, self.english, english)
    assert_in_order(self, self.chinese, chinese)

def test_opening_defines_purpose_and_extent_without_version_history(self) -> None:
    for text in (self.english, self.chinese):
        opening = text[: text.index("\n## ")]
        for marker in ("CUDA", "CUTLASS", "Triton", "kernel", "workload"):
            self.assertIn(marker, opening)
        self.assertNotIn("V2.2", opening)
        self.assertNotIn("V2.4", opening)

def test_readmes_are_not_manual_cli_guides(self) -> None:
    banned = (
        "python3 scripts/orchestrate.py",
        "python3 scripts/workload_controller.py",
        "python3 scripts/strategy_memory.py",
        "python3 tools/publish_dual_remote.py",
        "--run-dir",
        "register-change",
    )
    for text in (self.english, self.chinese):
        for marker in banned:
            self.assertNotIn(marker, text)
        self.assertEqual(len(MERMAID.findall(text)), 1)

def test_readmes_explain_inputs_outputs_evidence_and_limits(self) -> None:
    for text in (self.english, self.chinese):
        for marker in (
            "baseline",
            "profiling",
            "95%",
            "balanced",
            "RTX 5090",
            "ERR_NVGPUCTRPERM",
        ):
            self.assertIn(marker, text)
    self.assertIn("paired A/B", self.english)
    self.assertIn("成对 A/B", self.chinese)
    self.assertIn("host-level", self.english)
    self.assertIn("宿主机", self.chinese)
```

保留对 `quick`、`balanced`、`thorough`、真实 workload 必须由用户提供、5090
验收结论、关键引用链接和禁用营销词的测试，但改为匹配新正文。

- [ ] **步骤 2：运行测试确认旧 README 不满足新约束**

运行：

```bash
python3 -m unittest -v tests.test_readme_sync
```

预期：FAIL。失败原因应包括缺少新章节，以及仍然存在 `orchestrate.py`、
`workload_controller.py` 等手工 CLI。

- [ ] **步骤 3：提交测试约束**

```bash
git add tests/test_readme_sync.py
git commit -m "test(README): 固定面向使用者的信息架构"
```

### 任务 2：重写中文 README

**文件：**
- 修改：`README.zh-CN.md`
- 测试：`tests/test_readme_sync.py`

- [ ] **步骤 1：按确认的信息架构重写中文正文**

开篇使用三段式定位：

```markdown
`cuda-kernel-optimizer` 是一个面向 Codex 的 CUDA 性能优化项目，核心能力以
skill 形式提供。它既能优化单个 CUDA、CUTLASS 或 Triton kernel，也能分析和
优化用户提供的完整 GPU workload。

用户提供可运行代码、测试环境和性能目标后，AI 会完成环境检查、性能分析
（profiling）、瓶颈定位、代码修改和成对 A/B 测试。修改只有在结果正确、性能提升
达到目标且所有约束满足时才会保留。

项目可以修改授权范围内的 kernel、运行参数和项目代码。涉及驱动、权限、频率、
功耗或其他宿主机配置时，只给出建议，不自动执行。
```

正文必须包含：

- 一张能力表，列为「任务」「适用场景」「AI 会做什么」「主要结果」；
- 一张 Mermaid 高层流程图；
- `quick`、`balanced`（默认）、`thorough` 的用途和最长时间简表；
- 正确性、成对 A/B、95% 置信区间、性能门槛和业务约束的通俗解释；
- 交付物列表；
- 自动修改范围与宿主机建议边界；
- 4 个自然语言使用示例；
- 真实 RTX 5090、vLLM workload、NCU report 和 CPU 测试的简要结果；
- 指向 `SKILL.md`、workload controller 示例、compatibility、optimization catalog、
  SM120 测试说明和 LICENSE 的链接。

不得包含内部运行命令、JSON manifest、artifact 目录树、发布流程、镜像 SHA 或逐项
验收 hash。

- [ ] **步骤 2：运行 README 测试，确认中文版满足新结构**

运行：

```bash
python3 -m unittest -v tests.test_readme_sync
```

预期：仍可能因英文版尚未重写而 FAIL，但不得再出现中文版缺少章节或含内部 CLI 的
失败。

- [ ] **步骤 3：提交中文 README**

```bash
git add README.zh-CN.md
git commit -m "docs(README): 重写中文项目说明"
```

### 任务 3：重写英文 README

**文件：**
- 修改：`README.md`
- 测试：`tests/test_readme_sync.py`

- [ ] **步骤 1：编写信息对等的英文正文**

英文开篇使用自然英文，不逐句翻译中文：

```markdown
`cuda-kernel-optimizer` is a CUDA performance optimization project for Codex,
with its core workflow packaged as a reusable skill. It can optimize an
individual CUDA, CUTLASS, or Triton kernel, or analyze and improve a complete
user-provided GPU workload.
```

使用与中文版相同的十个章节、能力集合、预算和测试结论。专业术语使用英文行业惯例；
避免 `powerful`、`seamless`、`comprehensive` 等营销词。自然语言任务只放在
`Usage examples` 章节。

- [ ] **步骤 2：运行 README 测试确认中英文一致**

运行：

```bash
python3 -m unittest -v tests.test_readme_sync
```

预期：所有 README 测试通过。

- [ ] **步骤 3：提交英文 README**

```bash
git add README.md
git commit -m "docs(README): rewrite the user-facing overview"
```

### 任务 4：完整验证和文风检查

**文件：**
- 验证：`README.md`
- 验证：`README.zh-CN.md`
- 验证：`tests/test_readme_sync.py`

- [ ] **步骤 1：检查结构、术语和链接**

运行：

```bash
rg -n '^#{1,3} ' README.md README.zh-CN.md
rg -n 'python3 scripts/|publish_dual_remote' README.md README.zh-CN.md
git diff --check main...HEAD
```

预期：章节顺序与规格一致；无占位符、内部执行命令或空白错误。

- [ ] **步骤 2：运行完整测试与 skill 校验**

运行：

```bash
python3 -m unittest discover
python3 /Users/tcheng/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  skills/cuda-kernel-optimizer
python3 -m py_compile skills/cuda-kernel-optimizer/scripts/*.py
```

预期：完整测试通过，5 个 opt-in RTX 5090 测试在本机跳过；skill validator 返回
`Skill is valid!`，脚本语法检查退出码为 0。

- [ ] **步骤 3：检查提交范围**

运行：

```bash
git status --short
git diff --stat main...HEAD
git log --oneline main..HEAD
```

预期：仅包含规格、计划、README 与 README 测试变更，没有运行时代码修改。
