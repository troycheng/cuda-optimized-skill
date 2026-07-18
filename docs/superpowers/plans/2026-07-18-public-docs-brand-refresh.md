# Public Documentation and Brand Refresh Implementation Plan

> **面向 AI 代理的工作者：** 必需子技能：使用
> `superpowers-zh:executing-plans` 逐任务实现此计划。步骤使用复选框
> （`- [ ]`）语法跟踪进度。

**目标：** 把仓库首页改造成清晰的项目入口，增加 Thread Tile 横向字标，并建立不与
AI 执行协议和研发历史混杂的轻量公共文档导航。

**架构：** 保留 `skills/cuda-kernel-optimizer/` 为 Codex 的规范执行包；根目录双语
README 只承担品牌、Quick Start、能力和证据边界；`docs/` 新增五页用户指南并由
`mkdocs.yml` 导航；`docs/superpowers/` 保留为不进入公共导航的内部历史。

**技术栈：** Markdown、SVG、MkDocs 配置、CSS、Python `unittest`、XML
`ElementTree`。

---

## 文件职责

- 修改 `README.md`：英文产品入口和文档路由。
- 修改 `README.zh-CN.md`：与英文版能力和证据边界对等的中文入口。
- 创建 `asset/logo-wordmark.svg`：浅色背景横向字标。
- 创建 `asset/logo-wordmark-dark.svg`：深色背景横向字标。
- 创建 `mkdocs.yml`：公共文档站的唯一导航定义。
- 创建 `docs/index.md`：公共文档首页。
- 创建 `docs/getting-started.md`：安装、输入和第一次任务。
- 创建 `docs/workflows.md`：四条工作流的选择和输出边界。
- 创建 `docs/evidence-and-safety.md`：证据完整性、fail-closed 和安全范围。
- 创建 `docs/compatibility.md`：工具链、硬件和 NCU 兼容性入口。
- 创建 `docs/stylesheets/extra.css`：Carbon + Cyan 的轻量品牌样式。
- 创建 `docs/superpowers/README.md`：标记内部设计和实施历史。
- 修改 `tests/test_logo_assets.py`：固定横向字标资产契约。
- 修改 `tests/test_readme_sync.py`：固定双语首屏、章节顺序和证据边界。
- 创建 `tests/test_public_docs.py`：固定 MkDocs 导航、公共页面和相对链接。

### 任务 1：用测试固定横向字标

**文件：**

- 修改：`tests/test_logo_assets.py`
- 创建：`asset/logo-wordmark.svg`
- 创建：`asset/logo-wordmark-dark.svg`

- [ ] **步骤 1：编写失败的字标测试**

在 `LogoAssetTests` 中增加测试，解析两个新 SVG，并验证透明画布、可访问标题、
`0 0 720 152` 视图、既有配色以及两段字标文本：

```python
def test_wordmark_asset_contract(self) -> None:
    expected = {
        "logo-wordmark.svg": ("#172033", "#16B8A6"),
        "logo-wordmark-dark.svg": ("#F5F7FA", "#28D6C2"),
    }
    for name, (foreground, accent) in expected.items():
        root = parse_svg(name)
        self.assertEqual(root.attrib["viewBox"], "0 0 720 152")
        self.assertNotIn("width", root.attrib)
        self.assertNotIn("height", root.attrib)
        self.assertEqual(root.attrib["role"], "img")
        self.assertEqual(root.attrib["aria-labelledby"], "title")
        self.assertNotIn("<rect width=\"720\"", (ASSET_DIR / name).read_text())
        text = " ".join(element.text or "" for element in root.iter())
        self.assertIn("CUDA KERNEL", text)
        self.assertIn("OPTIMIZER", text)
        self.assertIn(foreground, (ASSET_DIR / name).read_text())
        self.assertIn(accent, (ASSET_DIR / name).read_text())
```

- [ ] **步骤 2：运行测试并确认缺少资产时失败**

运行：

```bash
python3 -m unittest tests.test_logo_assets.LogoAssetTests.test_wordmark_asset_contract -v
```

预期：`ERROR`，指出 `asset/logo-wordmark.svg` 不存在。

- [ ] **步骤 3：创建浅色和深色横向字标**

两个 SVG 均使用 720×152 透明 `viewBox`。在左侧复用 96×96 Thread Tile 几何；
右侧使用带系统字体回退的 SVG `<text>`，第一行为 `CUDA KERNEL`，第二行为
`OPTIMIZER`。浅色版仅使用 `#172033/#16B8A6`，深色版仅使用
`#F5F7FA/#28D6C2`。不要增加背景、渐变、阴影或第三方商标。

- [ ] **步骤 4：运行完整 Logo 测试**

运行：

```bash
python3 -m unittest tests.test_logo_assets -v
```

预期：所有 Logo 测试通过。

- [ ] **步骤 5：提交字标资产和测试**

```bash
git add tests/test_logo_assets.py asset/logo-wordmark.svg asset/logo-wordmark-dark.svg
git commit -m "feat(logo): add Thread Tile wordmark"
```

### 任务 2：把双语 README 改成项目入口

**文件：**

- 修改：`tests/test_readme_sync.py`
- 修改：`README.md`
- 修改：`README.zh-CN.md`

- [ ] **步骤 1：将旧章节测试替换为新信息架构测试**

更新双语章节顺序断言：

```python
english = (
    "## About",
    "## Quick start",
    "## Choose a workflow",
    "## How it works",
    "## Evidence, not best-sample claims",
    "## Tested scope",
    "## Documentation",
)
chinese = (
    "## 项目简介",
    "## 快速开始",
    "## 选择工作流",
    "## 工作方式",
    "## 以证据为准，而不是选择最快样本",
    "## 已测试范围",
    "## 文档",
)
```

增加以下精确契约：

```python
for text in (self.english, self.chinese):
    self.assertIn("asset/logo-wordmark-dark.svg", text)
    self.assertIn("asset/logo-wordmark.svg", text)
    self.assertLess(text.index("Quick start") if text is self.english else text.index("快速开始"),
                    text.index("evidence_integrity"))
    self.assertEqual(len(MERMAID.findall(text)), 1)
    self.assertLessEqual(len(text.splitlines()), 190)
```

保留并适配以下边界测试：四种工作流、真实 workload 归用户所有、成对 A/B、95%
置信区间、`shared-host`、`evidence_integrity`、`self_check`、
`c1/c2/c4/c8/c12`、CPU/static 不代表新增 GPU 验证、历史性能数字不构成承诺、所有
本地链接可解析。删除只服务于旧十章节结构和完整预算表的断言。

- [ ] **步骤 2：运行 README 测试并确认旧 README 失败**

运行：

```bash
python3 -m unittest tests.test_readme_sync -v
```

预期：因缺少新字标引用、Quick Start 位置和新章节标题而失败。

- [ ] **步骤 3：重写英文 README**

使用以下固定首屏和章节：

```markdown
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="asset/logo-wordmark-dark.svg">
    <img src="asset/logo-wordmark.svg" width="640" alt="CUDA Kernel Optimizer">
  </picture>
</p>

<p align="center"><strong>Evidence-driven CUDA, CUTLASS and Triton optimization for Codex</strong></p>

<p align="center">
  <a href="docs/getting-started.md">Get Started</a> ·
  <a href="docs/workflows.md">Workflows</a> ·
  <a href="docs/evidence-and-safety.md">Evidence &amp; Safety</a> ·
  <a href="skills/cuda-kernel-optimizer/examples/walkthrough.md">Examples</a> ·
  <a href="README.zh-CN.md">简体中文</a>
</p>
```

正文按已批准的七个章节组织。Quick Start 明确让 Codex 从 GitHub 仓库的
`skills/cuda-kernel-optimizer` 安装，并给出一个 `>` 自然语言任务。工作流表包含
kernel、完整 workload、serving validation、只读 NCU report。证据段保留 fail-closed
边界；测试段把既有数字标记为历史验收证据，不声称本次文档修改运行了 GPU。

- [ ] **步骤 4：重写中文 README**

使用同样的首屏、链接和章节顺序，中文独立成文而非逐句机械翻译。所有数字、能力、
正式证据术语和安全边界必须与英文版对等；保留 `CUDA`、`CUTLASS`、`Triton`、
`kernel`、`workload`、`shared-host`、`evidence_integrity` 和 `self_check` 原词。

- [ ] **步骤 5：运行 README 和 Logo 测试**

运行：

```bash
python3 -m unittest tests.test_readme_sync tests.test_logo_assets -v
```

预期：所有测试通过。

- [ ] **步骤 6：提交 README 重构**

```bash
git add README.md README.zh-CN.md tests/test_readme_sync.py
git commit -m "docs: turn README into project landing page"
```

### 任务 3：建立公共文档导航

**文件：**

- 创建：`tests/test_public_docs.py`
- 创建：`mkdocs.yml`
- 创建：`docs/index.md`
- 创建：`docs/getting-started.md`
- 创建：`docs/workflows.md`
- 创建：`docs/evidence-and-safety.md`
- 创建：`docs/compatibility.md`
- 创建：`docs/stylesheets/extra.css`
- 创建：`docs/superpowers/README.md`

- [ ] **步骤 1：编写失败的公共文档结构测试**

创建 `tests/test_public_docs.py`，验证配置和页面存在、导航顺序、Superpowers 历史不在
公共导航、相对链接可解析：

```python
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PAGES = (
    "docs/index.md",
    "docs/getting-started.md",
    "docs/workflows.md",
    "docs/evidence-and-safety.md",
    "docs/compatibility.md",
)

class PublicDocsTests(unittest.TestCase):
    def test_public_navigation_contract(self) -> None:
        config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        for page in PUBLIC_PAGES:
            self.assertIn(page.removeprefix("docs/"), config)
        self.assertNotIn("superpowers", config.lower())
        self.assertIn("stylesheets/extra.css", config)

    def test_public_pages_and_relative_links(self) -> None:
        pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
        for relative in PUBLIC_PAGES:
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            for target in pattern.findall(path.read_text(encoding="utf-8")):
                if "://" in target or target.startswith("#"):
                    continue
                resolved = (path.parent / target.split("#", 1)[0]).resolve()
                self.assertTrue(resolved.exists(), f"missing docs link: {target}")
```

增加内容边界断言：Getting Started 包含 Codex 安装入口；Workflows 包含四种模式；
Evidence & Safety 同时包含 `performance_verdict`、`evidence_integrity`、fail-closed、
shared-host 和 CPU/static self-check 限制；Compatibility 链接到 canonical compatibility
reference；`docs/superpowers/README.md` 明确是 internal history。

- [ ] **步骤 2：运行测试并确认配置和页面缺失**

运行：

```bash
python3 -m unittest tests.test_public_docs -v
```

预期：`ERROR` 或 `FAIL`，指出 `mkdocs.yml` 或公共页面不存在。

- [ ] **步骤 3：创建 MkDocs 配置和品牌 CSS**

`mkdocs.yml` 使用内置 `mkdocs` theme，设置 `site_name`、GitHub `repo_url`、五页 nav、
外部 Agent Protocol 链接以及 `extra_css: [stylesheets/extra.css]`。CSS 只定义
`#172033` 文本/导航色和 `#16B8A6` 链接/强调色，不引入字体、脚本或远程资源。

- [ ] **步骤 4：创建五页用户指南和内部历史说明**

每页只承担一个职责：

- `index.md`：一句话定位和五个入口；
- `getting-started.md`：安装、必需输入、预算和第一个任务；
- `workflows.md`：四种工作流的输入、允许修改和结论边界；
- `evidence-and-safety.md`：证据漏斗、正式尝试、环境 guard、判定分离和安全范围；
- `compatibility.md`：CUDA/CUTLASS/Triton/NCU 条件与 canonical reference；
- `superpowers/README.md`：内部历史用途及非用户协议声明。

文档不得复制完整 Schema 或声称历史 GPU 数据是本次验证结果。

- [ ] **步骤 5：运行公共文档和双语入口测试**

运行：

```bash
python3 -m unittest tests.test_public_docs tests.test_readme_sync -v
```

预期：所有测试通过。

- [ ] **步骤 6：提交公共文档层**

```bash
git add mkdocs.yml docs tests/test_public_docs.py
git commit -m "docs: add public documentation navigation"
```

### 任务 4：完成 CPU/static 验证和审查

**文件：**

- 验证：整个仓库
- 不修改：运行时脚本、证据 Schema、GPU fixture

- [ ] **步骤 1：运行完整 CPU/static 测试套件**

运行：

```bash
python3 -m unittest discover -s tests -v
```

预期：所有非 GPU 测试通过；RTX 5090 opt-in 测试按既有条件跳过。记录总数、通过、跳过、
失败和耗时。若出现 flaky，单独重复失败测试并如实报告。

- [ ] **步骤 2：运行 skill validator 和安装后 self-check**

运行：

```bash
python3 /Users/tcheng/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/cuda-kernel-optimizer
python3 skills/cuda-kernel-optimizer/scripts/self_check.py --skill-dir skills/cuda-kernel-optimizer
```

预期：`Skill is valid!`；self-check 返回 `PASS`、`gpu_checks_run: false`、
`network_checks_run: false`。

- [ ] **步骤 3：验证 Markdown、SVG 和仓库状态**

运行：

```bash
python3 -m compileall -q skills/cuda-kernel-optimizer/scripts tests
git diff --check
git status --short --branch
```

预期：compileall 和 diff check 返回 0；状态只包含本计划产生且尚未提交的预期文件，或
所有实施提交后为空。

- [ ] **步骤 4：目视检查横向字标**

把两个 SVG 渲染为本地预览，分别在浅色和深色背景检查字标间距、中心强调色、透明
背景和小尺寸可读性。若字标文字被裁切或比例失衡，调整 SVG 后重新运行 Logo 测试。

- [ ] **步骤 5：审查最终 diff 和提交历史**

运行：

```bash
git diff origin/main...HEAD --check
git diff --stat origin/main...HEAD
git log --oneline origin/main..HEAD
```

确认变更仅覆盖批准的品牌、README、公共文档、测试、规格和计划；不含运行时、GPU
fixture、`/data/triton-handoff` 或外部进程变更。
