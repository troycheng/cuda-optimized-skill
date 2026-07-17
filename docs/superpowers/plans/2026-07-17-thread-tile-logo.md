# Thread Tile logo 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 制作 Thread Tile 的浅色、深色 SVG 与透明 PNG，并把主版本加入中英文 README 后发布到双远端。

**架构：** SVG 是唯一几何源文件，两个版本只允许填充色不同。PNG 由 `librsvg` 从主 SVG 确定性导出；Python 标准库测试验证几何、颜色、PNG 尺寸、实际透明像素和 README 引用。

**技术栈：** SVG、Python `unittest`/`xml.etree.ElementTree`/`struct`/`zlib`、`librsvg`、macOS `sips`、Git。

---

## 文件结构

- `asset/logo.svg`：浅色背景主版本，透明背景。
- `asset/logo-dark.svg`：深色背景版本，透明背景。
- `asset/logo-128.png`：128×128 透明 PNG。
- `asset/logo-512.png`：512×512 透明 PNG。
- `tests/test_logo_assets.py`：验证 SVG 几何和颜色、PNG 头信息及 README 引用。
- `tests/test_readme_sync.py`：把自动化验收数字更新为 691/686。
- `README.md`、`README.zh-CN.md`：顶部展示 88 px 主版本。

### 任务 1：用测试固定资源契约

**文件：**
- 创建：`tests/test_logo_assets.py`

- [ ] **步骤 1：编写失败测试**

测试应完成以下检查：

```python
def test_logo_asset_contract(self):
    light = parse_svg("logo.svg")
    dark = parse_svg("logo-dark.svg")
    self.assertEqual(light.root.attrib["viewBox"], "0 0 96 96")
    self.assertEqual(geometry(light), geometry(dark))
    self.assertEqual(fills(light), {"#172033": 8, "#16B8A6": 1})
    self.assertEqual(fills(dark), {"#F5F7FA": 8, "#28D6C2": 1})
    self.assertEqual(read_png_ihdr("logo-128.png"), (128, 128, 6))
    self.assertEqual(read_png_ihdr("logo-512.png"), (512, 512, 6))
    for readme in ("README.md", "README.zh-CN.md"):
        text = read(readme)
        self.assertIn('<img src="asset/logo.svg" width="88"', text)
```

- [ ] **步骤 2：验证红灯**

运行：`python3 -m unittest -v tests.test_logo_assets`

预期：FAIL，原因是 `asset/logo.svg` 尚不存在。

- [ ] **步骤 3：提交测试**

```bash
git add tests/test_logo_assets.py
git commit -m "test(logo): 固定 Thread Tile 资源契约"
```

### 任务 2：制作资源并接入 README

**文件：**
- 创建：`asset/logo.svg`
- 创建：`asset/logo-dark.svg`
- 创建：`asset/logo-128.png`
- 创建：`asset/logo-512.png`
- 修改：`README.md`
- 修改：`README.zh-CN.md`
- 修改：`tests/test_readme_sync.py`

- [ ] **步骤 1：创建两个 SVG**

两个文件均使用 `viewBox="0 0 96 96"`，包含 8 个相同的圆角矩形和 1 个中心菱形。浅色版本使用 `#172033`/`#16B8A6`，深色版本使用 `#F5F7FA`/`#28D6C2`。

- [ ] **步骤 2：导出透明 PNG**

```bash
rsvg-convert --width 128 --height 128 --output asset/logo-128.png asset/logo.svg
rsvg-convert --width 512 --height 512 --output asset/logo-512.png asset/logo.svg
sips -g pixelWidth -g pixelHeight -g hasAlpha asset/logo-128.png asset/logo-512.png
```

预期：尺寸分别为 128×128、512×512，`hasAlpha: yes`。

- [ ] **步骤 3：更新 README**

在两个 README 的 H1 下方加入：

```html
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="asset/logo-dark.svg">
    <img src="asset/logo.svg" width="88" alt="cuda-kernel-optimizer Thread Tile logo">
  </picture>
</p>
```

同时把两个 README 的自动化测试数字更新为 691 项、686 项通过、5 项跳过；把 `tests/test_readme_sync.py` 中对应的事实约束从 690/685 更新为 691/686。

- [ ] **步骤 4：验证资源与文档**

运行：

```bash
python3 -m unittest -v tests.test_logo_assets tests.test_readme_sync
git diff --check
```

预期：所有定向测试通过，diff check 无输出。

- [ ] **步骤 5：目视检查**

把两个 SVG 分别渲染到浅色和 `#101826` 深色背景；检查 28 px、128 px 和 512 px 显示，确认九宫格及中心菱形可辨认且没有不透明底色。

- [ ] **步骤 6：提交实现**

```bash
git add asset/logo.svg asset/logo-dark.svg asset/logo-128.png asset/logo-512.png README.md README.zh-CN.md
git commit -m "feat(logo): 添加 Thread Tile 项目标识"
```

### 任务 3：全量验证与发布

- [ ] **步骤 1：运行完整验证**

```bash
python3 -m unittest discover -q
python3 /Users/tcheng/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/cuda-kernel-optimizer
git diff --check main...HEAD
```

预期：691 项测试中 686 项通过、5 项 GPU opt-in 测试跳过、0 项失败；skill validator 输出 `Skill is valid!`。

- [ ] **步骤 2：合并并重新验证**

将 `agent/thread-tile-logo` fast-forward 合并到 `main`，在合并后的主干重新运行完整测试。

- [ ] **步骤 3：双远端发布**

先运行 `python3 tools/publish_dual_remote.py --tag v2.4.0`，确认 dry-run 的 main commit 和两个远端；再运行带 `--execute` 的命令。最后回读 GitHub 与内网 GitLab 的 `refs/heads/main`，两者必须等于本地主干，`v2.4.0` 仍指向原发布提交。
