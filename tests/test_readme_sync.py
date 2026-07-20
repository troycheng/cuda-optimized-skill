from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README_EN = ROOT / "README.md"
README_ZH = ROOT / "README.zh-CN.md"

MERMAID = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)
EDGE = re.compile(
    r"^\s*([a-z][a-z0-9_]*)[^-\n]*?\s*(-->|-\.->)\s*"
    r"([a-z][a-z0-9_]*)",
    re.MULTILINE,
)


def assert_in_order(testcase, text: str, markers: tuple[str, ...]) -> None:
    positions = [text.index(marker) for marker in markers]
    testcase.assertEqual(positions, sorted(positions))


class ReadmeSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.english = README_EN.read_text(encoding="utf-8")
        self.chinese = README_ZH.read_text(encoding="utf-8")

    def test_readmes_use_the_same_landing_page_structure(self) -> None:
        english = (
            "## About",
            "## Quick start",
            "## Choose a workflow",
            "## How it works",
            "## Evidence, not best-sample claims",
            "## Validation status",
            "## Release notes",
            "## Documentation",
        )
        chinese = (
            "## 项目简介",
            "## 快速开始",
            "## 选择工作流",
            "## 工作方式",
            "## 以证据为准，而不是选择最快样本",
            "## 验证情况",
            "## 版本记录",
            "## 文档",
        )
        assert_in_order(self, self.english, english)
        assert_in_order(self, self.chinese, chinese)
        self.assertLessEqual(len(self.english.splitlines()), 210)
        self.assertLessEqual(len(self.chinese.splitlines()), 210)

    def test_hero_uses_wordmark_tagline_and_primary_navigation(self) -> None:
        for text in (self.english, self.chinese):
            opening = text[: text.index("\n## ")]
            self.assertIn("asset/logo-wordmark-dark.svg", opening)
            self.assertIn("asset/logo-wordmark.svg", opening)
            self.assertIn('width="640"', opening)
            self.assertIn("CUDA", opening)
            self.assertIn("CUTLASS", opening)
            self.assertIn("Triton", opening)
            for target in (
                "docs/getting-started.md",
                "docs/environment-readiness.md",
                "docs/workflows.md",
                "docs/evidence-and-safety.md",
                "skills/cuda-kernel-optimizer/examples/walkthrough.md",
            ):
                self.assertIn(target, opening)
        self.assertIn(
            "Evidence-driven CUDA, CUTLASS and Triton optimization for Codex",
            self.english,
        )
        self.assertIn("以证据驱动 Codex 优化 CUDA、CUTLASS 与 Triton", self.chinese)

    def test_quick_start_precedes_protocol_detail(self) -> None:
        self.assertLess(
            self.english.index("## Quick start"),
            self.english.index("evidence_integrity"),
        )
        self.assertLess(
            self.chinese.index("## 快速开始"),
            self.chinese.index("evidence_integrity"),
        )
        self.assertIn("Installation is performed by Codex", self.english)
        self.assertIn("安装由 Codex 完成", self.chinese)
        for text in (self.english, self.chinese):
            self.assertIn("github.com/troycheng/cuda-optimized-skill", text)
            self.assertIn("skills/cuda-kernel-optimizer", text)
            for budget in ("quick", "balanced", "thorough"):
                self.assertIn(budget, text)

    def test_readmes_publish_the_same_five_workflows(self) -> None:
        english = (
            "Environment readiness",
            "Kernel optimization",
            "Complete workload",
            "Serving validation",
            "Existing NCU report",
        )
        chinese = (
            "环境准备",
            "Kernel 优化",
            "完整 workload",
            "Serving 验证",
            "已有 NCU report",
        )
        for marker in english:
            self.assertIn(marker, self.english)
        for marker in chinese:
            self.assertIn(marker, self.chinese)

    def test_readmes_show_one_matching_ai_workflow(self) -> None:
        expected = sorted(
            (
                ("goal", "-->", "environment"),
                ("environment", "-->", "baseline"),
                ("baseline", "-->", "profiling"),
                ("profiling", "-->", "change"),
                ("change", "-->", "evaluation"),
                ("evaluation", "-->", "keep"),
                ("evaluation", "-->", "restore"),
            )
        )
        english = MERMAID.findall(self.english)
        chinese = MERMAID.findall(self.chinese)
        self.assertEqual(len(english), 1)
        self.assertEqual(len(chinese), 1)
        self.assertEqual(sorted(EDGE.findall(english[0])), expected)
        self.assertEqual(sorted(EDGE.findall(chinese[0])), expected)

    def test_readmes_keep_v2_5_evidence_boundaries(self) -> None:
        common = (
            "95%",
            "shared-host",
            "evidence_integrity",
            "performance_verdict",
            "self_check",
            "c1/c2/c4/c8/c12",
            "CPU/static",
            "fail closed",
        )
        for text in (self.english, self.chinese):
            for marker in common:
                self.assertIn(marker, text)
        for marker in ("correctness", "paired A/B", "confidence interval", "frozen"):
            self.assertIn(marker, self.english)
        for marker in ("正确性", "成对 A/B", "置信区间", "冻结"):
            self.assertIn(marker, self.chinese)
        self.assertIn("does not validate a GPU environment", self.english)
        self.assertIn("不验证 GPU 环境", self.chinese)

    def test_real_workload_and_host_boundaries_are_explicit(self) -> None:
        self.assertIn("A real workload must be supplied by the user", self.english)
        self.assertIn("真实 workload 必须由用户提供", self.chinese)
        self.assertIn("does not download or invent one", self.english)
        self.assertIn("不会自行下载或编造", self.chinese)
        self.assertIn("never changes host-level settings automatically", self.english)
        self.assertIn("不会自动修改宿主机配置", self.chinese)

    def test_readmes_explain_automatic_readiness_admission(self) -> None:
        english = " ".join(self.english.split()).replace("`", "")
        for marker in (
            "The AI runs readiness automatically",
            "required capability",
            "does not start the baseline",
            "hash-locked isolated pip",
            "only automatic repair",
            "explicit authorization",
            "self_check does not prove that the GPU environment is ready",
        ):
            self.assertIn(marker, english)
        chinese = "".join(self.chinese.split()).replace("`", "")
        for marker in (
            "AI会自动完成环境准入检查",
            "必需能力",
            "不会启动baseline",
            "带哈希锁定的隔离环境pip",
            "唯一允许自动执行的修复",
            "明确授权",
            "self_check通过不代表GPU环境已经可用",
        ):
            self.assertIn(marker, chinese)

    def test_readmes_explain_the_performance_first_iteration_loop(self) -> None:
        english = " ".join(self.english.split())
        chinese = "".join(self.chinese.split())
        for marker in (
            "falsifiable performance hypothesis",
            "rehashed V2.5 evidence closure",
            "hard time and repair limit",
            "Tool work is not a performance improvement",
        ):
            self.assertIn(marker, english)
        for marker in (
            "能被实测推翻的性能假设",
            "重新校验通过的V2.5证据闭环",
            "时间和次数上限",
            "修工具不等于性能提升",
        ):
            self.assertIn(marker, chinese)
        reference = (
            "skills/cuda-kernel-optimizer/references/performance_iteration.md"
        )
        self.assertIn(reference, self.english)
        self.assertIn(reference, self.chinese)

    def test_readmes_publish_matching_release_notes_from_v2_2(self) -> None:
        english = self.english[self.english.index("## Release notes"):]
        chinese = self.chinese[self.chinese.index("## 版本记录"):]
        versions = ("### V3.1 (development)", "### V3.0.1", "### V3.0") + tuple(
            f"### V2.{minor}" for minor in range(9, 1, -1)
        )
        assert_in_order(self, english, versions)
        assert_in_order(self, chinese, versions)
        for version in versions:
            heading = re.compile(r"^%s$" % re.escape(version), re.MULTILINE)
            self.assertEqual(len(heading.findall(english)), 1)
            self.assertEqual(len(heading.findall(chinese)), 1)
        self.assertIn("maintained release history starts with V2.2", english)
        self.assertIn("从 V2.2 开始维护", chinese)
        self.assertIn("Readiness admission is under development", english)
        self.assertIn("环境准入仍在开发中", chinese)
        for marker in (
            "nonstationary",
            "direction-level",
            "performance-first",
            "formal evidence",
            "workload controller",
            "portable",
            "dual-loop",
        ):
            self.assertIn(marker, english)
        for marker in (
            "非平稳",
            "方向级",
            "性能优先",
            "正式证据",
            "workload controller",
            "可移植",
            "双循环",
        ):
            self.assertIn(marker, chinese)

    def test_validation_and_case_studies_are_separate(self) -> None:
        for text in (self.english, self.chinese):
            self.assertIn("docs/validation.md", text)
            self.assertIn("docs/case-studies.md", text)
        validation = (ROOT / "docs/validation.md").read_text(encoding="utf-8")
        cases = (ROOT / "docs/case-studies.md").read_text(encoding="utf-8")
        for fact in (
            "1031",
            "1023",
            "15 of 15",
            "34.307",
            "18 of 18",
            "52.141",
            "ERR_NVGPUCTRPERM",
        ):
            self.assertIn(fact, validation)
        for fact in ("60.4616%", "26.3287%", "-0.0097%", "140"):
            self.assertIn(fact, cases)
        self.assertNotIn("60.4616%", validation)
        self.assertNotIn("811", cases)

    def test_readmes_route_to_public_and_canonical_documents(self) -> None:
        links = (
            "docs/getting-started.md",
            "docs/environment-readiness.md",
            "docs/workflows.md",
            "docs/evidence-and-safety.md",
            "docs/compatibility.md",
            "docs/validation.md",
            "docs/case-studies.md",
            "docs/knowledge-and-research.md",
            "docs/long-running-optimization.md",
            "skills/cuda-kernel-optimizer/SKILL.md",
            "skills/cuda-kernel-optimizer/examples/walkthrough.md",
            "skills/cuda-kernel-optimizer/references/evidence_automation.md",
            "skills/cuda-kernel-optimizer/references/performance_iteration.md",
            "skills/cuda-kernel-optimizer/references/compatibility.md",
            "skills/cuda-kernel-optimizer/references/long_running_control.md",
            "skills/cuda-kernel-optimizer/references/version_stack_audit.md",
            "tests/gpu/sm120/README.md",
            "LICENSE",
        )
        for text in (self.english, self.chinese):
            for marker in links:
                self.assertIn(marker, text)

    def test_readmes_are_not_internal_cli_or_marketing_guides(self) -> None:
        banned = (
            "python3 scripts/orchestrate.py",
            "python3 scripts/workload_controller.py",
            "python3 tools/publish_dual_remote.py",
            "--run-dir",
            "promotion authority",
            "terminal status",
            "powerful",
            "seamless",
            "revolutionary",
            "comprehensive",
            "可信边界",
            "终局状态",
            "赋能",
            "无缝",
            "强大",
        )
        for text in (self.english, self.chinese):
            for marker in banned:
                self.assertNotIn(marker, text.lower() if marker.isascii() else text)
            self.assertNotIn("```bash", text)

    def test_readmes_link_to_each_other(self) -> None:
        self.assertIn("README.zh-CN.md", self.english)
        self.assertIn("README.md", self.chinese)

    def test_local_readme_links_resolve(self) -> None:
        markdown = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
        html = re.compile(r'href="([^"]+)"')
        for path, text in ((README_EN, self.english), (README_ZH, self.chinese)):
            for target in markdown.findall(text) + html.findall(text):
                if "://" in target or target.startswith("#"):
                    continue
                resolved = (path.parent / target.split("#", 1)[0]).resolve()
                self.assertTrue(resolved.exists(), f"missing README link: {target}")


if __name__ == "__main__":
    unittest.main()
