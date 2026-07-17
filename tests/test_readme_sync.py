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


def section(text: str, heading: str, next_heading: str | None = None) -> str:
    start = text.index(heading)
    if next_heading is None:
        return text[start:]
    return text[start : text.index(next_heading, start + len(heading))]


class ReadmeSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.english = README_EN.read_text(encoding="utf-8")
        self.chinese = README_ZH.read_text(encoding="utf-8")

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
        for text in (self.english, self.chinese):
            h1s = [line for line in text.splitlines() if line.startswith("# ")]
            self.assertEqual(h1s, ["# cuda-kernel-optimizer"])

    def test_opening_defines_purpose_and_extent_without_version_history(self) -> None:
        for text in (self.english, self.chinese):
            opening = text[: text.index("\n## ")]
            for marker in ("Codex", "CUDA", "CUTLASS", "Triton", "kernel", "workload"):
                self.assertIn(marker, opening)
            self.assertNotIn("V2.2", opening)
            self.assertNotIn("V2.4", opening)
            self.assertIn("profiling", opening)
            self.assertIn("A/B", opening)

    def test_readmes_are_not_manual_cli_guides(self) -> None:
        banned = (
            "python3 scripts/orchestrate.py",
            "python3 scripts/workload_controller.py",
            "python3 scripts/strategy_memory.py",
            "python3 tools/publish_dual_remote.py",
            "--run-dir",
            "register-change",
            '"kind": "python"',
            "run_YYYYMMDD_HHMMSS/",
        )
        for text in (self.english, self.chinese):
            for marker in banned:
                self.assertNotIn(marker, text)
            self.assertEqual(len(MERMAID.findall(text)), 1)
            self.assertNotIn("```bash", text)

    def test_readmes_publish_the_same_capability_set(self) -> None:
        english = (
            "Optimize one kernel",
            "Optimize a complete GPU workload",
            "Validate an optimization on a real workload",
            "Analyze an existing NCU report",
        )
        chinese = (
            "优化单个 kernel",
            "优化完整 GPU workload",
            "在真实 workload 上验证优化",
            "分析已有 NCU report",
        )
        for marker in english:
            self.assertIn(marker, self.english)
        for marker in chinese:
            self.assertIn(marker, self.chinese)

    def test_readmes_explain_user_inputs_and_default_budget(self) -> None:
        for text in (self.english, self.chinese):
            for marker in (
                "baseline",
                "reference",
                "workload",
                "quick",
                "balanced",
                "thorough",
            ):
                self.assertIn(marker, text)
            self.assertIn("45", text)
            self.assertIn("3", text)
            self.assertIn("10", text)
        self.assertIn("balanced` is the default", self.english)
        self.assertIn("默认使用 `balanced`", self.chinese)
        self.assertIn("must be supplied by the user", self.english)
        self.assertIn("必须由用户提供", self.chinese)
        self.assertIn("performance goal", self.english)
        self.assertIn("性能目标", self.chinese)

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

    def test_readmes_explain_how_results_are_accepted(self) -> None:
        for text in (self.english, self.chinese):
            self.assertIn("95%", text)
        for marker in ("correctness", "confidence interval", "constraint", "restore"):
            self.assertIn(marker, self.english)
        for marker in ("正确性", "置信区间", "约束", "恢复"):
            self.assertIn(marker, self.chinese)
        self.assertIn("paired A/B", self.english)
        self.assertIn("成对 A/B", self.chinese)
        self.assertIn("same inputs", self.english)
        self.assertIn("相同输入", self.chinese)

    def test_readmes_describe_outputs_and_operation_limits(self) -> None:
        for marker in (
            "modified code",
            "bottleneck analysis",
            "performance comparison",
            "host recommendations",
            "isolated environment",
            "reviewer",
        ):
            self.assertIn(marker, self.english)
        for marker in (
            "修改后的代码",
            "瓶颈分析",
            "性能对比",
            "宿主机建议",
            "隔离环境",
            "reviewer",
        ):
            self.assertIn(marker, self.chinese)
        self.assertIn("host-level settings", self.english)
        self.assertIn("宿主机配置", self.chinese)
        self.assertIn("never applied automatically", self.english)
        self.assertIn("不会自动执行", self.chinese)

    def test_natural_language_requests_are_confined_to_examples(self) -> None:
        english = section(
            self.english,
            "## Usage examples",
            "## Tested environments and compatibility",
        )
        chinese = section(
            self.chinese,
            "## 使用示例",
            "## 测试情况与兼容性",
        )
        self.assertEqual(english.count("> "), 4)
        self.assertEqual(chinese.count("> "), 4)
        for marker in ("Triton", "GPU workload", "NCU", "balanced"):
            self.assertIn(marker, english)
            self.assertIn(marker, chinese)

    def test_readmes_keep_concise_current_validation_facts(self) -> None:
        facts = (
            "690",
            "685",
            "13/13",
            "34.302",
            "60.4616%",
            "26.3287%",
            "-0.0097%",
            "140",
            "ERR_NVGPUCTRPERM",
        )
        for fact in facts:
            self.assertIn(fact, self.english)
            self.assertEqual(self.english.count(fact), self.chinese.count(fact))
            self.assertIn(fact, self.chinese)
        for text in (self.english, self.chinese):
            self.assertNotIn(
                "sha256:a2d9d89bc4394eab3fadc62c6b5b3f739b6494c1f64c56f5ba5e6c008252a0e5",
                text,
            )
            self.assertNotIn(
                "01a1356a487cc1ce77c6af541508db2c5a673dbfa9370bed30d095162321574d",
                text,
            )

    def test_readmes_link_to_execution_and_evidence_documents(self) -> None:
        links = (
            "skills/cuda-kernel-optimizer/SKILL.md",
            "skills/cuda-kernel-optimizer/examples/workload-controller.md",
            "skills/cuda-kernel-optimizer/references/compatibility.md",
            "skills/cuda-kernel-optimizer/references/optimization_catalog.md",
            "tests/gpu/sm120/README.md",
            "LICENSE",
            "troycheng/cuda-optimized-skill",
        )
        for text in (self.english, self.chinese):
            for marker in links:
                self.assertIn(marker, text)

    def test_readmes_avoid_internal_and_marketing_language(self) -> None:
        for text in (self.english, self.chinese):
            lower = text.lower()
            for banned in (
                "promotion authority",
                "terminal status",
                "side evidence path",
                "powerful",
                "seamless",
                "revolutionary",
                "comprehensive",
            ):
                self.assertNotIn(banned, lower)
            for banned in ("可信边界", "终局状态", "旁路证据", "赋能", "无缝", "强大"):
                self.assertNotIn(banned, text)
        self.assertNotRegex(self.chinese, r"通过[^。\n]{0,80}从而")


if __name__ == "__main__":
    unittest.main()
