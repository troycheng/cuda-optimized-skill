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


def assert_in_order(testcase, text, markers):
    positions = [text.index(marker) for marker in markers]
    testcase.assertEqual(positions, sorted(positions))


def mermaid_topology(block):
    return sorted(EDGE.findall(block))


class ReadmeSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.english = README_EN.read_text(encoding="utf-8")
        self.chinese = README_ZH.read_text(encoding="utf-8")

    def test_readmes_identify_v2_2_dual_loop_and_balanced_default(self) -> None:
        for text in (self.english, self.chinese):
            self.assertIn("V2.2", text)
            self.assertIn("balanced", text)
            self.assertIn("kernel_only_win", text)
            self.assertIn("end_to_end_win", text)
        self.assertIn("dual-loop", self.english.lower())
        self.assertIn("双环", self.chinese)

    def test_readmes_publish_the_same_budget_presets(self) -> None:
        expected = {
            "quick": ("2700", "4", "2", "20", "50", "1", "3", "targeted"),
            "balanced": ("10800", "8", "4", "20", "100", "2", "10", "targeted"),
            "thorough": ("36000", "16", "8", "30", "200", "3", "unlimited", "full"),
        }
        for text in (self.english, self.chinese):
            for name, expected_cells in expected.items():
                row = next(
                    line for line in text.splitlines() if f"| `{name}`" in line
                )
                cells = tuple(cell.strip() for cell in row.strip("|").split("|"))
                self.assertIn(name, cells[0])
                self.assertEqual(cells[1:], expected_cells)

    def test_readmes_document_all_user_owned_workload_inputs(self) -> None:
        for text in (self.english, self.chinese):
            for option in ("--workload", "--workload-cmd", "--workload-manifest"):
                self.assertIn(option, text)
            self.assertIn("--objective", text)
            self.assertIn('"kind": "python"', text)
            self.assertIn('"source": "./workload.py"', text)
            self.assertIn('"cases": [', text)
        self.assertIn("embedded objective or --objective, never both", self.english)
        self.assertIn("内嵌 objective 或 --objective，不能同时使用", self.chinese)

    def test_readmes_describe_kernel_only_win_in_both_modes(self) -> None:
        self.assertIn(
            "`kernel_only_win` confirms only the kernel result", self.english
        )
        self.assertIn("may also be the terminal outcome in full mode", self.english)
        self.assertIn("`kernel_only_win` 只确认 kernel 收益", self.chinese)
        self.assertIn("也可能是 full 模式的终局结果", self.chinese)
        for text in (self.english, self.chinese):
            self.assertIn("workload failure/loss/inconclusive", text)
            self.assertIn("global best", text)

    def test_readmes_show_executable_installed_skill_command_flow(self) -> None:
        for text in (self.english, self.chinese):
            self.assertIn(
                'cd "${CODEX_HOME:-$HOME/.codex}/skills/cuda-kernel-optimizer"',
                text,
            )
            self.assertIn("python3 scripts/orchestrate.py setup", text)
            self.assertIn("python3 scripts/orchestrate.py open-iter", text)
            self.assertIn("python3 scripts/orchestrate.py close-iter", text)
            self.assertIn("python3 scripts/orchestrate.py resume", text)
            self.assertIn("python3 scripts/orchestrate.py finalize", text)
            self.assertNotIn(
                "python3 skills/cuda-kernel-optimizer/scripts/orchestrate.py", text
            )

            setup = text.index("python3 scripts/orchestrate.py setup")
            opened = text.index("python3 scripts/orchestrate.py open-iter")
            closed = text.index("python3 scripts/orchestrate.py close-iter")
            self.assertLess(setup, opened)
            self.assertLess(opened, closed)

    def test_readmes_document_artifacts_and_resume(self) -> None:
        for text in (self.english, self.chinese):
            self.assertIn("paired_samples.jsonl", text)
            self.assertIn("decision.json", text)
            self.assertIn("checkpoint.json", text)
            self.assertIn("orchestrate.py resume --run-dir", text)

    def test_readmes_record_completed_v2_2_5090_acceptance(self) -> None:
        self.assertIn("V2.2 was validated", self.english)
        self.assertIn("V2.2 已于", self.chinese)
        for text in (self.english, self.chinese):
            self.assertIn("RTX 5090", text)
            self.assertIn("11/11", text)
            self.assertIn("ERR_NVGPUCTRPERM", text)
            self.assertIn("kernel_only_win", text)
            self.assertIn("26.3287%", text)
            self.assertIn("2,232.43", text)
            self.assertIn("140", text)

    def test_readmes_record_completed_v2_4_workload_controller_acceptance(self) -> None:
        self.assertIn("V2.4 workload controller was validated", self.english)
        self.assertIn("V2.4 workload controller 已于", self.chinese)
        for text in (self.english, self.chinese):
            for fact in (
                "13/13",
                "34.302",
                "60.4616%",
                "[60.0894%, 61.4941%]",
                "3 valid pairs",
                "gpu_busy_pct=0.0",
                "inconclusive",
                "reviewer skipped",
                "ERR_NVGPUCTRPERM",
                "sha256:a2d9d89bc4394eab3fadc62c6b5b3f739b6494c1f64c56f5ba5e6c008252a0e5",
            ):
                self.assertIn(fact, text)

    def test_readmes_include_reproducible_fork_installation(self) -> None:
        for text in (self.english, self.chinese):
            self.assertIn("troycheng/cuda-optimized-skill", text)
            self.assertIn("--ref main", text)
            self.assertIn("--path skills/cuda-kernel-optimizer", text)

    def test_readmes_match_conditional_profiler_behavior_and_references(self) -> None:
        self.assertRegex(self.english, r"successful\s+profile with real metrics")
        self.assertRegex(self.chinese, r"真正采集到\s+metrics")
        for text in (self.english, self.chinese):
            self.assertIn("references/compatibility.md", text)

    def test_readmes_use_task_first_information_architecture(self) -> None:
        expected = (
            (
                self.english,
                (
                    "CUDA, CUTLASS, and Triton",
                    "## Start by task",
                    "## Install",
                    "## Start your first run",
                    "## Trusted promotion path",
                    "## Task commands",
                    "## Standalone tool boundaries",
                    "## Inputs, budgets, and statuses",
                    "## Artifacts and resume",
                    "## Compatibility and verification",
                    "## References and license",
                ),
            ),
            (
                self.chinese,
                (
                    "CUDA、CUTLASS 与 Triton",
                    "## 按任务开始",
                    "## 安装",
                    "## 开始第一次运行",
                    "## 可信晋级路径",
                    "## 各任务命令",
                    "## 独立工具边界",
                    "## 输入、预算与状态",
                    "## 产物与恢复",
                    "## 兼容性与验证",
                    "## 参考与许可证",
                ),
            ),
        )
        for text, markers in expected:
            assert_in_order(self, text, markers)
            h1s = [line for line in text.splitlines() if line.startswith("# ")]
            self.assertEqual(h1s, ["# cuda-kernel-optimizer"])

    def test_readmes_offer_the_same_five_task_entries(self) -> None:
        english_tasks = (
            "Optimize a kernel",
            "Optimize a GPU workload",
            "Validate a real workload",
            "Analyze an existing NCU report",
            "Use explicit advisory memory",
        )
        chinese_tasks = (
            "优化 kernel",
            "优化完整 GPU workload",
            "验证真实 workload",
            "分析已有 NCU report",
            "使用显式 advisory memory",
        )
        for marker in english_tasks:
            self.assertIn(marker, self.english)
        for marker in chinese_tasks:
            self.assertIn(marker, self.chinese)

    def test_readmes_have_exactly_three_matching_mermaid_topologies(self) -> None:
        expected = (
            sorted(
                (
                    ("candidate", "-->", "correctness"),
                    ("correctness", "-->", "paired_kernel"),
                    ("paired_kernel", "-->", "sanitizer"),
                    ("sanitizer", "-->", "workload"),
                    ("workload", "-->", "decision"),
                    ("decision", "-->", "promotion"),
                    ("compiler", "-.->", "evidence"),
                    ("sass", "-.->", "evidence"),
                    ("evidence", "-.->", "decision"),
                )
            ),
            sorted(
                (
                    ("report", "-->", "analysis_bundle"),
                    ("completed_run", "-->", "memory"),
                    ("memory", "-.->", "suggestion"),
                )
            ),
            sorted(
                (
                    ("runtime", "-->", "baseline"),
                    ("baseline", "-->", "probes"),
                    ("probes", "-->", "diagnosis"),
                    ("diagnosis", "-->", "change"),
                    ("change", "-->", "review"),
                    ("review", "-->", "evaluation"),
                    ("evaluation", "-->", "workload_decision"),
                    ("workload_decision", "-->", "promote_or_rollback"),
                )
            ),
        )
        english_blocks = MERMAID.findall(self.english)
        chinese_blocks = MERMAID.findall(self.chinese)
        self.assertEqual(len(english_blocks), 3)
        self.assertEqual(len(chinese_blocks), 3)
        for blocks in (english_blocks, chinese_blocks):
            self.assertEqual(tuple(map(mermaid_topology, blocks)), expected)
        self.assertEqual(
            tuple(map(mermaid_topology, english_blocks)),
            tuple(map(mermaid_topology, chinese_blocks)),
        )

    def test_readmes_document_v2_4_workload_controller_and_boundaries(self) -> None:
        for text in (self.english, self.chinese):
            self.assertIn("V2.4", text)
            for command in (
                "workload_controller.py run",
                "workload_controller.py register-change",
                "workload_controller.py evaluate",
                "workload_controller.py resume",
            ):
                self.assertIn(command, text)
            for category in (
                "kernel",
                "framework",
                "cpu_data",
                "transfer",
                "communication",
                "io",
                "environment",
                "mixed",
            ):
                self.assertIn(f"`{category}`", text)
            for marker in (
                "JSON stdin/stdout",
                "isolated_environment",
                "recommend_only",
                "examples/workload-controller.md",
            ):
                self.assertIn(marker, text)
        self.assertIn("Codex remains the primary optimizer", self.english)
        self.assertIn("advisory only", self.english.lower())
        self.assertIn("not an OS sandbox", self.english)
        self.assertIn("Codex 仍是主优化器", self.chinese)
        self.assertIn("只提供审阅意见", self.chinese)
        self.assertIn("不是 OS sandbox", self.chinese)
        self.assertIn("Host changes are recommendations only", self.english)
        self.assertIn("宿主机修改只给建议", self.chinese)

    def test_readmes_document_standalone_cli_surfaces_and_boundaries(self) -> None:
        for text in (self.english, self.chinese):
            for option in (
                "REPORT",
                "--source",
                "--out-dir",
                "--ncu-bin",
                "--ncu-num",
                "--timeout",
            ):
                self.assertIn(option, text)
            for command, options in (
                ("strategy_memory.py record", ("--memory", "--run-dir", "--out")),
                ("strategy_memory.py suggest", ("--memory", "--manifest", "--out")),
            ):
                self.assertIn(command, text)
                for option in options:
                    self.assertIn(option, text)
            self.assertIn("counter_access: not_probed", text)
            self.assertIn("references/serving_evidence_protocol.md", text)
            self.assertIn("references/systems_and_ir_coverage.md", text)

    def test_readmes_publish_identical_current_validation_facts(self) -> None:
        facts = (
            "690",
            "685",
            "28/28",
            "595.71.05",
            "2026.1.1.0",
            "5,966,669",
            "01a1356a487cc1ce77c6af541508db2c5a673dbfa9370bed30d095162321574d",
            "140",
            "6/6",
            "32/32",
            "af1ca2f57081f4420d13662127338906d5b808b52a75f53f18c27787d624359e",
        )
        for fact in facts:
            self.assertIn(fact, self.english)
            self.assertEqual(self.english.count(fact), self.chinese.count(fact))
            self.assertIn(fact, self.chinese)

    def test_readmes_do_not_restore_stale_or_marketing_claims(self) -> None:
        for text in (self.english, self.chinese):
            lower = text.lower()
            for banned in (
                "planned",
                "pending",
                "v2.1",
                "automatic memory",
                "automatic serving",
                "powerful",
                "seamless",
                "revolutionary",
                "comprehensive",
            ):
                self.assertNotIn(banned, lower)
            for banned in ("旨在", "赋能", "无缝", "强大", "全面"):
                self.assertNotIn(banned, text)
        self.assertNotRegex(self.chinese, r"通过[^。\n]{0,80}从而")

    def test_first_run_is_honest_about_stages_and_frozen_runtime(self) -> None:
        expectations = (
            (
                self.english,
                "## Start your first run",
                "## Trusted promotion path",
                "returned `run_dir`",
                "profiles the current best, computes Roofline evidence, and creates the branch directories",
                "prepare the requested branch kernels",
            ),
            (
                self.chinese,
                "## 开始第一次运行",
                "## 可信晋级路径",
                "返回的 `run_dir`",
                "profile 当前 best、计算 Roofline 证据并创建 branch 目录",
                "准备本轮要求的 branch kernel",
            ),
        )
        for text, start, end, run_dir, opened, prepare in expectations:
            section = text[text.index(start) : text.index(end)]
            blocks = re.findall(r"```bash\n(.*?)```", section, re.DOTALL)
            self.assertEqual(len(blocks), 3)
            self.assertIn("orchestrate.py setup", blocks[0])
            self.assertIn("orchestrate.py open-iter", blocks[0])
            self.assertNotIn("close-iter", blocks[0])
            self.assertNotIn("finalize", blocks[0])
            self.assertIn("orchestrate.py close-iter", blocks[1])
            self.assertNotIn("finalize", blocks[1])
            self.assertIn("orchestrate.py finalize", blocks[2])
            self.assertIn(run_dir, section)
            self.assertIn(opened, section)
            self.assertIn(prepare, section)
            self.assertIn("methods.json", section)
            self.assertIn("analysis.md", section)
            self.assertIn("10,800", section)
            self.assertNotIn("admits work", section)
        self.assertNotIn("Five-minute first run", self.english)
        self.assertNotIn("5 分钟首跑", self.chinese)

    def test_runtime_requirements_are_scoped_by_task(self) -> None:
        english = self.english[
            self.english.index("## Compatibility and verification") :
            self.english.index("## References and license")
        ]
        chinese = self.chinese[
            self.chinese.index("## 兼容性与验证") :
            self.chinese.index("## 参考与许可证")
        ]
        for text in (english, chinese):
            self.assertIn("Python 3.10+", text)
            self.assertIn("Strategy memory", text)
            self.assertIn("Darwin/Linux", text)
        for marker in ("CUDA-enabled `torch`", "compatible `ncu`", "launches no GPU target"):
            self.assertIn(marker, english)
        for marker in ("CUDA 版 `torch`", "兼容的 `ncu`", "不启动 GPU target"):
            self.assertIn(marker, chinese)
        self.assertIn("Kernel optimization requires", english)
        self.assertIn("Standalone report analysis requires", english)
        self.assertIn("优化 kernel 需要", chinese)
        self.assertIn("独立 report 分析需要", chinese)
        for text in (english, chinese):
            self.assertNotIn("`ncu` is optional", text)
            self.assertNotIn("`ncu` 是可选依赖", text)


if __name__ == "__main__":
    unittest.main()
