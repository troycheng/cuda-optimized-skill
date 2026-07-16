import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README_EN = ROOT / "README.md"
README_ZH = ROOT / "README.zh-CN.md"


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


if __name__ == "__main__":
    unittest.main()
