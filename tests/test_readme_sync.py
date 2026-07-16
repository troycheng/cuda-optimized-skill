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

    def test_readmes_document_artifacts_and_resume(self) -> None:
        for text in (self.english, self.chinese):
            self.assertIn("paired_samples.jsonl", text)
            self.assertIn("decision.json", text)
            self.assertIn("checkpoint.json", text)
            self.assertIn("orchestrate.py resume --run-dir", text)

    def test_readmes_preserve_5090_and_ncu_permission_facts(self) -> None:
        for text in (self.english, self.chinese):
            self.assertIn("RTX 5090", text)
            self.assertIn("ERR_NVGPUCTRPERM", text)
            self.assertIn("3/3", text)

    def test_readmes_include_reproducible_fork_installation(self) -> None:
        for text in (self.english, self.chinese):
            self.assertIn("troycheng/cuda-optimized-skill", text)
            self.assertIn("--ref main", text)
            self.assertIn("--path skills/cuda-kernel-optimizer", text)
            self.assertIn("cd skills/cuda-kernel-optimizer", text)

    def test_readmes_match_conditional_profiler_behavior_and_references(self) -> None:
        self.assertIn("when counter access is available", self.english)
        self.assertIn("counter 可用时", self.chinese)
        for text in (self.english, self.chinese):
            self.assertIn("references/compatibility.md", text)


if __name__ == "__main__":
    unittest.main()
