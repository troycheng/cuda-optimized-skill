import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README_EN = ROOT / "README.md"
README_ZH = ROOT / "README.zh-CN.md"


class ReadmeSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.english = README_EN.read_text(encoding="utf-8")
        self.chinese = README_ZH.read_text(encoding="utf-8")

    def test_readmes_identify_the_v2_1_release_and_five_mechanisms(self) -> None:
        self.assertIn("V2.1", self.english)
        self.assertIn("five mechanisms", self.english.lower())
        self.assertIn("V2.1", self.chinese)
        self.assertIn("五个机制", self.chinese)

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
