from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "cuda-kernel-optimizer"


class InformationArchitectureTests(unittest.TestCase):
    def test_docs_are_user_facing_and_history_is_maintainer_only(self) -> None:
        self.assertFalse((ROOT / "docs" / "superpowers").exists())
        self.assertTrue((ROOT / "maintainers" / "history").is_dir())
        for name in (
            "environment-readiness.md",
            "validation.md",
            "case-studies.md",
            "knowledge-and-research.md",
        ):
            self.assertTrue((ROOT / "docs" / name).is_file(), name)

    def test_readmes_separate_validation_from_case_studies(self) -> None:
        english = (ROOT / "README.md").read_text(encoding="utf-8")
        chinese = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
        self.assertNotIn("## Tested scope", english)
        self.assertNotIn("## 已测试范围", chinese)
        self.assertIn("## Validation status", english)
        self.assertIn("## 验证情况", chinese)
        for text in (english, chinese):
            self.assertIn("docs/validation.md", text)
            self.assertIn("docs/case-studies.md", text)

    def test_skill_is_a_small_router_with_on_demand_routes(self) -> None:
        text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        self.assertLessEqual(len(text.splitlines()), 240)
        self.assertLessEqual(len(text.split()), 1800)
        for marker in (
            "scripts/readiness.py",
            "scripts/knowledge_query.py",
            "references/environment_readiness.md",
            "references/research_augmentation.md",
            "references/offline_knowledge.md",
            "references/long_running_control.md",
        ):
            self.assertIn(marker, text)

    def test_offline_knowledge_has_freshness_and_primary_sources(self) -> None:
        manifest = json.loads(
            (SKILL / "references" / "knowledge_sources.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertRegex(manifest["as_of"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertIn("staleness_policy", manifest)
        sources = manifest["sources"]
        self.assertGreaterEqual(len(sources), 8)
        for source in sources:
            self.assertEqual(source["source_kind"], "primary")
            self.assertTrue(source["url"].startswith("https://"))
            self.assertIn("last_verified", source)

    def test_generated_python_artifacts_are_not_part_of_the_skill(self) -> None:
        tracked = subprocess.run(
            ["git", "ls-files", "*.pyc", "**/__pycache__/**"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.splitlines()
        self.assertEqual(tracked, [])


if __name__ == "__main__":
    unittest.main()
