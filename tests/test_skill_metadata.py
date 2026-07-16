from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / "skills" / "cuda-kernel-optimizer"
SKILL_MD = SKILL_DIR / "SKILL.md"
OPENAI_YAML = SKILL_DIR / "agents" / "openai.yaml"


class SkillMetadataTests(unittest.TestCase):
    def test_frontmatter_uses_portable_quoted_scalars(self) -> None:
        text = SKILL_MD.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("---\n"))
        frontmatter, _ = text[4:].split("\n---\n", 1)
        lines = [line for line in frontmatter.splitlines() if line.strip()]
        self.assertEqual([line.split(":", 1)[0] for line in lines], ["name", "description"])

        name = lines[0].split(":", 1)[1].strip()
        description_scalar = lines[1].split(":", 1)[1].strip()
        self.assertRegex(name, r"^[a-z0-9-]+$")
        description = json.loads(description_scalar)
        self.assertTrue(description.startswith("Use when "))
        self.assertLessEqual(len(description), 1024)

    def test_openai_agent_metadata_exists(self) -> None:
        text = OPENAI_YAML.read_text(encoding="utf-8")
        self.assertRegex(text, r'(?m)^interface:\s*$')
        self.assertIn('display_name: "CUDA Kernel Optimizer"', text)
        self.assertIn(
            'short_description: "Trustworthy CUDA kernel and workload optimization"',
            text,
        )
        self.assertIn("$cuda-kernel-optimizer", text)
        prompt = next(
            line for line in text.splitlines() if line.strip().startswith("default_prompt:")
        ).lower()
        self.assertIn("reference", prompt)
        self.assertIn("optional user-provided workload", prompt)

    def test_skill_requires_user_owned_workload_for_end_to_end_claims(self) -> None:
        text = SKILL_MD.read_text(encoding="utf-8")
        self.assertIn("user-provided workload", text)
        self.assertIn("paired", text.lower())
        self.assertIn("inconclusive", text)
        self.assertIn("ERR_NVGPUCTRPERM", text)
        self.assertIn("kernel_only_win", text)
        self.assertIn("end_to_end_win", text)

    def test_skill_documents_budget_default_and_delegates_details(self) -> None:
        text = SKILL_MD.read_text(encoding="utf-8")
        self.assertLessEqual(len(text.splitlines()), 500)
        self.assertIn("balanced", text)
        self.assertIn("default", text.lower())
        self.assertIn("references/compatibility.md", text)
        self.assertIn("references/sanitizer_policy.json", text)
        self.assertIn("templates/objective.schema.json", text)

    def test_skill_output_contract_contains_durable_evidence(self) -> None:
        text = SKILL_MD.read_text(encoding="utf-8")
        for artifact in (
            "manifest.json",
            "checkpoint.json",
            "paired_samples.jsonl",
            "decision.json",
            "summary.md",
        ):
            self.assertIn(artifact, text)

    def test_skill_artifacts_are_agent_neutral(self) -> None:
        suffixes = {".md", ".py", ".json", ".yaml", ".yml"}
        text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(SKILL_DIR.rglob("*"))
            if path.is_file() and path.suffix in suffixes
        )
        for pattern in (r"\bClaude\b", r"Chain-of-Thought", r"\(CoT\)"):
            self.assertIsNone(re.search(pattern, text, flags=re.IGNORECASE), pattern)


if __name__ == "__main__":
    unittest.main()
