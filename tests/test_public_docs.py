from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PAGES = (
    "docs/index.md",
    "docs/getting-started.md",
    "docs/environment-readiness.md",
    "docs/workflows.md",
    "docs/long-running-optimization.md",
    "docs/evidence-and-safety.md",
    "docs/compatibility.md",
    "docs/validation.md",
    "docs/case-studies.md",
    "docs/knowledge-and-research.md",
)


def assert_in_order(testcase, text: str, markers: tuple[str, ...]) -> None:
    positions = [text.index(marker) for marker in markers]
    testcase.assertEqual(positions, sorted(positions))


class PublicDocsTests(unittest.TestCase):
    def test_public_navigation_contract(self) -> None:
        config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        assert_in_order(
            self,
            config,
            (
                "index.md",
                "getting-started.md",
                "environment-readiness.md",
                "workflows.md",
                "long-running-optimization.md",
                "evidence-and-safety.md",
                "compatibility.md",
                "validation.md",
                "case-studies.md",
                "knowledge-and-research.md",
                "Agent Protocol",
            ),
        )
        self.assertNotIn("superpowers", config.lower())
        self.assertIn("stylesheets/extra.css", config)
        self.assertIn("https://github.com/troycheng/cuda-optimized-skill", config)

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

    def test_getting_started_defines_inputs_and_installation(self) -> None:
        text = (ROOT / "docs/getting-started.md").read_text(encoding="utf-8")
        for marker in (
            "Codex",
            "skills/cuda-kernel-optimizer",
            "runnable target",
            "correctness reference",
            "performance goal",
            "allowed modification scope",
            "quick",
            "balanced",
            "thorough",
        ):
            self.assertIn(marker, text)
        self.assertIn("must be supplied by the user", text)

    def test_environment_readiness_is_ai_run_and_fail_closed(self) -> None:
        public = (ROOT / "docs/environment-readiness.md").read_text(
            encoding="utf-8"
        )
        agent = (
            ROOT
            / "skills"
            / "cuda-kernel-optimizer"
            / "references"
            / "environment_readiness.md"
        ).read_text(encoding="utf-8")
        public_prose = " ".join(public.split())
        for marker in (
            "The AI runs readiness automatically",
            "user-provided workload",
            "explicit authorization",
            "required",
            "baseline",
            "hash-locked isolated pip",
            "only automatic repair",
            "host changes remain recommendations",
            "self_check",
            "does not prove that the GPU environment is ready",
            "readiness_contract.schema.json",
            "readiness_report.schema.json",
            "ready",
            "degraded",
            "user_action_required",
            "blocked",
        ):
            self.assertIn(marker, public_prose)
        agent_prose = " ".join(agent.split())
        for marker in (
            "Run capability readiness before baseline",
            "required",
            "isolated_pip",
            "recommend_only",
            "readiness_action",
            "Do not ask the user to run these commands manually",
        ):
            self.assertIn(marker, agent_prose)

    def test_workflows_define_four_distinct_claims(self) -> None:
        text = (ROOT / "docs/workflows.md").read_text(encoding="utf-8")
        for marker in (
            "Kernel optimization",
            "Complete workload",
            "Serving validation",
            "Existing NCU report",
            "kernel-level claim",
            "end-to-end claim",
            "read-only",
        ):
            self.assertIn(marker, text)

    def test_long_running_page_explains_contract_controller_and_cadence(self) -> None:
        text = (ROOT / "docs/long-running-optimization.md").read_text(
            encoding="utf-8"
        )
        for marker in (
            "Workload Contract",
            "Controller",
            "Capability Registry",
            "append-only ledger",
            "minimum detectable effect",
            "green",
            "yellow",
            "red",
            "audit_every_candidates",
            "External",
        ):
            self.assertIn(marker, text)

    def test_workflows_keep_ai_iterations_on_performance_work(self) -> None:
        text = (ROOT / "docs/workflows.md").read_text(encoding="utf-8")
        for marker in (
            "Performance-first iteration",
            "falsifiable hypothesis",
            "candidate_evaluated",
            "measurement_blocked",
            "infrastructure_only",
            "performance_iteration.md",
        ):
            self.assertIn(marker, text)
        for marker in (
            "Direction admission",
            "direction_guard.py",
            "same-layer",
            "full-elimination",
            "unrankable",
            "direction_admission.md",
        ):
            self.assertIn(marker, text)

    def test_evidence_page_preserves_formal_boundaries(self) -> None:
        text = (ROOT / "docs/evidence-and-safety.md").read_text(encoding="utf-8")
        for marker in (
            "performance_verdict",
            "evidence_integrity",
            "fail closed",
            "shared-host",
            "c1/c2/c4/c8/c12",
            "self_check",
            "CPU/static",
            "does not validate a GPU environment",
            "never changes host configuration automatically",
        ):
            self.assertIn(marker, text)

    def test_compatibility_routes_to_canonical_reference(self) -> None:
        text = (ROOT / "docs/compatibility.md").read_text(encoding="utf-8")
        for marker in (
            "CUDA",
            "CUTLASS",
            "Triton",
            "Nsight Compute",
            "references/compatibility.md",
            "ERR_NVGPUCTRPERM",
        ):
            self.assertIn(marker, text)

    def test_validation_records_v3_1_readiness_without_speed_claim(self) -> None:
        text = (ROOT / "docs/validation.md").read_text(encoding="utf-8")
        for marker in (
            "18 of 18",
            "52.141",
            "8.793",
            "9.297",
            "Nsys",
            "ERR_NVGPUCTRPERM",
            "not evidence that V3.1 finds a useful direction faster",
        ):
            self.assertIn(marker, text)

    def test_internal_history_is_explicit_and_not_public_protocol(self) -> None:
        text = (ROOT / "maintainers/history/README.md").read_text(encoding="utf-8")
        self.assertIn("internal design and implementation history", text)
        self.assertIn("not the user guide", text)
        self.assertIn("not the agent execution protocol", text)
        self.assertFalse((ROOT / "docs/superpowers").exists())


if __name__ == "__main__":
    unittest.main()
