from __future__ import annotations

import json
import importlib.util
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / "skills" / "cuda-kernel-optimizer"
SKILL_MD = SKILL_DIR / "SKILL.md"
OPENAI_YAML = SKILL_DIR / "agents" / "openai.yaml"


def _load_self_check():
    path = SKILL_DIR / "scripts" / "self_check.py"
    module_name = "cuda_optimizer_self_check_metadata_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class SkillMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = SKILL_MD.read_text(encoding="utf-8")

    def test_frontmatter_is_portable_and_trigger_focused(self) -> None:
        self.assertTrue(self.text.startswith("---\n"))
        frontmatter, _ = self.text[4:].split("\n---\n", 1)
        lines = [line for line in frontmatter.splitlines() if line.strip()]
        self.assertEqual([line.split(":", 1)[0] for line in lines], ["name", "description"])
        self.assertRegex(lines[0].split(":", 1)[1].strip(), r"^[a-z0-9-]+$")
        description = json.loads(lines[1].split(":", 1)[1].strip())
        self.assertTrue(description.startswith("Use when "))
        self.assertLessEqual(len(description), 1024)
        for trigger in ("CUDA", "CUTLASS", "Triton", "PyTorch", "vLLM", "NCU"):
            self.assertIn(trigger, description)

    def test_skill_is_a_router_not_a_version_history(self) -> None:
        self.assertLessEqual(len(self.text.splitlines()), 240)
        self.assertLessEqual(len(self.text.split()), 1800)
        self.assertNotIn("## V2.", self.text)
        self.assertIn("Route before loading details", self.text)
        self.assertIn("Do not load the whole", self.text)

    def test_router_covers_readiness_kernel_workload_serving_and_report_paths(self) -> None:
        for marker in (
            "scripts/readiness.py",
            "references/environment_readiness.md",
            "references/performance_iteration.md",
            "examples/workload-controller.md",
            "references/serving_evidence_protocol.md",
            "references/nonstationary_serving_evidence.md",
            "Existing `.ncu-rep` only",
            "references/ncu_metrics_guide.md",
        ):
            self.assertIn(marker, self.text)

    def test_v3_router_binds_contract_capabilities_and_long_run_control(self) -> None:
        for marker in (
            "scripts/workload_contract.py",
            "scripts/stability_calibration.py",
            "scripts/capability_query.py",
            "scripts/evidence_controller.py",
            "scripts/planner_boundary.py",
            "references/long_running_control.md",
            "audit_every_candidates",
        ):
            self.assertIn(marker, self.text)

    def test_claim_ceiling_requires_user_owned_workload_and_measurement(self) -> None:
        prose = " ".join(self.text.split())
        for marker in (
            "correctness reference",
            "stable kernel benchmark",
            "user-approved real workload",
            "static hypotheses",
            "kernel claim",
            "end-to-end claim",
            "Never download, invent, or silently substitute a workload",
        ):
            self.assertIn(marker, prose)

    def test_budget_has_user_choice_and_balanced_default(self) -> None:
        for marker in ("quick", "balanced", "thorough", "Use `balanced` by default"):
            self.assertIn(marker, self.text)

    def test_workload_diagnosis_is_not_kernel_only(self) -> None:
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
            self.assertIn("`%s`" % category, self.text)
        self.assertIn("do not apply a universal method ranking", self.text)

    def test_knowledge_query_is_bounded_exact_arch_and_offline_capable(self) -> None:
        for marker in (
            "scripts/knowledge_query.py",
            "--limit 5",
            "Exact SM capabilities",
            "never inherit features by numeric ordering",
            "bundled snapshot must work offline",
            "references/offline_knowledge.md",
            "references/optimizer_limits.md",
        ):
            self.assertIn(marker, self.text)

    def test_external_research_is_optional_private_and_non_promotional(self) -> None:
        for marker in (
            "External search and multi-model challenge are optional",
            "references/research_augmentation.md",
            "primary sources",
            "redact private material",
            "preserve disagreement",
            "adjudicate locally",
            "Network or provider failure",
        ):
            self.assertIn(marker, self.text)

    def test_formal_evidence_routes_to_detailed_contracts(self) -> None:
        for marker in (
            "references/evidence_automation.md",
            "references/direction_admission.md",
            "references/performance_iteration.md",
            "references/nonstationary_serving_evidence.md",
            "performance_verdict",
            "evidence_integrity",
            "comparable_paired_state",
            "inconclusive_nonstationary",
            "fail",
        ):
            self.assertIn(marker, self.text)

    def test_v2_5_to_v2_8_reference_contracts_remain_machine_verifiable(self) -> None:
        files_and_markers = {
            "evidence_automation.md": (
                "seal -> audit -> decision",
                "fail closed",
                "execution-path",
                "non_promotional",
            ),
            "performance_iteration.md": (
                "candidate_evaluated",
                "measurement_blocked",
                "infrastructure_only",
                "stop_direction",
            ),
            "direction_admission.md": (
                "full-elimination",
                "create-once",
                "hash chain",
                "never runs",
            ),
            "nonstationary_serving_evidence.md": (
                "create-once",
                "fixed-duration",
                "comparable_paired_state",
                "inconclusive_nonstationary",
            ),
        }
        for name, markers in files_and_markers.items():
            lower = (SKILL_DIR / "references" / name).read_text(encoding="utf-8").lower()
            for marker in markers:
                self.assertIn(marker.lower(), lower, "%s: %s" % (name, marker))

    def test_profiler_degradation_and_host_boundary_are_explicit(self) -> None:
        for marker in (
            "ERR_NVGPUCTRPERM",
            "lower valid evidence layer",
            "does not provide an OS sandbox",
            "recommend_only",
            "driver",
            "GPU counter permission",
        ):
            self.assertIn(marker, self.text)

    def test_durable_output_keeps_decisions_and_raw_evidence(self) -> None:
        for marker in (
            "readiness report and claim ceiling",
            "manifest",
            "checkpoint",
            "raw paired samples",
            "decision.json",
            "evidence integrity",
            "summary.md",
        ):
            self.assertIn(marker, self.text)

    def test_every_skill_reference_path_resolves(self) -> None:
        paths = set(re.findall(r"`((?:references|examples)/[^`]+\.(?:md|json))`", self.text))
        self.assertGreaterEqual(len(paths), 12)
        for relative in paths:
            self.assertTrue((SKILL_DIR / relative).is_file(), relative)

    def test_commands_are_location_independent(self) -> None:
        self.assertIsNone(re.search(r"(?m)^python3 scripts/", self.text))
        for script in ("readiness.py", "knowledge_query.py", "orchestrate.py"):
            self.assertIn("python3 <skill>/scripts/%s" % script, self.text)

    def test_openai_agent_metadata_matches_current_routes(self) -> None:
        text = OPENAI_YAML.read_text(encoding="utf-8")
        self.assertIn('display_name: "CUDA Kernel Optimizer"', text)
        self.assertIn("$cuda-kernel-optimizer", text)
        for marker in (
            "readiness",
            "claim ceiling",
            "workload",
            "architecture-compatible",
            "primary-source search",
            "host changes recommend-only",
        ):
            self.assertIn(marker, text.lower())

    def test_serving_and_systems_references_preserve_cross_layer_coverage(self) -> None:
        serving = (SKILL_DIR / "references" / "serving_evidence_protocol.md").read_text(
            encoding="utf-8"
        ).lower()
        for layer in ("generated code", "isolated operator", "matched runtime", "serving endpoint"):
            self.assertIn(layer, serving)
        systems = (SKILL_DIR / "references" / "systems_and_ir_coverage.md").read_text(
            encoding="utf-8"
        ).lower()
        for term in (
            "copies",
            "synchronization",
            "cuda graphs",
            "cutlass",
            "cute",
            "ttir",
            "ttgir",
            "llvm",
            "ptx",
            "serving",
        ):
            self.assertIn(term, systems)

    def test_skill_artifacts_are_agent_neutral(self) -> None:
        for path in SKILL_DIR.rglob("*"):
            if path.is_file() and path.suffix in {".md", ".json", ".yaml", ".py"}:
                text = path.read_text(encoding="utf-8", errors="ignore")
                self.assertNotIn("Claude Code", text, str(path))

    def test_self_check_reports_static_readiness_admission_coverage(self) -> None:
        result = _load_self_check().check_installation(SKILL_DIR)

        self.assertEqual(result["readiness_contract"], "passed")
        self.assertEqual(result["readiness_probe_schema"], "passed")
        self.assertEqual(result["readiness_report_schema"], "passed")
        self.assertFalse(result["gpu_environment_validated"])
        self.assertIn("v3_1_readiness_admission", result["checks"])


if __name__ == "__main__":
    unittest.main()
