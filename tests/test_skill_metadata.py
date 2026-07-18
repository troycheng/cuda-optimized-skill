from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / "skills" / "cuda-kernel-optimizer"
SKILL_MD = SKILL_DIR / "SKILL.md"
OPENAI_YAML = SKILL_DIR / "agents" / "openai.yaml"
SERVING_EVIDENCE = SKILL_DIR / "references" / "serving_evidence_protocol.md"
SYSTEMS_IR_COVERAGE = SKILL_DIR / "references" / "systems_and_ir_coverage.md"
EVIDENCE_AUTOMATION = SKILL_DIR / "references" / "evidence_automation.md"
MIGRATION_V2_5 = SKILL_DIR / "references" / "migration_v2_5.md"
PERFORMANCE_ITERATION = SKILL_DIR / "references" / "performance_iteration.md"
DIRECTION_ADMISSION = SKILL_DIR / "references" / "direction_admission.md"
NONSTATIONARY_EVIDENCE = SKILL_DIR / "references" / "nonstationary_serving_evidence.md"
ITERATION_REPORT = SKILL_DIR / "templates" / "iteration_report.md"


class SkillMetadataTests(unittest.TestCase):
    def test_skill_routes_v2_8_nonstationarity_before_direction_admission(self) -> None:
        text = SKILL_MD.read_text(encoding="utf-8")
        for marker in (
            "V2.8",
            "scripts/nonstationarity_guard.py init",
            "scripts/nonstationarity_guard.py check",
            "create-once",
            "comparable_paired_state",
            "inconclusive_nonstationary",
            "site_randomized_balanced",
            "fixed-duration",
            "pair tolerance",
            "phase tolerance",
            "performance_gain_claimed",
            "recommend_only",
            "references/nonstationary_serving_evidence.md",
            "templates/nonstationarity_anchor.schema.json",
            "templates/nonstationarity_design.schema.json",
            "templates/nonstationarity_series.schema.json",
            "templates/nonstationarity_verdict.schema.json",
        ):
            self.assertIn(marker, text)
        self.assertLess(text.index("## Nonstationary serving evidence"), text.index("## Direction admission"))

    def test_nonstationarity_reference_explains_comparability_not_speedup(self) -> None:
        self.assertTrue(NONSTATIONARY_EVIDENCE.is_file())
        text = NONSTATIONARY_EVIDENCE.read_text(encoding="utf-8").lower()
        for marker in (
            "already-collected",
            "create-once",
            "four blocks",
            "ab/ba",
            "fixed-duration",
            "burn-in",
            "pair tolerance",
            "phase tolerance",
            "post-hoc",
            "raw source",
            "storage rollback",
            "comparable_paired_state",
            "inconclusive_nonstationary",
            "does not claim",
            "recommend_only",
        ):
            self.assertIn(marker, text)

    def test_skill_routes_v2_7_direction_admission_before_candidate_iterations(self) -> None:
        text = SKILL_MD.read_text(encoding="utf-8")
        for marker in (
            "V2.7",
            "scripts/direction_guard.py init",
            "scripts/direction_guard.py check",
            "scripts/direction_guard.py status",
            "admit_direction",
            "switch_to_higher_impact",
            "close_direction",
            "direction_closed",
            "unrankable",
            "full-elimination upper bound",
            "never runs the target",
            "host_policy",
            "recommend_only",
            "references/direction_admission.md",
            "templates/direction_portfolio.schema.json",
            "templates/direction_evidence.schema.json",
            "templates/direction_lineage.schema.json",
            "templates/direction_decision.schema.json",
        ):
            self.assertIn(marker, text)
        self.assertLess(text.index("## Direction admission"), text.index("## Performance-first iteration gate"))

    def test_direction_admission_reference_defines_the_machine_boundary(self) -> None:
        self.assertTrue(DIRECTION_ADMISSION.is_file())
        text = DIRECTION_ADMISSION.read_text(encoding="utf-8").lower()
        for marker in (
            "direction family",
            "mechanism",
            "component_metric / baseline_total_metric",
            "same-layer",
            "additive",
            "create-once",
            "hash chain",
            "reopen",
            "new evidence",
            "component artifact",
            "baseline total",
            "expected-tail-sha256",
            "storage rollback",
            "never runs",
            "does not claim",
        ):
            self.assertIn(marker, text)

    def test_skill_routes_v2_6_performance_first_iterations(self) -> None:
        text = SKILL_MD.read_text(encoding="utf-8")
        for marker in (
            "V2.6",
            "performance hypothesis",
            "scripts/iteration_guard.py check",
            "candidate_evaluated",
            "measurement_blocked",
            "infrastructure_only",
            "performance_gain",
            "15%",
            "20 minutes",
            "one infrastructure repair",
            "two consecutive",
            "prevalidated",
            "stop the direction",
            "references/performance_iteration.md",
            "templates/performance_iteration.schema.json",
            "templates/measurement_path_registry.schema.json",
            "templates/iteration_binding.schema.json",
            "templates/iteration_lineage.schema.json",
        ):
            self.assertIn(marker, text)

    def test_performance_iteration_reference_is_minimal_and_machine_verifiable(self) -> None:
        self.assertTrue(PERFORMANCE_ITERATION.is_file())
        text = PERFORMANCE_ITERATION.read_text(encoding="utf-8")
        lower = text.lower()
        for marker in (
            "falsifiable",
            "candidate_evaluated",
            "measurement_blocked",
            "infrastructure_only",
            "performance_gain",
            "anchor_sha256",
            "definition_sha256",
            "v2.5 closure",
            "hash chain",
            "min(1200",
            "one repair",
            "two consecutive",
            "different frozen path",
            "stop_direction",
            "never runs",
            "never emits",
        ):
            self.assertIn(marker, lower)

    def test_iteration_report_leads_with_performance_work_not_tool_chores(self) -> None:
        text = ITERATION_REPORT.read_text(encoding="utf-8")
        markers = (
            "## Performance hypothesis",
            "## Candidate and result",
            "## Decision and next performance action",
            "## Measurement blocker, if any",
        )
        positions = [text.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("not an optimization result", text)
    def test_skill_routes_formal_v2_5_evidence_automation(self) -> None:
        text = SKILL_MD.read_text(encoding="utf-8")
        for marker in (
            "V2.5",
            "scripts/evidence.py guard-audit",
            "scripts/evidence.py seal",
            "scripts/evidence.py audit",
            "scripts/evidence.py decide",
            "scripts/self_check.py",
            "valid",
            "invalid_contaminated",
            "invalid_identity",
            "partial",
            "superseded",
            "evidence_integrity",
            "performance verdict",
            "references/evidence_automation.md",
            "references/migration_v2_5.md",
            "templates/execution_path.schema.json",
            "templates/artifact_identities.schema.json",
            "templates/performance_verdict.schema.json",
            "templates/evidence_manifest.schema.json",
        ):
            self.assertIn(marker, text)

    def test_evidence_reference_is_executable_fail_closed_and_non_promotional(self) -> None:
        self.assertTrue(EVIDENCE_AUTOMATION.is_file())
        text = EVIDENCE_AUTOMATION.read_text(encoding="utf-8")
        lower = text.lower()
        for marker in (
            "target",
            "peer",
            "sibling",
            "cpu",
            "numa",
            "allowlist",
            "swap",
            "memory pressure",
            "clock",
            "temperature",
            "power",
            "thermal",
            "continuous",
            "maximum gap",
            "watcher-ready",
            "correctness",
            "sanitizer",
            "diagnostic",
            "timing",
            "fail closed",
            "execution-path",
            "c1/c2/c4/c8/c12",
            "qps",
            "p95",
            "p99",
            "server input/infer/output",
            "seal -> audit -> decision",
            "read-only imported",
            "non_promotional",
        ):
            self.assertIn(marker, lower)
        self.assertIn("normalized", lower)
        self.assertIn("does not collect", lower)

    def test_v2_5_migration_preserves_v2_4_1_without_rewriting_evidence(self) -> None:
        self.assertTrue(MIGRATION_V2_5.is_file())
        text = MIGRATION_V2_5.read_text(encoding="utf-8")
        for marker in (
            "V2.4.1",
            "legacy_unsealed",
            "no in-place",
            "frozen experiment design",
            "guard",
            "execution-path",
            "artifact identities",
            "seal",
            "audit",
            "decision",
        ):
            self.assertIn(marker, text)

    def test_serving_protocol_routes_to_executable_v2_5_contract(self) -> None:
        text = SERVING_EVIDENCE.read_text(encoding="utf-8")
        for marker in (
            "references/evidence_automation.md",
            "scripts/evidence.py",
            "c1/c2/c4/c8/c12",
            "evidence_integrity",
            "non_promotional",
        ):
            self.assertIn(marker, text)

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
            'short_description: "Evidence-driven GPU kernel and workload optimization"',
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

    def test_skill_documents_real_stage_order_and_open_iter(self) -> None:
        text = SKILL_MD.read_text(encoding="utf-8")
        self.assertIn("orchestrate.py open-iter", text)
        self.assertIn("correctness → paired → sanitizer → SASS", text)
        self.assertIn("not hard promotion gates", text)
        self.assertIn(
            "setup does not profile or create branch directories", text.lower()
        )

    def test_skill_kernel_only_win_is_not_restricted_to_kernel_only_mode(self) -> None:
        text = SKILL_MD.read_text(encoding="utf-8")
        self.assertIn("confirms only a kernel win", text)
        self.assertIn("can also be terminal in full mode", text)
        self.assertIn("workload failure/loss/inconclusive", text)
        self.assertIn("global best", text)
        self.assertNotIn(
            "kernel_only_win is limited to kernel-only mode", text
        )

    def test_skill_uses_real_workload_pair_prefix_path(self) -> None:
        text = SKILL_MD.read_text(encoding="utf-8")
        self.assertIn(
            "workload/<candidate-hash-prefix>/paired_samples.jsonl", text
        )

    def test_skill_documents_budget_default_and_delegates_details(self) -> None:
        text = SKILL_MD.read_text(encoding="utf-8")
        self.assertLessEqual(len(text.splitlines()), 500)
        self.assertIn("balanced", text)
        self.assertIn("default", text.lower())
        self.assertIn("references/compatibility.md", text)
        self.assertIn("references/sanitizer_policy.json", text)
        self.assertIn("templates/objective.schema.json", text)

    def test_skill_exposes_standalone_report_analysis_next_to_profiling(self) -> None:
        text = SKILL_MD.read_text(encoding="utf-8")
        command = (
            "python3 <skill>/scripts/analyze_ncu_rep.py REPORT --out-dir OUTPUT"
        )
        self.assertIn(command, text)
        self.assertIn("existing `.ncu-rep`", text)
        self.assertIn("counter_access: not_probed", text)
        self.assertLess(text.index(command), text.index("## Dual-loop workflow"))
        self.assertIn("python3 <skill>/scripts/analyze_ncu_rep.py --help", text)

    def test_skill_exposes_explicit_advisory_memory_after_finalize(self) -> None:
        text = SKILL_MD.read_text(encoding="utf-8")
        record = (
            "python3 <skill>/scripts/strategy_memory.py record --memory MEMORY "
            "--run-dir RUN_DIR --out OUT"
        )
        suggest = (
            "python3 <skill>/scripts/strategy_memory.py suggest --memory MEMORY "
            "--manifest MANIFEST --out OUT"
        )
        finalize = "python3 <skill>/scripts/orchestrate.py finalize"
        self.assertIn(record, text)
        self.assertIn(suggest, text)
        self.assertLess(text.index(finalize), text.index(record))
        self.assertLess(text.index(record), text.index(suggest))
        self.assertIn("advisory", text.lower())
        self.assertIn("explicit `--memory`", text)
        self.assertIn("no default memory", text.lower())
        self.assertIn("`decision.json` owns promotion", text)
        self.assertIn("python3 <skill>/scripts/strategy_memory.py --help", text)

    def test_skill_has_no_location_dependent_new_cli_commands(self) -> None:
        text = SKILL_MD.read_text(encoding="utf-8")
        self.assertIsNone(
            re.search(
                r"(?m)^python3 scripts/(?:analyze_ncu_rep|strategy_memory)\.py",
                text,
            )
        )

    def test_skill_routes_specialized_evidence_to_on_demand_references(self) -> None:
        text = SKILL_MD.read_text(encoding="utf-8")
        prose = " ".join(text.split())
        self.assertIn("systems-path, CUTLASS/CuTe, or Triton IR", prose)
        self.assertIn("runtime or serving", prose)
        self.assertIn("references/systems_and_ir_coverage.md", prose)
        self.assertIn("references/serving_evidence_protocol.md", prose)

    def test_agent_prompt_discovers_report_and_runtime_evidence_tasks(self) -> None:
        text = OPENAI_YAML.read_text(encoding="utf-8")
        prompt = next(
            line for line in text.splitlines() if line.strip().startswith("default_prompt:")
        ).lower()
        self.assertIn("existing ncu report", prompt)
        self.assertIn("runtime or serving evidence", prompt)
        self.assertIn("user-provided workload", prompt)
        self.assertIn("end-to-end", prompt)
        self.assertIn("formal evidence", prompt)
        self.assertIn("shared-host", prompt)

    def test_skill_documents_v2_4_workload_controller_and_safety_boundaries(self) -> None:
        text = SKILL_MD.read_text(encoding="utf-8")
        lower = text.lower()
        self.assertIn("V2.4", text)
        self.assertIn("scripts/workload_controller.py run", text)
        self.assertIn("scripts/workload_controller.py register-change", text)
        self.assertIn("scripts/workload_controller.py evaluate", text)
        self.assertIn("scripts/workload_controller.py resume", text)
        self.assertIn("user-provided runnable workload", lower)
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
        self.assertIn("Codex is the primary optimizer", text)
        self.assertIn("optional local reviewer", lower)
        self.assertIn("JSON stdin/stdout", text)
        self.assertIn("advisory only", lower)
        self.assertIn("does not provide an OS sandbox", text)
        self.assertIn("project", lower)
        self.assertIn("isolated_environment", text)
        self.assertIn("recommend_only", text)
        self.assertIn("host changes", lower)
        self.assertIn("normalized probe", lower)
        self.assertIn("examples/workload-controller.md", text)

    def test_agent_prompt_routes_full_workload_optimization_to_v2_4(self) -> None:
        text = OPENAI_YAML.read_text(encoding="utf-8").lower()
        for marker in (
            "workload bottleneck",
            "framework",
            "data pipeline",
            "local reviewer",
            "host changes are recommendations only",
        ):
            self.assertIn(marker, text)

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

    def test_serving_reference_defines_claim_ladder_and_evidence_boundary(self) -> None:
        self.assertTrue(SERVING_EVIDENCE.is_file())
        text = SERVING_EVIDENCE.read_text(encoding="utf-8")
        lower = text.lower()
        for layer in (
            "generated code",
            "isolated operator",
            "matched runtime",
            "serving endpoint",
        ):
            self.assertIn(layer, lower)
        for contract in (
            "kernel_only_win",
            "end_to_end_win",
            "user-provided workload",
            "paired A/B",
            "clean window",
            "shared-host contamination",
            "raw request",
            "environment evidence",
        ):
            self.assertIn(contract, text)
        self.assertRegex(
            lower,
            r"generated code[^\n]*(only|does not)[^\n]*(emit|mechanism|serving)",
        )
        self.assertRegex(
            lower,
            r"operator timing[^\n]*(does not|cannot)[^\n]*serving",
        )

    def test_systems_ir_reference_routes_evidence_without_copying_catalog(self) -> None:
        self.assertTrue(SYSTEMS_IR_COVERAGE.is_file())
        text = SYSTEMS_IR_COVERAGE.read_text(encoding="utf-8")
        lower = text.lower()
        for term in (
            "copies",
            "allocation",
            "synchronization",
            "cuda graphs",
            "launch density",
            "cutlass",
            "cute",
            "dispatch",
            "layout",
            "epilogue",
            "cluster",
            "architecture",
            "autotune",
            "ttir",
            "ttgir",
            "llvm",
            "ptx",
            "cache",
            "generated code",
            "sparse",
            "variable-length",
            "fused",
            "serving",
            "real request distribution",
        ):
            self.assertIn(term, lower)
        for reference in (
            "optimization_catalog.md",
            "compatibility.md",
            "serving_evidence_protocol.md",
        ):
            self.assertIn(reference, text)
        self.assertIn("configured direction", lower)
        self.assertNotIn("reduce the target objective", lower)

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
