from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "skills" / "cuda-kernel-optimizer"
SCRIPT = (
    SKILL_ROOT / "scripts" / "capability_query.py"
)
CAPABILITY_ROOT = (
    ROOT
    / "skills"
    / "cuda-kernel-optimizer"
    / "references"
    / "capabilities"
)
VALID_GATES = {
    "pre_execution": [
        "correctness_reference",
        "dispatch_identity",
        "target_compile_probe",
    ],
    "promotion": [
        "candidate_correctness",
        "paired_measurement",
        "workload_replay",
    ],
}


def load_module():
    spec = importlib.util.spec_from_file_location("cuda_optimizer_capability", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CapabilityQueryTests(unittest.TestCase):
    def test_real_registry_returns_only_matching_metadata(self) -> None:
        result = load_module().query(
            arch="sm_120",
            task="decode_attention",
            layer="kernel",
            signals=["gqa_head_ratio", "tail_block_error", "kv_gather_dram"],
            available_evidence=["pytorch_profile", "nsys_timeline"],
            framework_versions={"triton": "3.4.0"},
            context_budget_bytes=3000,
            limit=2,
        )

        self.assertEqual(result["registry_variant"], "real")
        self.assertEqual(len(result["capabilities"]), 1)
        capability = result["capabilities"][0]
        self.assertEqual(capability["id"], "triton.decode-attention-gqa")
        self.assertEqual(capability["retrieval_status"], "ready")
        self.assertLessEqual(result["selected_context_bytes"], 3000)
        self.assertIn("playbook_sha256", capability)
        self.assertEqual(
            capability["gate_requirements"],
            {
                "pre_execution": [
                    "correctness_reference",
                    "dispatch_identity",
                    "target_compile_probe",
                ],
                "promotion": [
                    "candidate_correctness",
                    "paired_measurement",
                    "workload_replay",
                ],
            },
        )
        self.assertTrue(capability["contract_binding_required"])
        self.assertNotIn("candidate_admissible", capability)
        self.assertNotIn("steps", capability)
        self.assertNotIn("playbook_body", capability)
        self.assertEqual(result["execution_authority"], "none")
        unsigned = dict(result)
        recorded_query_sha = unsigned.pop("query_sha256")
        expected_query_sha = hashlib.sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(recorded_query_sha, expected_query_sha)
        self.assertEqual(load_module().validate_query_result(result), result)

    def test_unknown_arch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown architecture"):
            load_module().query(
                arch="sm_999",
                task="decode_attention",
                layer="kernel",
                signals=["gqa_head_ratio"],
                available_evidence=[],
                framework_versions={"triton": "3.4.0"},
            )

    def test_counter_signal_prevents_loading_playbook(self) -> None:
        result = load_module().query(
            arch="sm_120",
            task="decode_attention",
            layer="kernel",
            signals=["gqa_head_ratio", "attention_not_material"],
            available_evidence=["pytorch_profile", "nsys_timeline"],
            framework_versions={"triton": "3.4.0"},
        )
        self.assertEqual(result["capabilities"], [])
        self.assertEqual(result["rejected"][0]["reason"], "counter_signal_hit")

    def test_missing_evidence_is_returned_as_collection_work_not_ready_advice(self) -> None:
        result = load_module().query(
            arch="sm_120",
            task="decode_attention",
            layer="kernel",
            signals=["gqa_head_ratio", "kv_gather_dram"],
            available_evidence=[],
            framework_versions={"triton": "3.4.0"},
        )
        capability = result["capabilities"][0]
        self.assertEqual(capability["retrieval_status"], "needs_evidence")
        self.assertEqual(
            capability["missing_evidence"], ["nsys_timeline", "pytorch_profile"]
        )
        self.assertNotIn("candidate_admissible", capability)

    def test_partial_signal_group_does_not_match(self) -> None:
        result = load_module().query(
            arch="sm_120",
            task="decode_attention",
            layer="kernel",
            signals=["gqa_head_ratio"],
            available_evidence=["pytorch_profile", "nsys_timeline"],
            framework_versions={"triton": "3.4.0"},
        )
        self.assertEqual(result["capabilities"], [])
        self.assertEqual(result["rejected"][0]["reason"], "no_complete_signal_group")

    def test_framework_version_mismatch_is_rejected(self) -> None:
        result = load_module().query(
            arch="sm_120",
            task="decode_attention",
            layer="kernel",
            signals=["gqa_head_ratio"],
            available_evidence=["pytorch_profile", "nsys_timeline"],
            framework_versions={"triton": "2.1.0"},
        )
        self.assertEqual(result["capabilities"], [])
        self.assertEqual(result["rejected"][0]["reason"], "framework_version_mismatch")

    def test_context_budget_is_a_hard_limit(self) -> None:
        result = load_module().query(
            arch="sm_120",
            task="decode_attention",
            layer="kernel",
            signals=["gqa_head_ratio", "kv_gather_dram"],
            available_evidence=["pytorch_profile", "nsys_timeline"],
            framework_versions={"triton": "3.4.0"},
            context_budget_bytes=100,
        )
        self.assertEqual(result["capabilities"], [])
        self.assertEqual(result["rejected"][0]["reason"], "context_budget_exceeded")
        self.assertEqual(result["selected_context_bytes"], 0)

    def test_stale_capability_is_downgraded_not_presented_as_current(self) -> None:
        result = load_module().query(
            arch="sm_120",
            task="decode_attention",
            layer="kernel",
            signals=["gqa_head_ratio", "kv_gather_dram"],
            available_evidence=["pytorch_profile", "nsys_timeline"],
            framework_versions={"triton": "3.4.0"},
            as_of="2028-07-19",
            max_review_age_days=365,
        )
        capability = result["capabilities"][0]
        self.assertEqual(capability["knowledge_status"], "unverified_stale")
        self.assertNotIn("candidate_admissible", capability)

    def test_historical_replay_does_not_use_future_capability_or_sources(self) -> None:
        result = load_module().query(
            arch="sm_120",
            task="decode_attention",
            layer="kernel",
            signals=["gqa_head_ratio", "kv_gather_dram"],
            available_evidence=["pytorch_profile", "nsys_timeline"],
            framework_versions={"triton": "3.4.0"},
            as_of="2026-07-18",
        )
        capability = result["capabilities"][0]
        self.assertEqual(capability["knowledge_status"], "unverified_future")
        self.assertTrue(capability["knowledge_time_violation"])

    def test_registry_binds_playbook_and_complete_sources(self) -> None:
        result = load_module().validate_registry()
        self.assertEqual(result["status"], "PASS")
        self.assertGreaterEqual(result["source_count"], 3)
        self.assertEqual(result["capability_count"], 1)
        self.assertIn("sources_sha256", result)

        registry = json.loads(
            (CAPABILITY_ROOT / "registry.json").read_text(encoding="utf-8")
        )
        capability = registry["capabilities"][0]
        playbook = CAPABILITY_ROOT / capability["playbook"]
        self.assertEqual(
            hashlib.sha256(playbook.read_bytes()).hexdigest(),
            capability["playbook_sha256"],
        )

    def test_shuffled_registry_is_eval_only_and_routes_differently(self) -> None:
        module = load_module()
        kwargs = dict(
            arch="sm_120",
            task="decode_attention",
            layer="kernel",
            signals=["gqa_head_ratio", "tail_block_error", "kv_gather_dram"],
            available_evidence=["pytorch_profile", "nsys_timeline"],
            framework_versions={"triton": "3.4.0"},
        )
        with self.assertRaisesRegex(ValueError, "ablation"):
            module.query(**kwargs, registry_variant="shuffled")

        shuffled = module.query(
            **kwargs, registry_variant="shuffled", allow_ablation=True
        )
        self.assertEqual(shuffled["registry_variant"], "shuffled")
        self.assertEqual(shuffled["capabilities"], [])

    def test_query_routes_the_exact_validated_registry_snapshot(self) -> None:
        module = load_module()
        original = module._read_json_snapshot
        with mock.patch.object(
            module, "_read_json_snapshot", wraps=original
        ) as read_snapshot:
            result = module.query(
                arch="sm_120",
                task="decode_attention",
                layer="kernel",
                signals=["tail_block_error"],
                available_evidence=["pytorch_profile", "nsys_timeline"],
                framework_versions={"triton": "3.4.0"},
                context_budget_bytes=3000,
            )
        paths = [Path(call.args[0]) for call in read_snapshot.call_args_list]
        self.assertEqual(paths.count(module.REGISTRY_PATH), 1)
        self.assertEqual(paths.count(module.SOURCES_PATH), 1)
        self.assertIn("sources_sha256", result)

    def test_query_result_replay_rejects_forged_routing_metadata(self) -> None:
        module = load_module()
        result = module.query(
            arch="sm_120",
            task="decode_attention",
            layer="kernel",
            signals=["gqa_head_ratio", "kv_gather_dram"],
            available_evidence=["pytorch_profile", "nsys_timeline"],
            framework_versions={"triton": "3.4.0"},
            as_of="2026-07-19",
        )
        forged = json.loads(json.dumps(result))
        forged["capabilities"][0]["risk"] = "low"
        unsigned = dict(forged)
        unsigned.pop("query_sha256")
        forged["query_sha256"] = hashlib.sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "replay|registry|query"):
            module.validate_query_result(forged)

    def test_tampered_playbook_fails_validation(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            playbook = root / "card.md"
            playbook.write_text("tampered", encoding="utf-8")
            sources = root / "sources.json"
            sources.write_text(
                json.dumps(
                    {
                        "schema_version": "cuda-optimizer/capability-sources-v1",
                        "sources": [
                            {
                                "id": "source.one",
                                "title": "One",
                                "url": "https://example.com",
                                "kind": "documentation",
                                "license": "unknown",
                                "last_reviewed": "2026-07-19",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "schema_version": "cuda-optimizer/capability-registry-v1",
                        "known_architectures": {"sm_120": []},
                        "capabilities": [
                            {
                                "id": "bad.card",
                                "version": "1.0.0",
                                "status": "experimental",
                                "task": "decode_attention",
                                "layer": "kernel",
                                "axes": ["latency"],
                                "architectures": ["sm_120"],
                                "required_features": [],
                                "frameworks": {},
                                "signal_groups": [["signal"]],
                                "counter_signals_any": [],
                                "requires_evidence": [],
                                "gate_requirements": VALID_GATES,
                                "contract_binding_required": True,
                                "conflicts": [],
                                "context_cost_bytes": 8,
                                "playbook": "card.md",
                                "playbook_sha256": "0" * 64,
                                "source_ids": ["source.one"],
                                "last_reviewed": "2026-07-19",
                                "risk": "low",
                                "method_ids": [],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "playbook hash mismatch"):
                module.validate_registry(
                    registry_path=registry,
                    sources_path=sources,
                    capability_root=root,
                )

    def test_declared_context_cost_must_equal_playbook_utf8_bytes(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            playbook = root / "card.md"
            playbook.write_text("x" * 4000, encoding="utf-8")
            sources = root / "sources.json"
            sources.write_text(
                json.dumps(
                    {
                        "schema_version": "cuda-optimizer/capability-sources-v1",
                        "sources": [
                            {
                                "id": "source.one",
                                "title": "One",
                                "url": "https://example.com",
                                "kind": "documentation",
                                "license": "unknown",
                                "last_reviewed": "2026-07-19",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            capability = {
                "id": "bad.cost",
                "version": "1.0.0",
                "status": "experimental",
                "task": "decode_attention",
                "layer": "kernel",
                "axes": ["latency"],
                "architectures": ["sm_120"],
                "required_features": [],
                "frameworks": {},
                "signal_groups": [["signal"]],
                "counter_signals_any": [],
                "requires_evidence": [],
                "gate_requirements": VALID_GATES,
                "contract_binding_required": True,
                "conflicts": [],
                "context_cost_bytes": 10,
                "playbook": "card.md",
                "playbook_sha256": hashlib.sha256(playbook.read_bytes()).hexdigest(),
                "source_ids": ["source.one"],
                "last_reviewed": "2026-07-19",
                "risk": "low",
                "method_ids": [],
            }
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "schema_version": "cuda-optimizer/capability-registry-v1",
                        "known_architectures": {"sm_120": []},
                        "capabilities": [capability],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "context byte cost mismatch"):
                module.validate_registry(
                    registry_path=registry,
                    sources_path=sources,
                    capability_root=root,
                )

    def test_gate_vocabulary_and_minimum_sets_are_closed(self) -> None:
        module = load_module()
        registry = json.loads((CAPABILITY_ROOT / "registry.json").read_text())
        capability = registry["capabilities"][0]
        invalid_gates = [
            {"pre_execution": [], "promotion": VALID_GATES["promotion"]},
            {
                "pre_execution": VALID_GATES["pre_execution"],
                "promotion": ["candidate_correctness", "unknown_gate"],
            },
            {
                "pre_execution": ["paired_measurement"],
                "promotion": VALID_GATES["promotion"],
            },
        ]
        for gates in invalid_gates:
            with self.subTest(gates=gates):
                mutated = dict(capability)
                mutated["gate_requirements"] = gates
                with self.assertRaisesRegex(ValueError, "gate"):
                    module._validate_capability(
                        mutated,
                        0,
                        registry["known_architectures"],
                        set(mutated["source_ids"]),
                        CAPABILITY_ROOT,
                    )

    def test_runtime_validation_matches_closed_schema_constraints(self) -> None:
        module = load_module()
        registry = json.loads((CAPABILITY_ROOT / "registry.json").read_text())
        capability = registry["capabilities"][0]
        invalid_mutations = [
            ("risk", "extreme", "risk"),
            ("version", "1", "version"),
            ("playbook", "card.txt", "playbook"),
            (
                "frameworks",
                {"triton": {"min_inclusive": "4.0.0", "max_exclusive": "3.0.0"}},
                "version range",
            ),
        ]
        for field, value, message in invalid_mutations:
            with self.subTest(field=field):
                mutated = dict(capability)
                mutated[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    module._validate_capability(
                        mutated,
                        0,
                        registry["known_architectures"],
                        set(mutated["source_ids"]),
                        CAPABILITY_ROOT,
                    )

    def test_capability_root_and_manifest_symlinks_are_rejected(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            shutil.copytree(CAPABILITY_ROOT, real)
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                module.validate_registry(
                    registry_path=linked / "registry.json",
                    sources_path=linked / "sources.json",
                    capability_root=linked,
                )

            manifest_link = real / "registry-link.json"
            manifest_link.symlink_to(real / "registry.json")
            with self.assertRaisesRegex(ValueError, "symlink"):
                module.validate_registry(
                    registry_path=manifest_link,
                    sources_path=real / "sources.json",
                    capability_root=real,
                )

    def test_trusted_parent_rejects_intermediate_directory_symlink(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = root / "external"
            shutil.copytree(CAPABILITY_ROOT, external)
            package = root / "package"
            package.mkdir()
            references = package / "references"
            references.symlink_to(root, target_is_directory=True)
            capability_root = references / "external"
            with self.assertRaisesRegex(ValueError, "symlink"):
                module.validate_registry(
                    registry_path=capability_root / "registry.json",
                    sources_path=capability_root / "sources.json",
                    capability_root=capability_root,
                    trusted_root=package,
                )

    def test_validate_cli_does_not_require_query_arguments(self) -> None:
        result = __import__("subprocess").run(
            [__import__("sys").executable, str(SCRIPT), "--validate"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "PASS")

    def test_cli_entrypoints_reject_symlinked_references_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            installed = root / "cuda-kernel-optimizer"
            shutil.copytree(SKILL_ROOT, installed)
            references = installed / "references"
            external = root / "external-references"
            shutil.copytree(references, external)
            shutil.rmtree(references)
            references.symlink_to(external, target_is_directory=True)
            script = installed / "scripts" / "capability_query.py"
            commands = [
                [__import__("sys").executable, str(script), "--validate"],
                [
                    __import__("sys").executable,
                    str(script),
                    "--arch",
                    "sm_120",
                    "--task",
                    "decode_attention",
                    "--layer",
                    "kernel",
                    "--signals",
                    "gqa_head_ratio,kv_gather_dram",
                    "--framework-versions",
                    '{"triton":"3.4.0"}',
                ],
            ]
            for command in commands:
                with self.subTest(command=command[-1]):
                    result = __import__("subprocess").run(
                        command, text=True, capture_output=True, check=False
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("symlink", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
