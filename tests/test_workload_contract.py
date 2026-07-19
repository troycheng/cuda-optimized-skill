import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "skills"
    / "cuda-kernel-optimizer"
    / "scripts"
    / "workload_contract.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("cuda_workload_contract", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _draft(project: Path, environment: Path) -> dict:
    return {
        "schema_version": "cuda-optimizer/workload-contract-draft-v1",
        "run_id": "sm120-attention-001",
        "requested_claim": "workload",
        "project_root": str(project),
        "artifacts": [
            {"role": "workload_manifest", "path": "workload.json"},
            {"role": "correctness_reference", "path": "reference.py"},
        ],
        "workload": {
            "argv": ["python3", "workload.py"],
            "input_distribution": "production-shape-snapshot-2026-07-19",
            "representative_cases": ["prefill-1k", "decode-128"],
        },
        "objective": {
            "metric": "request_latency",
            "unit": "ms",
            "direction": "lower",
            "aggregation": "median",
            "minimum_practical_effect_pct": 1.0,
            "constraints": ["exact output within declared tolerance"],
        },
        "budget": {
            "preset": "balanced",
            "max_seconds": 10800,
            "max_candidates": 24,
        },
        "mutation": {
            "project_paths": ["kernels"],
            "environment_root": str(environment),
            "host_policy": "recommend_only",
        },
        "evidence": {"max_age_seconds": 1800},
    }


class WorkloadContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = _load_module()

    def _fixture(self, root: Path) -> tuple[Path, Path, dict]:
        project = root / "project"
        project.mkdir()
        (project / "kernels").mkdir()
        (project / "workload.json").write_text('{"name":"demo"}\n', "utf-8")
        (project / "reference.py").write_text("def reference(x): return x\n", "utf-8")
        environment = root / "env"
        environment.mkdir()
        return project, environment, _draft(project, environment)

    def test_freeze_binds_files_and_produces_stable_contract_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, _environment, draft = self._fixture(root)
            out = root / "contract.json"

            frozen = self.contract.freeze_contract(draft, out)
            verified = self.contract.verify_frozen_contract(out)

            self.assertEqual(
                frozen["schema_version"], "cuda-optimizer/workload-contract-v1"
            )
            self.assertEqual(frozen, verified)
            self.assertEqual(
                [item["role"] for item in frozen["artifacts"]],
                ["workload_manifest", "correctness_reference"],
            )
            expected = hashlib.sha256((project / "workload.json").read_bytes()).hexdigest()
            self.assertEqual(frozen["artifacts"][0]["sha256"], expected)
            self.assertEqual(len(frozen["contract_sha256"]), 64)

    def test_frozen_contract_is_create_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _project, _environment, draft = self._fixture(root)
            out = root / "contract.json"
            self.contract.freeze_contract(draft, out)
            with self.assertRaisesRegex(ValueError, "exists|create"):
                self.contract.freeze_contract(draft, out)

    def test_verify_rejects_artifact_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, _environment, draft = self._fixture(root)
            out = root / "contract.json"
            self.contract.freeze_contract(draft, out)
            (project / "workload.json").write_text('{"name":"changed"}\n', "utf-8")
            with self.assertRaisesRegex(ValueError, "changed|sha256|identity"):
                self.contract.verify_frozen_contract(out)

    def test_rejects_unknown_duplicate_and_nonfinite_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _project, _environment, draft = self._fixture(root)
            with self.subTest("unknown"):
                draft["surprise"] = True
                with self.assertRaisesRegex(ValueError, "unknown"):
                    self.contract.validate_draft(draft)
                draft.pop("surprise")

            with self.subTest("duplicate"):
                source = root / "duplicate.json"
                source.write_text(
                    '{"schema_version":"cuda-optimizer/workload-contract-draft-v1",'
                    '"schema_version":"again"}',
                    "utf-8",
                )
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    self.contract.load_json_strict(source)

            with self.subTest("nonfinite"):
                draft["budget"]["max_seconds"] = float("nan")
                with self.assertRaisesRegex(ValueError, "finite"):
                    self.contract.validate_draft(draft)

    def test_rejects_symlink_artifact_and_unsafe_host_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, _environment, draft = self._fixture(root)
            (project / "linked.json").symlink_to(project / "workload.json")
            draft["artifacts"][0]["path"] = "linked.json"
            with self.assertRaisesRegex(ValueError, "symlink|unsafe"):
                self.contract.freeze_contract(draft, root / "contract.json")

            draft["artifacts"][0]["path"] = "workload.json"
            draft["mutation"]["host_policy"] = "auto_tune"
            with self.assertRaisesRegex(ValueError, "recommend_only"):
                self.contract.validate_draft(draft)

    def test_rejects_symlinked_mutation_and_environment_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, environment, draft = self._fixture(root)
            outside = root / "outside"
            outside.mkdir()
            (project / "linked-kernels").symlink_to(outside, target_is_directory=True)
            draft["mutation"]["project_paths"] = ["linked-kernels"]
            with self.assertRaisesRegex(ValueError, "symlink|unsafe|project_paths"):
                self.contract.validate_draft(draft)

            draft["mutation"]["project_paths"] = ["kernels"]
            linked_environment = root / "linked-env"
            linked_environment.symlink_to(environment, target_is_directory=True)
            draft["mutation"]["environment_root"] = str(linked_environment)
            with self.assertRaisesRegex(ValueError, "symlink|unsafe|environment_root"):
                self.contract.validate_draft(draft)

    def test_rejects_claim_objective_and_mutation_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _project, _environment, draft = self._fixture(root)
            cases = (
                ("requested_claim", "unknown", "requested_claim"),
                ("direction", "sideways", "direction"),
                ("project_paths", ["../outside"], "project_paths"),
            )
            for field, value, message in cases:
                changed = json.loads(json.dumps(draft))
                if field == "direction":
                    changed["objective"][field] = value
                elif field == "project_paths":
                    changed["mutation"][field] = value
                else:
                    changed[field] = value
                with self.subTest(field), self.assertRaisesRegex(ValueError, message):
                    self.contract.validate_draft(changed)


if __name__ == "__main__":
    unittest.main()
