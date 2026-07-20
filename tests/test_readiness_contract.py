import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "cuda-kernel-optimizer"
MODULE_PATH = SKILL / "scripts" / "readiness_contract.py"
TEMPLATES = SKILL / "templates"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "cuda_readiness_contract", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _contract(project: Path, environment: Path) -> dict:
    python = environment / "bin" / "python3"
    requirements = project / "requirements-gpu.lock"
    return {
        "schema_version": "cuda-workload-optimizer/readiness-contract-v1",
        "requested_claim": "workload",
        "budget": {"max_seconds": 300, "max_repairs": 1},
        "requirements": [
            {
                "id": "gpu-execute",
                "necessity": "required",
                "control_scope": "isolated_environment",
                "phase": "foundation",
                "kind": "gpu_execute",
                "max_age_seconds": 300,
                "probe": {
                    "argv": [str(python), "tools/gpu_smoke.py"],
                    "timeout_seconds": 30,
                },
                "remediation": {
                    "mode": "isolated_pip",
                    "authorization_id": "user-approved-env-20260720",
                    "python": str(python),
                    "requirements_file": str(requirements),
                    "requirements_sha256": "a" * 64,
                    "timeout_seconds": 180,
                },
            }
        ],
    }


class ReadinessContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def _fixture(self, root: Path) -> tuple[Path, Path, dict]:
        project = root / "project"
        environment = root / "env"
        (environment / "bin").mkdir(parents=True)
        project.mkdir()
        python = environment / "bin" / "python3"
        python.write_text("#!/bin/sh\nexit 0\n", "utf-8")
        python.chmod(0o755)
        (project / "requirements-gpu.lock").write_text(
            "demo==1.0 --hash=sha256:" + "b" * 64 + "\n", "utf-8"
        )
        return project, environment, _contract(project, environment)

    def test_valid_contract_is_detached_and_digest_is_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project, environment, value = self._fixture(Path(tmp))
            validated = self.module.validate_contract(
                value, project_root=project, environment_root=environment
            )
            digest = self.module.contract_digest(validated)
            value["requirements"][0]["id"] = "mutated"

        self.assertEqual(validated["requirements"][0]["id"], "gpu-execute")
        self.assertEqual(len(digest), 64)
        reordered = json.loads(json.dumps(validated, sort_keys=True))
        self.assertEqual(digest, self.module.contract_digest(reordered))

        changed_version = json.loads(json.dumps(validated))
        changed_version["schema_version"] = "cuda-workload-optimizer/readiness-contract-v2"
        self.assertNotEqual(digest, self.module.contract_digest(changed_version))

    def test_load_contract_rejects_duplicate_nonfinite_and_symlink_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            duplicate = root / "duplicate.json"
            duplicate.write_text(
                '{"schema_version":"a","schema_version":"b"}', "utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self.module.load_contract(duplicate)

            nonfinite = root / "nonfinite.json"
            nonfinite.write_text('{"value":NaN}', "utf-8")
            with self.assertRaisesRegex(ValueError, "finite"):
                self.module.load_contract(nonfinite)

            linked = root / "linked.json"
            linked.symlink_to(duplicate)
            with self.assertRaisesRegex(ValueError, "unsafe|symlink"):
                self.module.load_contract(linked)

    def test_closed_contract_rejects_unknown_and_missing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project, environment, value = self._fixture(Path(tmp))
            value["surprise"] = True
            with self.assertRaisesRegex(ValueError, "unknown"):
                self.module.validate_contract(
                    value, project_root=project, environment_root=environment
                )
            value.pop("surprise")
            value.pop("budget")
            with self.assertRaisesRegex(ValueError, "missing.*budget|budget.*missing"):
                self.module.validate_contract(
                    value, project_root=project, environment_root=environment
                )

    def test_duplicate_requirement_ids_and_invalid_enums_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project, environment, value = self._fixture(Path(tmp))
            value["requirements"].append(
                json.loads(json.dumps(value["requirements"][0]))
            )
            with self.assertRaisesRegex(ValueError, "duplicate.*id"):
                self.module.validate_contract(
                    value, project_root=project, environment_root=environment
                )

            value["requirements"].pop()
            for field, invalid in (
                ("necessity", "nice_to_have"),
                ("control_scope", "global"),
                ("phase", "later"),
                ("kind", "query_metrics_only"),
            ):
                with self.subTest(field=field):
                    value["requirements"][0][field] = invalid
                    with self.assertRaisesRegex(ValueError, field):
                        self.module.validate_contract(
                            value,
                            project_root=project,
                            environment_root=environment,
                        )
                    value = _contract(project, environment)

    def test_isolated_authorization_ids_cannot_be_replayed_in_one_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project, environment, value = self._fixture(Path(tmp))
            second = json.loads(json.dumps(value["requirements"][0]))
            second["id"] = "benchmark-noise"
            second["kind"] = "benchmark_noise"
            value["requirements"].append(second)
            with self.assertRaisesRegex(ValueError, "duplicate authorization_id"):
                self.module.validate_contract(
                    value, project_root=project, environment_root=environment
                )

    def test_numeric_limits_and_empty_argv_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project, environment, value = self._fixture(Path(tmp))
            cases = (
                (("budget", "max_seconds"), float("nan"), "finite"),
                (("budget", "max_repairs"), -1, "max_repairs"),
                (("requirement", "max_age_seconds"), 0, "max_age_seconds"),
                (("probe", "timeout_seconds"), 0, "timeout_seconds"),
                (("probe", "argv"), [], "argv"),
            )
            for path, invalid, message in cases:
                with self.subTest(path=path):
                    changed = json.loads(json.dumps(value))
                    if path[0] == "budget":
                        changed["budget"][path[1]] = invalid
                    elif path[0] == "requirement":
                        changed["requirements"][0][path[1]] = invalid
                    else:
                        changed["requirements"][0]["probe"][path[1]] = invalid
                    if isinstance(invalid, float):
                        changed["budget"]["max_seconds"] = invalid
                    with self.assertRaisesRegex(ValueError, message):
                        self.module.validate_contract(
                            changed,
                            project_root=project,
                            environment_root=environment,
                        )

    def test_host_requirement_cannot_auto_install(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project, environment, value = self._fixture(Path(tmp))
            value["requirements"][0]["control_scope"] = "host"
            with self.assertRaisesRegex(ValueError, "host.*user_action"):
                self.module.validate_contract(
                    value, project_root=project, environment_root=environment
                )

    def test_isolated_pip_paths_must_be_absolute_contained_regular_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, environment, value = self._fixture(root)
            remediation = value["requirements"][0]["remediation"]

            remediation["python"] = "env/bin/python3"
            with self.assertRaisesRegex(ValueError, "python.*absolute"):
                self.module.validate_contract(
                    value, project_root=project, environment_root=environment
                )

            value = _contract(project, environment)
            outside = root / "outside.lock"
            outside.write_text("outside\n", "utf-8")
            value["requirements"][0]["remediation"][
                "requirements_file"
            ] = str(outside)
            with self.assertRaisesRegex(ValueError, "requirements_file.*project"):
                self.module.validate_contract(
                    value, project_root=project, environment_root=environment
                )

            value = _contract(project, environment)
            real = project / "requirements-gpu.lock"
            linked = project / "linked.lock"
            linked.symlink_to(real)
            value["requirements"][0]["remediation"][
                "requirements_file"
            ] = str(linked)
            with self.assertRaisesRegex(ValueError, "requirements_file.*symlink|unsafe"):
                self.module.validate_contract(
                    value, project_root=project, environment_root=environment
                )

    def test_isolated_python_accepts_a_bounded_venv_leaf_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, environment, value = self._fixture(root)
            python = environment / "bin" / "python3"
            python.unlink()
            real_python = root / "system-python"
            real_python.write_text("#!/bin/sh\nexit 0\n", "utf-8")
            real_python.chmod(0o755)
            python.symlink_to(real_python)

            validated = self.module.validate_contract(
                value, project_root=project, environment_root=environment
            )

            self.assertEqual(
                validated["requirements"][0]["remediation"]["python"],
                str(python),
            )

    def test_remediation_variants_are_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project, environment, value = self._fixture(Path(tmp))
            requirement = value["requirements"][0]
            requirement["control_scope"] = "host"
            requirement["remediation"] = {
                "mode": "user_action",
                "message": "Enable one visible target GPU.",
                "command": "sudo chmod 777 /dev/nvidia0",
            }
            with self.assertRaisesRegex(ValueError, "unknown.*command"):
                self.module.validate_contract(
                    value, project_root=project, environment_root=environment
                )

            requirement["remediation"].pop("command")
            validated = self.module.validate_contract(
                value, project_root=project, environment_root=environment
            )
            self.assertEqual(
                validated["requirements"][0]["remediation"]["mode"],
                "user_action",
            )

    def test_runtime_and_schema_protocols_match(self) -> None:
        schemas = {
            name: json.loads((TEMPLATES / name).read_text("utf-8"))
            for name in (
                "readiness_contract.schema.json",
                "readiness_probe.schema.json",
                "readiness_report.schema.json",
            )
        }
        contract = schemas["readiness_contract.schema.json"]
        self.assertFalse(contract["additionalProperties"])
        self.assertEqual(
            set(contract["properties"]["requested_claim"]["enum"]),
            self.module.REQUESTED_CLAIMS,
        )
        requirement = contract["$defs"]["requirement"]
        self.assertFalse(requirement["additionalProperties"])
        self.assertEqual(
            set(requirement["properties"]["kind"]["enum"]), self.module.KINDS
        )
        for name, schema in schemas.items():
            with self.subTest(schema=name):
                self.assertEqual(
                    schema["$schema"],
                    "https://json-schema.org/draft/2020-12/schema",
                )
                self.assertFalse(schema["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
