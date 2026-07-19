from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.test_evidence_protocol import (
    _build_attempt,
    _clean_samples,
    _guard_policy,
    _phase_markers,
    _write_json,
)


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "cuda-kernel-optimizer"
EVIDENCE = SKILL / "scripts" / "evidence.py"
SELF_CHECK = SKILL / "scripts" / "self_check.py"
SCHEMAS = (
    "guard_policy.schema.json",
    "experiment_design.schema.json",
    "attempt.schema.json",
    "execution_path.schema.json",
    "serving_experiment.schema.json",
    "artifact_identities.schema.json",
    "profiler_bundle.schema.json",
    "performance_verdict.schema.json",
    "evidence_manifest.schema.json",
)
V2_6_SCHEMAS = (
    "iteration_binding.schema.json",
    "iteration_lineage.schema.json",
    "measurement_path_registry.schema.json",
    "performance_iteration.schema.json",
)
V2_7_SCHEMAS = (
    "direction_portfolio.schema.json",
    "direction_lineage.schema.json",
    "direction_decision.schema.json",
)
V2_8_SCHEMAS = (
    "nonstationarity_anchor.schema.json",
    "nonstationarity_design.schema.json",
    "nonstationarity_series.schema.json",
    "nonstationarity_verdict.schema.json",
)


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    environment = {"PATH": os.environ.get("PATH", "")}
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )


class EvidenceCliTests(unittest.TestCase):
    def test_help_works_without_site_packages(self) -> None:
        result = _run(EVIDENCE, "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in (
            "guard-audit",
            "coverage-audit",
            "seal",
            "audit",
            "decide",
            "audit-imported",
        ):
            self.assertIn(command, result.stdout)

    def test_guard_audit_writes_pass_and_formal_unknown_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "policy.json", _guard_policy())
            _write_json(root / "markers.json", _phase_markers())
            (root / "samples.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in _clean_samples()),
                encoding="utf-8",
            )

            passed = _run(
                EVIDENCE,
                "guard-audit",
                "--policy",
                str(root / "policy.json"),
                "--samples",
                str(root / "samples.jsonl"),
                "--markers",
                str(root / "markers.json"),
                "--out",
                str(root / "pass.json"),
            )
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertEqual(json.loads((root / "pass.json").read_text())["status"], "PASS")

            bad_rows = _clean_samples()
            del bad_rows[5]["memory"]["pressure_pct"]
            (root / "bad-samples.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in bad_rows), encoding="utf-8"
            )
            failed = _run(
                EVIDENCE,
                "guard-audit",
                "--policy",
                str(root / "policy.json"),
                "--samples",
                str(root / "bad-samples.jsonl"),
                "--markers",
                str(root / "markers.json"),
                "--out",
                str(root / "fail.json"),
            )
            self.assertEqual(failed.returncode, 3, failed.stderr)
            self.assertEqual(json.loads((root / "fail.json").read_text())["status"], "FAIL")

    def test_cli_closes_attempt_without_mutating_main_or_external_processes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = _build_attempt(root)
            commands = (
                ("seal", "--attempt", attempt, "--out", root / "seal.json"),
                ("audit", "--seal", root / "seal.json", "--out", root / "audit.json"),
                (
                    "decide",
                    "--seal",
                    root / "seal.json",
                    "--audit",
                    root / "audit.json",
                    "--out",
                    root / "decision.json",
                    "--manifest",
                    root / "evidence-manifest.json",
                ),
            )
            for command in commands:
                result = _run(EVIDENCE, *(str(item) for item in command))
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads((root / "decision.json").read_text())["decision"],
                "promote",
            )


class InstalledSelfCheckTests(unittest.TestCase):
    def test_repository_schemas_are_closed_and_have_v2_5_ids(self) -> None:
        for name in SCHEMAS:
            with self.subTest(name=name):
                payload = json.loads((SKILL / "templates" / name).read_text())
                self.assertIn("v2.5", payload["$id"])
                self.assertEqual(payload["additionalProperties"], False)

    def test_repository_iteration_schemas_are_closed_and_have_v2_6_ids(self) -> None:
        for name in V2_6_SCHEMAS:
            with self.subTest(name=name):
                payload = json.loads((SKILL / "templates" / name).read_text())
                self.assertIn("v2.6", payload["$id"])
                self.assertEqual(payload["additionalProperties"], False)

    def test_repository_direction_schemas_are_closed_and_have_v2_7_ids(self) -> None:
        for name in V2_7_SCHEMAS:
            with self.subTest(name=name):
                payload = json.loads((SKILL / "templates" / name).read_text())
                self.assertIn("v2.7", payload["$id"])
                self.assertEqual(payload["additionalProperties"], False)

    def test_repository_nonstationarity_schemas_are_closed_and_have_v2_8_ids(self) -> None:
        for name in V2_8_SCHEMAS:
            with self.subTest(name=name):
                payload = json.loads((SKILL / "templates" / name).read_text())
                self.assertIn("v2.8", payload["$id"])
                self.assertEqual(payload["additionalProperties"], False)

    def test_self_check_passes_installed_skill_without_gpu_or_network(self) -> None:
        result = _run(SELF_CHECK, "--skill-dir", str(SKILL))
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["gpu_checks_run"], False)
        self.assertEqual(payload["network_checks_run"], False)
        self.assertIn("v2_6_iteration_guard", payload["checks"])
        self.assertIn("v2_7_direction_guard", payload["checks"])
        self.assertIn("v2_8_nonstationarity_guard", payload["checks"])
        self.assertIn("v3_control_runtime", payload["checks"])
        self.assertIn("v3_capability_registry", payload["checks"])

    def test_self_check_fails_closed_for_missing_installation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = _run(SELF_CHECK, "--skill-dir", str(Path(tmp) / "missing"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing", result.stderr.lower())

    def test_self_check_fails_closed_for_corrupt_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            installed = Path(tmp) / "cuda-kernel-optimizer"
            shutil.copytree(SKILL, installed)
            (installed / "templates" / "attempt.schema.json").write_text(
                "{not-json\n", encoding="utf-8"
            )

            result = _run(SELF_CHECK, "--skill-dir", str(installed))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("error", result.stderr.lower())

    def test_self_check_fails_closed_for_tampered_capability_playbook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            installed = Path(tmp) / "cuda-kernel-optimizer"
            shutil.copytree(SKILL, installed)
            playbook = (
                installed
                / "references"
                / "capabilities"
                / "triton-decode-attention-gqa.md"
            )
            playbook.write_text(
                playbook.read_text(encoding="utf-8") + "\ntampered\n",
                encoding="utf-8",
            )

            result = _run(SELF_CHECK, "--skill-dir", str(installed))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("playbook hash mismatch", result.stderr.lower())

    def test_self_check_detects_diagnostic_kind_contract_drift(self) -> None:
        def conditional(schema):
            schema["allOf"][0]["then"]["properties"]["producer"]["properties"]["id"]["const"] = "pytorch-profile-adapter"

        def top_producer(schema):
            schema["properties"]["producer"]["properties"]["id"]["enum"].remove(
                "nsys-timeline-adapter"
            )

        def top_signal(schema):
            schema["properties"]["signals"]["items"]["enum"].remove(
                "cpu_launch_overhead"
            )

        for mutate, message in (
            (conditional, "per-kind"),
            (top_producer, "producer vocabulary"),
            (top_signal, "signal vocabulary"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                installed = Path(tmp) / "cuda-kernel-optimizer"
                shutil.copytree(SKILL, installed)
                schema_path = installed / "templates" / "diagnostic_evidence.schema.json"
                schema = json.loads(schema_path.read_text("utf-8"))
                mutate(schema)
                schema_path.write_text(json.dumps(schema), encoding="utf-8")

                result = _run(SELF_CHECK, "--skill-dir", str(installed))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr.lower())

    def test_self_check_rejects_symlinked_capability_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            installed = base / "cuda-kernel-optimizer"
            shutil.copytree(SKILL, installed)
            capability_root = installed / "references" / "capabilities"
            external = base / "external-capabilities"
            shutil.copytree(capability_root, external)
            shutil.rmtree(capability_root)
            capability_root.symlink_to(external, target_is_directory=True)

            result = _run(SELF_CHECK, "--skill-dir", str(installed))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("symlink", result.stderr.lower())

    def test_self_check_rejects_symlinked_references_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            installed = base / "cuda-kernel-optimizer"
            shutil.copytree(SKILL, installed)
            references = installed / "references"
            external = base / "external-references"
            shutil.copytree(references, external)
            shutil.rmtree(references)
            references.symlink_to(external, target_is_directory=True)

            result = _run(SELF_CHECK, "--skill-dir", str(installed))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("symlink", result.stderr.lower())

    def test_self_check_rejects_symlinked_scripts_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            installed = base / "cuda-kernel-optimizer"
            shutil.copytree(SKILL, installed)
            scripts = installed / "scripts"
            external = base / "external-scripts"
            shutil.copytree(scripts, external)
            shutil.rmtree(scripts)
            scripts.symlink_to(external, target_is_directory=True)

            result = _run(SELF_CHECK, "--skill-dir", str(installed))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("symlink", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
