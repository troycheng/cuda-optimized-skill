from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"
CONTROLLER_PATH = SCRIPTS / "evidence_controller.py"
LEDGER_PATH = SCRIPTS / "evidence_ledger.py"
CONTRACT_SHA = "a" * 64
ENVIRONMENT_SHA = "b" * 64
REFERENCE_SHA = "1" * 64
SEAL_KEY = b"controller-only-secret" * 2


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _adapter(root: Path, payload: dict) -> tuple[Path, str]:
    path = root / "adapter.py"
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "request = json.load(sys.stdin)\n"
        "assert 'controller_seal_key' not in request\n"
        "assert all('SEAL' not in key.upper() for key in os.environ)\n"
        f"print(json.dumps({payload!r}, sort_keys=True))\n",
        encoding="utf-8",
    )
    path.chmod(0o700)
    raw = path.read_bytes()
    return path, hashlib.sha256(raw).hexdigest()


def _measurement() -> dict:
    return {
        "schema_version": "cuda-optimizer/gate-measurement-v1",
        "subject": {"reference_sha256": REFERENCE_SHA},
        "result": {"oracle_sha256": "4" * 64, "cases_total": 8},
        "checks": [
            {"name": "all_reference_cases_passed", "passed": True},
            {"name": "oracle_is_deterministic", "passed": True},
        ],
    }


class EvidenceControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.controller = _load(CONTROLLER_PATH, "cuda_v3_evidence_controller")
        cls.ledger = _load(LEDGER_PATH, "cuda_v3_controller_ledger")

    def _runtime(self, root: Path, adapter: Path, digest: str):
        runtime_path = Path(sys.executable).resolve()
        runtime_sha = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
        implementation_sha = self.controller.adapter_implementation_sha256(
            producer_id="correctness-reference-adapter",
            producer_version="1.0.0",
            entrypoint_sha256=digest,
            runtime_sha256=runtime_sha,
        )
        return self.controller.EvidenceController(
            run_id="run-1",
            ledger_id="ledger-1",
            contract_sha256=CONTRACT_SHA,
            environment_sha256=ENVIRONMENT_SHA,
            ledger_path=root / "ledger",
            artifact_root=root / "sealed",
            controller_seal_key=SEAL_KEY,
            adapters={
                "correctness_reference": {
                    "id": "correctness-reference-adapter",
                    "version": "1.0.0",
                    "path": str(adapter),
                    "entrypoint_sha256": digest,
                    "runtime_path": str(runtime_path),
                    "runtime_sha256": runtime_sha,
                    "implementation_sha256": implementation_sha,
                    "timeout_seconds": 5.0,
                    "max_output_bytes": 4096,
                }
            },
            clock=lambda: 100.0,
        )

    def test_controller_runs_allowlisted_adapter_and_derives_sealed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sealed").mkdir()
            adapter, digest = _adapter(root, _measurement())
            runtime = self._runtime(root, adapter, digest)

            result = runtime.run_and_seal(
                kind="correctness_reference",
                observation_id="obs-reference",
                request={"reference_sha256": REFERENCE_SHA},
            )
            records = self.ledger.verify_ledger(
                root / "ledger", expected_contract_sha256=CONTRACT_SHA
            )
            artifact = json.loads(Path(result["artifact_path"]).read_text("utf-8"))
            implementation_sha = runtime._adapters["correctness_reference"][
                "implementation_sha256"
            ]

        self.assertEqual(result["record"]["event_type"], "observation_sealed")
        self.assertEqual(records[0]["payload"]["run_id"], "run-1")
        self.assertEqual(records[0]["payload"]["ledger_id"], "ledger-1")
        self.assertEqual(
            records[0]["payload"]["adapter_implementation_sha256"], implementation_sha
        )
        self.assertEqual(artifact["kind"], "correctness_reference")
        self.assertEqual(artifact["status"], "PASS")
        self.assertEqual(artifact["recorded_at"], 100.0)
        self.assertEqual(
            artifact["producer"],
            {
                "id": "correctness-reference-adapter",
                "version": "1.0.0",
                "implementation_sha256": implementation_sha,
            },
        )

    def test_controller_rejects_adapter_identity_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sealed").mkdir()
            adapter, digest = _adapter(root, _measurement())
            runtime = self._runtime(root, adapter, digest)
            adapter.write_text(adapter.read_text("utf-8") + "\n# changed\n", "utf-8")

            with self.assertRaisesRegex(ValueError, "implementation|identity|hash"):
                runtime.run_and_seal(
                    kind="correctness_reference",
                    observation_id="obs-reference",
                    request={"reference_sha256": REFERENCE_SHA},
                )

    def test_controller_rejects_source_drift_during_captured_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sealed").mkdir()
            adapter, digest = _adapter(root, _measurement())
            source = adapter.read_text("utf-8").replace(
                "request = json.load(sys.stdin)",
                "request = json.load(sys.stdin); __import__('time').sleep(0.2)",
            )
            adapter.write_text(source, encoding="utf-8")
            adapter.chmod(0o700)
            digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
            runtime = self._runtime(root, adapter, digest)

            def mutate() -> None:
                time.sleep(0.05)
                adapter.write_text(source + "\n# drift\n", encoding="utf-8")

            worker = threading.Thread(target=mutate)
            worker.start()
            try:
                with self.assertRaisesRegex(ValueError, "implementation|identity"):
                    runtime.run_and_seal(
                        kind="correctness_reference",
                        observation_id="obs-reference",
                        request={"reference_sha256": REFERENCE_SHA},
                    )
            finally:
                worker.join()
            self.assertFalse((root / "ledger").exists())

    def test_controller_does_not_load_unbound_local_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sealed").mkdir()
            helper = root / "sealed" / "helper.py"
            helper.write_text("ORACLE = '4' * 64\n", encoding="utf-8")
            payload = _measurement()
            adapter, _digest = _adapter(root, payload)
            source = adapter.read_text("utf-8")
            source = source.replace(
                "import json, os, sys",
                "import json, os, sys\nfrom helper import ORACLE",
            ).replace("'4' * 64", "ORACLE")
            adapter.write_text(source, encoding="utf-8")
            adapter.chmod(0o700)
            digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
            runtime = self._runtime(root, adapter, digest)

            with self.assertRaisesRegex(ValueError, "adapter.*status|execution"):
                runtime.run_and_seal(
                    kind="correctness_reference",
                    observation_id="obs-helper",
                    request={"reference_sha256": REFERENCE_SHA},
                )
            helper.write_text("ORACLE = '5' * 64\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "adapter.*status|execution"):
                runtime.run_and_seal(
                    kind="correctness_reference",
                    observation_id="obs-helper-2",
                    request={"reference_sha256": REFERENCE_SHA},
                )
            self.assertFalse((root / "ledger").exists())

    def test_same_observation_and_request_is_idempotent_but_conflict_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sealed").mkdir()
            adapter, digest = _adapter(root, _measurement())
            ticks = iter((100.0, 101.0, 102.0))
            runtime = self._runtime(root, adapter, digest)
            runtime._clock = lambda: next(ticks)

            first = runtime.run_and_seal(
                kind="correctness_reference",
                observation_id="obs-idempotent",
                request={"reference_sha256": REFERENCE_SHA},
            )
            second = runtime.run_and_seal(
                kind="correctness_reference",
                observation_id="obs-idempotent",
                request={"reference_sha256": REFERENCE_SHA},
            )
            with self.assertRaisesRegex(ValueError, "observation|request|conflict"):
                runtime.run_and_seal(
                    kind="correctness_reference",
                    observation_id="obs-idempotent",
                    request={"reference_sha256": "f" * 64},
                )
            records = self.ledger.verify_ledger(
                root / "ledger", expected_contract_sha256=CONTRACT_SHA
            )

        self.assertEqual(first, second)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["sequence"], 1)

    def test_concurrent_identical_observation_race_returns_one_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sealed").mkdir()
            adapter, digest = _adapter(root, _measurement())
            runtime = self._runtime(root, adapter, digest)
            runtime._clock = lambda: 100.0
            barrier = threading.Barrier(2)
            original_find = runtime._find_existing_observation
            find_lock = threading.Lock()
            calls = 0

            def synchronized_find(*, observation_id):
                nonlocal calls
                with find_lock:
                    calls += 1
                    synchronize = calls <= 2
                if synchronize:
                    barrier.wait()
                return original_find(observation_id=observation_id)

            runtime._find_existing_observation = synchronized_find
            results = []
            errors = []

            def execute() -> None:
                try:
                    results.append(
                        runtime.run_and_seal(
                            kind="correctness_reference",
                            observation_id="obs-race",
                            request={"reference_sha256": REFERENCE_SHA},
                        )
                    )
                except Exception as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            workers = [threading.Thread(target=execute) for _ in range(2)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            records = self.ledger.verify_ledger(
                root / "ledger", expected_contract_sha256=CONTRACT_SHA
            )

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(records), 1)

    def test_concurrent_conflicting_request_cannot_delete_committed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sealed = root / "sealed"
            sealed.mkdir()
            adapter, digest = _adapter(root, _measurement())
            runtime = self._runtime(root, adapter, digest)
            runtime._clock = lambda: 100.0
            barrier = threading.Barrier(2)
            original_find = runtime._find_existing_observation
            find_lock = threading.Lock()
            calls = 0

            def synchronized_find(*, observation_id):
                nonlocal calls
                with find_lock:
                    calls += 1
                    synchronize = calls <= 2
                if synchronize:
                    barrier.wait()
                return original_find(observation_id=observation_id)

            runtime._find_existing_observation = synchronized_find
            results = []
            errors = []

            def execute(reference_sha: str) -> None:
                try:
                    results.append(
                        runtime.run_and_seal(
                            kind="correctness_reference",
                            observation_id="obs-conflict-race",
                            request={"reference_sha256": reference_sha},
                        )
                    )
                except Exception as exc:
                    errors.append(exc)

            workers = [
                threading.Thread(target=execute, args=(REFERENCE_SHA,)),
                threading.Thread(target=execute, args=("f" * 64,)),
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            records = self.ledger.verify_ledger(
                root / "ledger", expected_contract_sha256=CONTRACT_SHA
            )
            referenced = sealed / records[0]["payload"]["artifact"]["path"]
            summary = self.controller._SUMMARY.build_summary(
                root / "ledger",
                artifact_root=sealed,
                run_id="run-1",
                ledger_id="ledger-1",
                contract_sha256=CONTRACT_SHA,
                environment_sha256=ENVIRONMENT_SHA,
                as_of=110.0,
                max_age_seconds=60.0,
                max_observations=4,
                context_budget_bytes=10000,
                controller_seal_key=SEAL_KEY,
            )
            referenced_exists = referenced.is_file()

        self.assertEqual(len(results), 1)
        self.assertEqual(len(errors), 1)
        self.assertRegex(str(errors[0]), "observation|request|conflict")
        self.assertEqual(len(records), 1)
        self.assertTrue(referenced_exists)
        self.assertEqual(len(summary["observations"]), 1)

    def test_controller_recovers_when_ledger_commit_raises_after_publish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sealed").mkdir()
            adapter, digest = _adapter(root, _measurement())
            runtime = self._runtime(root, adapter, digest)
            original = self.controller._SUMMARY._append_controller_gate_observation

            def commit_then_raise(*args, **kwargs):
                original(*args, **kwargs)
                raise OSError("simulated acknowledgement loss")

            with mock.patch.object(
                self.controller._SUMMARY,
                "_append_controller_gate_observation",
                side_effect=commit_then_raise,
            ):
                result = runtime.run_and_seal(
                    kind="correctness_reference",
                    observation_id="obs-committed",
                    request={"reference_sha256": REFERENCE_SHA},
                )

            self.assertTrue(Path(result["artifact_path"]).is_file())
            records = self.ledger.verify_ledger(
                root / "ledger", expected_contract_sha256=CONTRACT_SHA
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["payload"]["observation_id"], "obs-committed")

    def test_controller_enforces_output_and_request_byte_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sealed").mkdir()
            adapter, digest = _adapter(root, _measurement())
            runtime = self._runtime(root, adapter, digest)
            runtime._adapters["correctness_reference"]["max_output_bytes"] = 16
            with self.assertRaisesRegex(ValueError, "status|output|measurement"):
                runtime.run_and_seal(
                    kind="correctness_reference",
                    observation_id="obs-output",
                    request={"reference_sha256": REFERENCE_SHA},
                )

            runtime._adapters["correctness_reference"]["max_output_bytes"] = 4096
            with self.assertRaisesRegex(ValueError, "request.*byte"):
                runtime.run_and_seal(
                    kind="correctness_reference",
                    observation_id="obs-request",
                    request={"padding": "x" * (70 * 1024)},
                )
            self.assertEqual(
                [path.name for path in (root / "sealed").iterdir()],
                [],
                "rejected requests must not leave executable snapshots",
            )

    def test_controller_rejects_self_reported_gate_metadata_and_failed_checks(self) -> None:
        cases = []
        self_reported = _measurement()
        self_reported["status"] = "PASS"
        self_reported["producer"] = {"id": "fake", "version": "9.9.9"}
        cases.append(self_reported)
        failed = _measurement()
        failed["checks"][0]["passed"] = False
        cases.append(failed)

        for index, measurement in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "sealed").mkdir()
                adapter, digest = _adapter(root, measurement)
                runtime = self._runtime(root, adapter, digest)
                with self.assertRaisesRegex(ValueError, "measurement|check|closed|PASS"):
                    runtime.run_and_seal(
                        kind="correctness_reference",
                        observation_id="obs-reference",
                        request={"reference_sha256": REFERENCE_SHA},
                    )
                self.assertFalse((root / "ledger").exists())


if __name__ == "__main__":
    unittest.main()
