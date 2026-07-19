import importlib.util
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "skills"
    / "cuda-kernel-optimizer"
    / "scripts"
    / "evidence_ledger.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("cuda_evidence_ledger_v3", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EvidenceLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ledger = _load_module()

    def test_append_and_verify_hash_chained_create_once_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger"
            first = self.ledger.append_event(
                path,
                event_type="contract_frozen",
                contract_sha256="a" * 64,
                payload={"run_id": "demo"},
            )
            second = self.ledger.append_event(
                path,
                event_type="candidate_registered",
                contract_sha256="a" * 64,
                payload={"candidate_id": "c1"},
            )
            records = self.ledger.verify_ledger(path, expected_contract_sha256="a" * 64)

            self.assertEqual([item["sequence"] for item in records], [1, 2])
            self.assertEqual(first["previous_sha256"], "0" * 64)
            self.assertEqual(second["previous_sha256"], first["record_sha256"])
            self.assertEqual(records, [first, second])
            self.assertTrue((path / "00000000000000000001.json").is_file())

    def test_tampering_or_gap_breaks_verification_and_blocks_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger"
            self.ledger.append_event(
                path,
                event_type="contract_frozen",
                contract_sha256="b" * 64,
                payload={"value": 1},
            )
            self.ledger.append_event(
                path,
                event_type="baseline_ready",
                contract_sha256="b" * 64,
                payload={"value": 2},
            )
            event = path / "00000000000000000001.json"
            changed = json.loads(event.read_text("utf-8"))
            changed["payload"]["value"] = 9
            event.write_text(json.dumps(changed), "utf-8")
            with self.assertRaisesRegex(ValueError, "hash|changed|chain"):
                self.ledger.verify_ledger(path)
            with self.assertRaisesRegex(ValueError, "hash|changed|chain"):
                self.ledger.append_event(
                    path,
                    event_type="candidate_registered",
                    contract_sha256="b" * 64,
                    payload={},
                )

            event.write_text(json.dumps(self.ledger.seal_record(changed | {"payload": {"value": 1}})), "utf-8")
            (path / "00000000000000000002.json").unlink()
            (path / "00000000000000000003.json").write_text("{}", "utf-8")
            with self.assertRaisesRegex(ValueError, "sequence|gap"):
                self.ledger.verify_ledger(path)

    def test_concurrent_appends_form_one_unbroken_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger"
            errors = []

            def append(index: int) -> None:
                try:
                    self.ledger.append_event(
                        path,
                        event_type="observation",
                        contract_sha256="c" * 64,
                        payload={"index": index},
                    )
                except Exception as error:  # pragma: no cover - asserted below
                    errors.append(error)

            threads = [threading.Thread(target=append, args=(index,)) for index in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            records = self.ledger.verify_ledger(path, expected_contract_sha256="c" * 64)
            self.assertEqual(len(records), 12)
            self.assertEqual({item["payload"]["index"] for item in records}, set(range(12)))

    def test_compare_and_append_rejects_a_stale_controller_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger"
            first = self.ledger.append_event(
                path,
                event_type="run_initialized",
                contract_sha256="f" * 64,
                payload={"state": "INIT"},
                expected_previous_sha256="0" * 64,
            )
            self.ledger.append_event(
                path,
                event_type="state_transition",
                contract_sha256="f" * 64,
                payload={"state": "FROZEN"},
                expected_previous_sha256=first["record_sha256"],
            )
            with self.assertRaisesRegex(ValueError, "stale|previous"):
                self.ledger.append_event(
                    path,
                    event_type="candidate_registered",
                    contract_sha256="f" * 64,
                    payload={"candidate": "c1"},
                    expected_previous_sha256=first["record_sha256"],
                )

    def test_process_death_during_write_leaves_no_partial_event_and_can_resume(self) -> None:
        if not hasattr(os, "fork"):
            self.skipTest("fork is unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger"
            first = self.ledger.append_event(
                path,
                event_type="run_initialized",
                contract_sha256="1" * 64,
                payload={"state": "INIT"},
            )
            pid = os.fork()
            if pid == 0:  # pragma: no cover - child exits without unittest cleanup
                real_write = self.ledger.os.write

                def crash_after_partial_write(fd, payload):
                    real_write(fd, payload[:17])
                    os._exit(91)

                self.ledger.os.write = crash_after_partial_write
                self.ledger.append_event(
                    path,
                    event_type="state_transition",
                    contract_sha256="1" * 64,
                    payload={"state": "FROZEN"},
                    expected_previous_sha256=first["record_sha256"],
                )
                os._exit(92)
            waited, status = os.waitpid(pid, 0)
            self.assertEqual(waited, pid)
            self.assertEqual(os.waitstatus_to_exitcode(status), 91)

            errors = []
            snapshots = []
            barrier = threading.Barrier(16)

            def verify_after_crash() -> None:
                try:
                    barrier.wait()
                    snapshots.append(self.ledger.verify_ledger(path))
                except Exception as error:  # pragma: no cover - asserted below
                    errors.append(error)

            verifiers = [threading.Thread(target=verify_after_crash) for _ in range(16)]
            for verifier in verifiers:
                verifier.start()
            for verifier in verifiers:
                verifier.join()
            self.assertEqual(errors, [])
            self.assertEqual(snapshots, [[first]] * 16)
            resumed = self.ledger.append_event(
                path,
                event_type="state_transition",
                contract_sha256="1" * 64,
                payload={"state": "FROZEN"},
                expected_previous_sha256=first["record_sha256"],
            )
            self.assertEqual(resumed["sequence"], 2)
            self.assertEqual(len(self.ledger.verify_ledger(path)), 2)

    def test_rejects_nonfinite_payload_contract_mismatch_and_unsafe_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "ledger"
            with self.assertRaisesRegex(ValueError, "finite"):
                self.ledger.append_event(
                    path,
                    event_type="observation",
                    contract_sha256="d" * 64,
                    payload={"metric": float("nan")},
                )
            self.ledger.append_event(
                path,
                event_type="contract_frozen",
                contract_sha256="d" * 64,
                payload={},
            )
            with self.assertRaisesRegex(ValueError, "contract"):
                self.ledger.append_event(
                    path,
                    event_type="observation",
                    contract_sha256="e" * 64,
                    payload={},
                )

            real = root / "real"
            real.mkdir()
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink|unsafe"):
                self.ledger.append_event(
                    linked,
                    event_type="observation",
                    contract_sha256="d" * 64,
                    payload={},
                )

    def test_reserved_observation_id_is_unique_under_the_ledger_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger"
            first = self.ledger._append_reserved_event(
                path,
                event_type="observation_sealed",
                contract_sha256="a" * 64,
                payload={"observation_id": "obs-1", "value": 1},
            )
            with self.assertRaisesRegex(ValueError, "observation_id|duplicate|unique"):
                self.ledger._append_reserved_event(
                    path,
                    event_type="observation_sealed",
                    contract_sha256="a" * 64,
                    payload={"observation_id": "obs-1", "value": 2},
                )
            self.assertEqual(self.ledger.verify_ledger(path), [first])


if __name__ == "__main__":
    unittest.main()
