from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_STORE_PATH = (
    ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "artifact_store.py"
)


def _load_artifact_store():
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_artifact_store", ARTIFACT_STORE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ArtifactStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.artifacts = _load_artifact_store()

    def _inputs(self, root: Path) -> tuple[Path, Path]:
        baseline = root / "baseline.py"
        ref = root / "ref.py"
        baseline.write_text("baseline\n", encoding="utf-8")
        ref.write_text("reference\n", encoding="utf-8")
        return baseline, ref

    def test_initialize_writes_v2_manifest_with_stable_hash_and_no_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            baseline, ref = self._inputs(base)
            inputs = {"ref": ref, "baseline": baseline}
            budget = {"iterations_total": 3, "limits": {"branches": 4}}
            environment = {"gpu": {"name": "test"}}
            original_inputs = dict(inputs)
            original_budget = json.loads(json.dumps(budget))
            original_environment = json.loads(json.dumps(environment))

            first_store = self.artifacts.ArtifactStore(base / "run-a")
            first = first_store.initialize(
                inputs=inputs, budget=budget, environment=environment
            )
            second = self.artifacts.ArtifactStore(base / "run-b").initialize(
                inputs={"baseline": baseline, "ref": ref},
                budget=budget,
                environment=environment,
            )

            self.assertEqual(first["schema_version"], 2)
            self.assertEqual(first["input_hash"], second["input_hash"])
            self.assertEqual(len(first["input_hash"]), 64)
            self.assertEqual(first["budget"], budget)
            self.assertEqual(first["environment"], environment)
            self.assertEqual(inputs, original_inputs)
            self.assertEqual(budget, original_budget)
            self.assertEqual(environment, original_environment)
            self.assertEqual(
                first["inputs"]["baseline"]["path"], str(baseline.resolve())
            )
            self.assertEqual(
                first["inputs"]["baseline"]["size_bytes"],
                baseline.stat().st_size,
            )
            self.assertEqual(
                first["inputs"]["baseline"]["sha256"],
                self.artifacts.sha256_file(baseline),
            )
            self.assertEqual(
                json.loads((first_store.root / "manifest.json").read_text("utf-8")),
                first,
            )
            for name in ("workload", "baseline", "candidates"):
                self.assertTrue((first_store.root / name).is_dir())

            budget["limits"]["branches"] = 99
            environment["gpu"]["name"] = "changed"
            self.assertEqual(first["budget"]["limits"]["branches"], 4)
            self.assertEqual(first["environment"]["gpu"]["name"], "test")

    def test_sha256_file_rejects_missing_path_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.py"
            with self.assertRaisesRegex(ValueError, str(missing)):
                self.artifacts.sha256_file(missing)
            with self.assertRaisesRegex(ValueError, str(root)):
                self.artifacts.sha256_file(root)

    def test_append_jsonl_preserves_order_and_read_ignores_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self.artifacts.ArtifactStore(Path(tmp) / "run")
            store.append_jsonl("events/history.jsonl", {"index": 1})
            store.append_jsonl("events/history.jsonl", {"index": 2})
            path = store.root / "events" / "history.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                stream.write("\n")

            self.assertEqual(
                store.read_jsonl("events/history.jsonl"),
                [{"index": 1}, {"index": 2}],
            )

    def test_write_methods_return_normalized_paths_inside_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self.artifacts.ArtifactStore(Path(tmp) / "parent" / ".." / "run")
            results = (
                (
                    store.write_json("nested/../result.json", {"ok": True}),
                    store.root / "result.json",
                ),
                (
                    store.append_jsonl("events/../history.jsonl", {"index": 1}),
                    store.root / "history.jsonl",
                ),
            )

            for result, expected in results:
                with self.subTest(expected=expected.name):
                    self.assertIsInstance(result, Path)
                    self.assertEqual(result, expected.resolve())
                    self.assertEqual(result.relative_to(store.root), Path(expected.name))

    def test_read_jsonl_missing_is_empty_and_bad_line_reports_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self.artifacts.ArtifactStore(Path(tmp) / "run")
            self.assertEqual(store.read_jsonl("missing.jsonl"), [])
            store.root.mkdir(parents=True)
            (store.root / "bad.jsonl").write_text(
                '{"ok": true}\nnot json\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, r"line 2"):
                store.read_jsonl("bad.jsonl")

    def test_paths_and_candidate_ids_cannot_escape_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self.artifacts.ArtifactStore(Path(tmp) / "run")
            absolute = str(Path(tmp) / "outside.json")
            for method, path, payload in (
                (store.write_json, "../outside.json", {}),
                (store.write_json, absolute, {}),
                (store.append_jsonl, "nested/../../outside.jsonl", {}),
                (store.read_jsonl, absolute, None),
            ):
                with self.subTest(method=method.__name__, path=path):
                    with self.assertRaises(ValueError):
                        method(path) if payload is None else method(path, payload)

            for candidate_id in (
                "",
                ".",
                "..",
                "../escape",
                "a/b",
                "a\\b",
                "space id",
            ):
                with self.subTest(candidate_id=candidate_id):
                    with self.assertRaises(ValueError):
                        store.candidate_dir(candidate_id)

            valid = store.candidate_dir("candidate_1.2-ok")
            self.assertEqual(valid, store.root / "candidates" / "candidate_1.2-ok")
            self.assertTrue(valid.is_dir())

    def test_candidate_dir_rejects_existing_symlink_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = self.artifacts.ArtifactStore(base / "run")
            candidates = store.root / "candidates"
            candidates.mkdir(parents=True)
            outside = base / "outside-candidate"
            outside.mkdir()
            marker = outside / "marker.txt"
            marker.write_text("unchanged", encoding="utf-8")
            (candidates / "linked").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, r"escapes run root"):
                store.candidate_dir("linked")

            self.assertEqual(marker.read_text("utf-8"), "unchanged")
            self.assertEqual([path.name for path in outside.iterdir()], ["marker.txt"])

    def test_candidate_dir_rejects_candidates_parent_symlink_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = self.artifacts.ArtifactStore(base / "run")
            store.root.mkdir(parents=True)
            outside = base / "outside-candidates"
            outside.mkdir()
            (store.root / "candidates").symlink_to(
                outside, target_is_directory=True
            )

            with self.assertRaisesRegex(ValueError, r"escapes run root"):
                store.candidate_dir("new-candidate")

            self.assertFalse((outside / "new-candidate").exists())

    def test_checkpoint_requires_matching_v2_schema_and_frozen_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self.artifacts.ArtifactStore(Path(tmp) / "run")
            with self.assertRaisesRegex(ValueError, r"checkpoint.*not found"):
                store.load_checkpoint(expected_input_hash="abc")

            payload = {"input_hash": "abc", "schema_version": 1, "nested": {"x": 1}}
            path = store.write_checkpoint(payload)
            payload["nested"]["x"] = 9
            self.assertEqual(path, store.root / "checkpoint.json")
            loaded = store.load_checkpoint(expected_input_hash="abc")
            self.assertEqual(loaded["schema_version"], 2)
            self.assertEqual(loaded["nested"]["x"], 1)

            with self.assertRaisesRegex(ValueError, r"frozen input"):
                store.load_checkpoint(expected_input_hash="different")

            store.write_json(
                "checkpoint.json", {"schema_version": 1, "input_hash": "abc"}
            )
            with self.assertRaisesRegex(ValueError, r"schema.*2"):
                store.load_checkpoint(expected_input_hash="abc")

    def test_load_checkpoint_rejects_symlink_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = self.artifacts.ArtifactStore(base / "run")
            store.root.mkdir(parents=True)
            outside = base / "outside-checkpoint.json"
            outside.write_text(
                json.dumps({"schema_version": 2, "input_hash": "abc"}),
                encoding="utf-8",
            )
            (store.root / "checkpoint.json").symlink_to(outside)

            with self.assertRaisesRegex(ValueError, r"escapes run root"):
                store.load_checkpoint(expected_input_hash="abc")

            self.assertEqual(
                json.loads(outside.read_text("utf-8")),
                {"schema_version": 2, "input_hash": "abc"},
            )

    def test_write_checkpoint_rejects_symlink_outside_root_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = self.artifacts.ArtifactStore(base / "run")
            store.root.mkdir(parents=True)
            outside = base / "outside-checkpoint.json"
            outside.write_text('{"sentinel": "unchanged"}', encoding="utf-8")
            checkpoint_link = store.root / "checkpoint.json"
            checkpoint_link.symlink_to(outside)

            with self.assertRaisesRegex(ValueError, r"escapes run root"):
                store.write_checkpoint({"input_hash": "abc"})

            self.assertTrue(checkpoint_link.is_symlink())
            self.assertEqual(
                json.loads(outside.read_text("utf-8")),
                {"sentinel": "unchanged"},
            )

    def test_atomic_json_replaces_and_serialization_failure_preserves_old_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "value.json"
            self.artifacts.atomic_write_json(path, {"value": "old"})
            self.artifacts.atomic_write_json(path, {"value": "new"})
            self.assertEqual(json.loads(path.read_text("utf-8")), {"value": "new"})

            with self.assertRaises(TypeError):
                self.artifacts.atomic_write_json(path, {"bad": object()})

            self.assertEqual(json.loads(path.read_text("utf-8")), {"value": "new"})
            self.assertEqual([item.name for item in path.parent.iterdir()], [path.name])


if __name__ == "__main__":
    unittest.main()
