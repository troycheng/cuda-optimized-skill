from __future__ import annotations

import copy
import importlib.util
import json
import multiprocessing
import os
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "skills"
    / "cuda-kernel-optimizer"
    / "scripts"
    / "strategy_memory.py"
)
WORKLOAD_ADAPTER_PATH = MODULE_PATH.with_name("workload_adapter.py")


def _load_strategy_memory():
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_strategy_memory", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_workload_adapter():
    module_name = "cuda_optimizer_strategy_memory_workload_adapter_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, WORKLOAD_ADAPTER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _worker_append(memory: str, scope: dict, record: dict, start) -> None:
    module = _load_strategy_memory()
    start.wait()
    module.append_run(memory, scope, record)


class StrategyMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = _load_strategy_memory()
        self.workloads = _load_workload_adapter()

    @staticmethod
    def _objective() -> dict:
        return {
            "primary_metric": {"name": "latency_ms", "direction": "lower"},
            "min_effect_pct": 1.0,
            "constraints": [],
        }

    def _python_workload(self, root: Path, suffix: str) -> dict:
        helper = root / f"helper{suffix}.py"
        helper.write_text("VALUE = 1\n", encoding="utf-8")
        source = root / f"workload{suffix}.py"
        source.write_text(
            textwrap.dedent(
                f"""
                WORKLOAD_DEPENDENCIES = ["{helper.name}"]
                def prepare(candidate): return None
                def validate(candidate): return True
                def benchmark(candidate): return {{"latency_ms": 1.0}}
                def metrics(): return {self._objective()!r}
                def cleanup(): return None
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        contract = root / f"workload-contract{suffix}.json"
        contract.write_text(
            json.dumps(
                {
                    "kind": "python",
                    "source": source.name,
                    "objective": self._objective(),
                    "cases": [{"M": 128}],
                }
            ),
            encoding="utf-8",
        )
        spec = self.workloads.normalize_workload(workload_manifest=contract)
        return {
            "kind": spec.kind,
            "source": spec.source,
            "objective": spec.objective,
            "cases": list(spec.cases),
            "source_hash": spec.source_hash,
        }

    def _manifest(
        self,
        root: Path,
        *,
        suffix: str = "",
        mode: str = "kernel-only",
        key_order: bool = False,
    ) -> Path:
        baseline = root / f"baseline{suffix}.py"
        reference = root / f"reference{suffix}.py"
        baseline.write_bytes(f"baseline{suffix}\n".encode())
        reference.write_bytes(f"reference{suffix}\n".encode())
        sha = self.memory.hashlib.sha256
        inputs = {
            "baseline": {
                "path": str(baseline),
                "sha256": sha(baseline.read_bytes()).hexdigest(),
                "size_bytes": baseline.stat().st_size,
            },
            "ref": {
                "path": str(reference),
                "sha256": sha(reference.read_bytes()).hexdigest(),
                "size_bytes": reference.stat().st_size,
            },
        }
        workload = None
        if mode == "full":
            workload = self._python_workload(root, suffix)
        manifest = {
            "schema_version": 2,
            "input_hash": ("a" if not suffix else "b") * 64,
            "inputs": inputs,
            "environment": {"primary_sm_arch": "sm_120", "gpu": "RTX 5090"},
            "backend": "triton",
            "dims": {"N": 256, "M": 128},
            "ptr_size": 8,
            "mode": mode,
            "workload": workload,
            "budget": {"max_rounds": 3},
            "confidence": 0.95,
            "min_effect_pct": 0.5,
            "started_at": 1.0,
        }
        if key_order:
            manifest = {key: manifest[key] for key in reversed(manifest)}
            manifest["inputs"] = {
                "ref": inputs["ref"],
                "baseline": inputs["baseline"],
            }
            manifest["dims"] = {"M": 128, "N": 256}
        path = root / f"manifest{suffix or '-a'}.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def _record(self, marker: str = "1") -> dict:
        return {
            "input_hash": "a" * 64,
            "candidate_sha256": marker * 64,
            "decision_sha256": "d" * 64,
            "checkpoint_identity": "e" * 64,
        }

    def _scope(self, marker: str = "a") -> dict:
        return {
            "manifest_schema_version": 2,
            "input_hash": marker * 64,
            "backend": "triton",
            "primary_sm_arch": "sm_120",
            "dims": {"M": 128},
            "ptr_size": 8,
            "baseline_sha256": "b" * 64,
            "ref_sha256": "c" * 64,
            "workload": {"mode": "kernel-only"},
        }

    def test_scope_is_canonical_and_contains_complete_kernel_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            first = self._manifest(root)
            payload = json.loads(first.read_text("utf-8"))
            reordered = root / "manifest-reordered.json"
            reordered.write_text(
                json.dumps({key: payload[key] for key in reversed(payload)}),
                encoding="utf-8",
            )

            scope = self.memory.scope_document(first)
            self.assertEqual(self.memory.scope_key(first), self.memory.scope_key(reordered))
            self.assertEqual(
                set(scope),
                {
                    "manifest_schema_version",
                    "input_hash",
                    "backend",
                    "primary_sm_arch",
                    "dims",
                    "ptr_size",
                    "baseline_sha256",
                    "ref_sha256",
                    "workload",
                },
            )
            self.assertEqual(scope["dims"], {"M": 128, "N": 256})
            self.assertEqual(scope["workload"], {"mode": "kernel-only"})

    def test_full_scope_captures_complete_workload_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._manifest(Path(tmp).resolve(), mode="full")
            workload = self.memory.scope_document(manifest)["workload"]
            self.assertEqual(workload["mode"], "full")
            self.assertEqual(
                set(workload),
                {"mode", "source", "source_hash", "objective", "cases", "kind"},
            )
            self.assertTrue(Path(workload["source"]).is_absolute())

    def test_full_scope_rejects_source_and_dependency_byte_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for field in ("source", "dependency"):
                case_root = root / field
                case_root.mkdir()
                manifest = self._manifest(case_root, mode="full")
                payload = json.loads(manifest.read_text("utf-8"))
                source = Path(payload["workload"]["source"])
                target = source if field == "source" else case_root / "helper.py"
                before = target.stat()
                original = target.read_text("utf-8")
                changed = original.replace("1.0", "2.0") if field == "source" else original.replace("1", "2")
                self.assertEqual(len(original), len(changed))
                target.write_text(changed, encoding="utf-8")
                os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
                with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, "source_hash"
                ):
                    self.memory.scope_key(manifest)

    def test_full_scope_rejects_source_leaf_and_parent_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            leaf_root = root / "leaf"
            leaf_root.mkdir()
            leaf_manifest = self._manifest(leaf_root, mode="full")
            leaf_payload = json.loads(leaf_manifest.read_text("utf-8"))
            source = Path(leaf_payload["workload"]["source"])
            real_source = source.with_name("real-workload.py")
            source.rename(real_source)
            source.symlink_to(real_source)
            with self.assertRaisesRegex(ValueError, "symlink|unsafe"):
                self.memory.scope_key(leaf_manifest)

            parent_root = root / "parent"
            parent_root.mkdir()
            parent_manifest = self._manifest(parent_root, mode="full")
            parent_payload = json.loads(parent_manifest.read_text("utf-8"))
            real_dir = root / "real-parent"
            parent_root.rename(real_dir)
            parent_root.symlink_to(real_dir, target_is_directory=True)
            parent_payload["workload"]["source"] = str(
                parent_root / Path(parent_payload["workload"]["source"]).name
            )
            for input_record in parent_payload["inputs"].values():
                input_record["path"] = str(real_dir / Path(input_record["path"]).name)
            linked_manifest = root / "parent-source-manifest.json"
            linked_manifest.write_text(json.dumps(parent_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "parent.*symlink|unsafe"):
                self.memory.scope_key(linked_manifest)

    def test_command_workload_source_list_is_preserved_and_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            runner = root / "runner.sh"
            runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            runner.chmod(0o755)
            objective = root / "objective.json"
            objective.write_text(json.dumps(self._objective()), encoding="utf-8")
            spec = self.workloads.normalize_workload(
                workload_cmd=[str(runner), "--label", "two words"],
                objective=objective,
            )
            manifest = self._manifest(root)
            payload = json.loads(manifest.read_text("utf-8"))
            payload["mode"] = "full"
            payload["workload"] = {
                "kind": spec.kind,
                "source": list(spec.source),
                "objective": spec.objective,
                "cases": list(spec.cases),
                "source_hash": spec.source_hash,
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            workload = self.memory.scope_document(manifest)["workload"]
            self.assertEqual(workload["source"], list(spec.source))

    def test_each_relevant_identity_change_changes_scope_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            manifest = self._manifest(root, mode="full")
            original = json.loads(manifest.read_text("utf-8"))
            base_key = self.memory.scope_key(manifest)
            changes = {
                "input_hash": lambda value: value.update(input_hash="f" * 64),
                "backend": lambda value: value.update(backend="cuda"),
                "arch": lambda value: value["environment"].update(primary_sm_arch="sm_100"),
                "dims": lambda value: value.update(dims={"M": 64, "N": 256}),
                "ptr_size": lambda value: value.update(ptr_size=4),
            }
            for name, mutate in changes.items():
                changed = copy.deepcopy(original)
                mutate(changed)
                candidate = root / f"changed-{name}.json"
                candidate.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(name=name):
                    self.assertNotEqual(base_key, self.memory.scope_key(candidate))

            changed = copy.deepcopy(original)
            changed["workload"] = self._python_workload(root, "-changed")
            candidate = root / "changed-workload.json"
            candidate.write_text(json.dumps(changed), encoding="utf-8")
            self.assertNotEqual(base_key, self.memory.scope_key(candidate))

    def test_same_filename_with_changed_bytes_never_shares_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            manifest = self._manifest(root)
            first_key = self.memory.scope_key(manifest)
            payload = json.loads(manifest.read_text("utf-8"))
            baseline = Path(payload["inputs"]["baseline"]["path"])
            baseline.write_bytes(b"different bytes\n")
            payload["inputs"]["baseline"]["sha256"] = self.memory.hashlib.sha256(
                baseline.read_bytes()
            ).hexdigest()
            payload["inputs"]["baseline"]["size_bytes"] = baseline.stat().st_size
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            self.assertNotEqual(first_key, self.memory.scope_key(manifest))

    def test_scope_rejects_stale_input_bytes_and_unsafe_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            manifest = self._manifest(root)
            payload = json.loads(manifest.read_text("utf-8"))
            Path(payload["inputs"]["baseline"]["path"]).write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "baseline.*sha256|content"):
                self.memory.scope_key(manifest)

            real = self._manifest(root, suffix="-safe")
            manifest_link = root / "manifest-link.json"
            input_link = root / "baseline-link.py"
            input_link.symlink_to(Path(json.loads(real.read_text())["inputs"]["baseline"]["path"]))
            linked_payload = json.loads(real.read_text())
            linked_payload["inputs"]["baseline"]["path"] = str(input_link)
            linked_manifest = root / "linked-input.json"
            linked_manifest.write_text(json.dumps(linked_payload))
            manifest_link.symlink_to(real)
            for unsafe in (manifest_link, linked_manifest):
                with self.subTest(unsafe=unsafe.name):
                    with self.assertRaisesRegex(ValueError, "symlink|unsafe"):
                        self.memory.scope_key(unsafe)

    def test_scope_rejects_symlinked_input_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            real = root / "real"
            real.mkdir()
            manifest = self._manifest(real)
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "parent.*symlink|unsafe"):
                self.memory.scope_key(linked / manifest.name)

    def test_scope_rejects_missing_arch_bad_sha_nonfinite_and_unknown_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            manifest = self._manifest(root, mode="full")
            original = json.loads(manifest.read_text())
            cases = []
            missing_arch = copy.deepcopy(original)
            del missing_arch["environment"]["primary_sm_arch"]
            cases.append(missing_arch)
            bad_sha = copy.deepcopy(original)
            bad_sha["input_hash"] = "bad"
            cases.append(bad_sha)
            unknown_workload = copy.deepcopy(original)
            unknown_workload["workload"]["unscoped"] = "danger"
            cases.append(unknown_workload)
            bad_dims = copy.deepcopy(original)
            bad_dims["dims"] = {"M": True}
            cases.append(bad_dims)
            for index, payload in enumerate(cases):
                path = root / f"invalid-{index}.json"
                path.write_text(json.dumps(payload))
                with self.subTest(index=index):
                    with self.assertRaises(ValueError):
                        self.memory.scope_key(path)

            manifest.write_text('{"schema_version":2,"schema_version":2}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self.memory.scope_key(manifest)
            manifest.write_text('{"schema_version":2,"input_hash":NaN}')
            with self.assertRaisesRegex(ValueError, "non-finite"):
                self.memory.scope_key(manifest)

    def test_new_memory_and_adjacent_lock_are_private_regular_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = Path(tmp).resolve() / "memory.json"
            self.assertTrue(self.memory.append_run(memory, self._scope(), self._record()))
            lock = memory.with_name(memory.name + ".lock")
            for path in (memory, lock):
                metadata = path.lstat()
                self.assertTrue(stat.S_ISREG(metadata.st_mode))
                self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o600)
            stored = self.memory.load_memory(memory)
            self.assertEqual(stored["schema_version"], self.memory.MEMORY_SCHEMA)
            self.assertEqual(len(next(iter(stored["scopes"].values()))["runs"]), 1)

    def test_memory_lock_and_parent_symlinks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            target = root / "target.json"
            target.write_text("{}")
            memory_link = root / "memory-link.json"
            memory_link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink|unsafe"):
                self.memory.append_run(memory_link, self._scope(), self._record())

            memory = root / "memory.json"
            lock = memory.with_name(memory.name + ".lock")
            lock.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "lock.*symlink|unsafe"):
                self.memory.append_run(memory, self._scope(), self._record())

            real = root / "real"
            real.mkdir()
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "parent.*symlink|unsafe"):
                self.memory.append_run(
                    linked / "memory.json", self._scope(), self._record()
                )

    def test_corrupt_memory_is_never_repaired(self) -> None:
        invalid_payloads = [
            {"schema_version": "wrong", "scopes": {}},
            {"schema_version": self.memory.MEMORY_SCHEMA, "scopes": {}, "extra": 1},
            {
                "schema_version": self.memory.MEMORY_SCHEMA,
                "scopes": {
                    "0" * 64: {
                        "scope": self._scope(),
                        "runs": [],
                        "methods": {},
                        "bundles": {},
                    }
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for index, payload in enumerate(invalid_payloads):
                memory = root / f"invalid-{index}.json"
                original = json.dumps(payload, sort_keys=True).encode()
                memory.write_bytes(original)
                with self.subTest(index=index):
                    with self.assertRaises(ValueError):
                        self.memory.append_run(memory, self._scope(), self._record())
                    self.assertEqual(memory.read_bytes(), original)

            nonfinite = root / "nonfinite.json"
            nonfinite.write_text(
                '{"schema_version":"cuda-kernel-optimizer/strategy-memory-v1",'
                '"scopes":{},"x":NaN}'
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                self.memory.load_memory(nonfinite)

    def test_update_exception_and_path_replacement_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            memory = root / "memory.json"
            self.memory.append_run(memory, self._scope(), self._record())
            original = memory.read_bytes()

            def fail(_store):
                raise RuntimeError("stop")

            with self.assertRaisesRegex(RuntimeError, "stop"):
                self.memory._locked_memory_update(memory, fail)
            self.assertEqual(memory.read_bytes(), original)

            replacement = b'{"replacement":true}'

            def replace(store):
                moved = root / "moved.json"
                os.replace(memory, moved)
                memory.write_bytes(replacement)
                return store

            with self.assertRaisesRegex(ValueError, "replaced|changed"):
                self.memory._locked_memory_update(memory, replace)
            self.assertEqual(memory.read_bytes(), replacement)

    def test_concurrent_distinct_records_survive_and_duplicates_dedupe(self) -> None:
        if "fork" not in multiprocessing.get_all_start_methods():
            self.skipTest("requires fork multiprocessing context")
        context = multiprocessing.get_context("fork")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for duplicate in (False, True):
                memory = root / f"memory-{duplicate}.json"
                scope = self._scope()
                first = self._record("1")
                second = first if duplicate else self._record("2")
                start = context.Event()
                processes = [
                    context.Process(
                        target=_worker_append,
                        args=(str(memory), scope, record, start),
                    )
                    for record in (first, second)
                ]
                for process in processes:
                    process.start()
                start.set()
                for process in processes:
                    process.join(10)
                    self.assertEqual(process.exitcode, 0)
                stored = self.memory.load_memory(memory)
                runs = next(iter(stored["scopes"].values()))["runs"]
                self.assertEqual(len(runs), 1 if duplicate else 2)

    def test_capacity_rejects_new_unique_entries_without_eviction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            memory = root / "scope-cap.json"
            record = self._record()
            for index in range(self.memory.MAX_SCOPES):
                scope = self._scope(f"{index:064x}"[-1])
                scope["input_hash"] = f"{index:064x}"
                scoped_record = copy.deepcopy(record)
                scoped_record["input_hash"] = scope["input_hash"]
                self.memory.append_run(memory, scope, scoped_record)
            before = memory.read_bytes()
            overflow = self._scope("f")
            overflow["input_hash"] = "f" * 64
            overflow_record = copy.deepcopy(record)
            overflow_record["input_hash"] = overflow["input_hash"]
            with self.assertRaisesRegex(ValueError, "scope capacity"):
                self.memory.append_run(memory, overflow, overflow_record)
            self.assertEqual(memory.read_bytes(), before)

            run_memory = root / "run-cap.json"
            scope = self._scope()
            for index in range(self.memory.MAX_RUNS_PER_SCOPE):
                unique = self._record()
                unique["candidate_sha256"] = f"{index:064x}"
                self.memory.append_run(run_memory, scope, unique)
            duplicate = self._record()
            duplicate["candidate_sha256"] = f"{0:064x}"
            before_duplicate = run_memory.read_bytes()
            self.assertFalse(self.memory.append_run(run_memory, scope, duplicate))
            self.assertEqual(run_memory.read_bytes(), before_duplicate)
            new_record = self._record()
            new_record["candidate_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "run capacity"):
                self.memory.append_run(run_memory, scope, new_record)
            self.assertEqual(len(next(iter(self.memory.load_memory(run_memory)["scopes"].values()))["runs"]), self.memory.MAX_RUNS_PER_SCOPE)

    def test_record_shape_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = Path(tmp).resolve() / "memory.json"
            for bad in (
                {**self._record(), "unknown": 1},
                {key: value for key, value in self._record().items() if key != "input_hash"},
                {**self._record(), "decision_sha256": "bad"},
            ):
                with self.subTest(bad=bad):
                    with self.assertRaises(ValueError):
                        self.memory.append_run(memory, self._scope(), bad)

    def test_record_input_hash_must_match_scope_on_append_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            memory = root / "memory.json"
            scope = self._scope()
            self.memory.append_run(memory, scope, self._record())
            original = memory.read_bytes()
            mismatch = self._record("2")
            mismatch["input_hash"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "input_hash.*scope"):
                self.memory.append_run(memory, scope, mismatch)
            self.assertEqual(memory.read_bytes(), original)

            corrupt = json.loads(original)
            entry = next(iter(corrupt["scopes"].values()))
            entry["runs"][0]["input_hash"] = "f" * 64
            memory.write_text(json.dumps(corrupt), encoding="utf-8")
            corrupt_bytes = memory.read_bytes()
            with self.assertRaisesRegex(ValueError, "input_hash.*scope"):
                self.memory.load_memory(memory)
            self.assertEqual(memory.read_bytes(), corrupt_bytes)

    def test_last_boundary_replacement_is_restored_without_overwrite_or_temp_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            memory = root / "memory.json"
            scope = self._scope()
            self.memory.append_run(memory, scope, self._record())
            original = memory.read_bytes()
            unexpected = root / "unexpected.json"
            unexpected_payload = b'{"unexpected":true}\n'
            unexpected.write_bytes(unexpected_payload)
            saved_old = root / "saved-old.json"
            real_exchange = self.memory._atomic_exchange
            raced = False

            def race_then_exchange(directory_fd, source_leaf, target_leaf):
                nonlocal raced
                if not raced:
                    raced = True
                    os.rename(memory, saved_old)
                    os.rename(unexpected, memory)
                return real_exchange(directory_fd, source_leaf, target_leaf)

            with mock.patch.object(
                self.memory, "_atomic_exchange", side_effect=race_then_exchange
            ), self.assertRaisesRegex(ValueError, "replaced|changed|compare"):
                self.memory.append_run(memory, scope, self._record("2"))

            self.assertTrue(raced)
            self.assertEqual(memory.read_bytes(), unexpected_payload)
            self.assertEqual(saved_old.read_bytes(), original)
            self.assertEqual(list(root.glob(f".{memory.name}.*.tmp")), [])

    def test_publication_exception_preserves_old_store_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            memory = root / "memory.json"
            scope = self._scope()
            self.memory.append_run(memory, scope, self._record())
            original = memory.read_bytes()
            with mock.patch.object(
                self.memory,
                "_atomic_exchange",
                side_effect=OSError("injected exchange failure"),
            ), self.assertRaisesRegex(OSError, "injected"):
                self.memory.append_run(memory, scope, self._record("2"))
            self.assertEqual(memory.read_bytes(), original)
            self.assertEqual(list(root.glob(f".{memory.name}.*.tmp")), [])

    def test_interruption_after_exchange_leaves_complete_new_and_recoverable_old_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            memory = root / "memory.json"
            scope = self._scope()
            self.memory.append_run(memory, scope, self._record())
            original = memory.read_bytes()
            real_exchange = self.memory._atomic_exchange

            def exchange_then_interrupt(directory_fd, source_leaf, target_leaf):
                real_exchange(directory_fd, source_leaf, target_leaf)
                raise SystemExit("simulated crash after atomic exchange")

            with mock.patch.object(
                self.memory,
                "_atomic_exchange",
                side_effect=exchange_then_interrupt,
            ), self.assertRaisesRegex(SystemExit, "simulated crash"):
                self.memory.append_run(memory, scope, self._record("2"))

            stored = self.memory.load_memory(memory)
            self.assertEqual(len(next(iter(stored["scopes"].values()))["runs"]), 2)
            recovery = list(root.glob(f".{memory.name}.*.tmp"))
            self.assertEqual(len(recovery), 1)
            self.assertEqual(recovery[0].read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
