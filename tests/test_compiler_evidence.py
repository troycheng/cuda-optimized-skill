from __future__ import annotations

import importlib.util
import json
import os
import stat
import tempfile
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"


def _load(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"cuda_optimizer_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CompilerEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.compiler_evidence = _load("compiler_evidence")

    def test_records_hashes_and_missing_stages_without_fabrication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "kernel.cu"
            binary = root / "kernel.so"
            ptx = root / "kernel.ptx"
            source.write_text("source", encoding="utf-8")
            binary.write_bytes(b"binary")
            ptx.write_text("ptx", encoding="utf-8")

            result = self.compiler_evidence.collect(
                source=source,
                binary=binary,
                discovered={"ptx": ptx, "ttgir": root / "missing.ttgir"},
                compile_command=["nvcc", "kernel.cu"],
                backend="cuda",
                arch="sm_120",
            )

        self.assertEqual(
            self.compiler_evidence.STAGES,
            ("source", "ttir", "ttgir", "llvm_ir", "ptx", "sass", "binary"),
        )
        self.assertEqual(result["source"]["status"], "available")
        self.assertEqual(result["ptx"]["status"], "available")
        self.assertEqual(result["ttgir"], {
            "status": "unavailable",
            "path": None,
            "sha256": None,
            "size_bytes": None,
        })
        self.assertRegex(result["binary"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(result["binary"]["size_bytes"], 6)
        self.assertEqual(result["compile_command"], ["nvcc", "kernel.cu"])
        self.assertEqual(result["backend"], "cuda")
        self.assertEqual(result["arch"], "sm_120")

    def test_identical_binary_is_reported_by_content_not_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary_a = root / "a.so"
            binary_b = root / "b.so"
            binary_c = root / "c.so"
            binary_a.write_bytes(b"same")
            binary_b.write_bytes(b"same")
            binary_c.write_bytes(b"different")

            self.assertTrue(self.compiler_evidence.same_artifact(binary_a, binary_b))
            self.assertFalse(self.compiler_evidence.same_artifact(binary_a, binary_c))
            self.assertFalse(
                self.compiler_evidence.same_artifact(binary_a, root / "missing.so")
            )

    def test_symlink_is_not_registered_as_compiler_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "kernel.py"
            real_ptx = root / "real.ptx"
            linked_ptx = root / "linked.ptx"
            source.write_text("source", encoding="utf-8")
            real_ptx.write_text("ptx", encoding="utf-8")
            try:
                linked_ptx.symlink_to(real_ptx)
            except OSError:
                self.skipTest("symlinks are unavailable")

            result = self.compiler_evidence.collect(
                source=source,
                discovered={"ptx": linked_ptx},
                backend="triton",
            )

        self.assertEqual(result["ptx"]["status"], "unavailable")

    def test_update_manifest_preserves_existing_compile_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "kernel.cu"
            binary = root / "kernel.so"
            sass = root / "sass.txt"
            evidence_dir = root / "compiler_evidence"
            source.write_text("source", encoding="utf-8")
            binary.write_bytes(b"binary")
            sass.write_text("SASS", encoding="utf-8")

            initial = self.compiler_evidence.update_manifest(
                evidence_dir,
                source=source,
                binary=binary,
                compile_command=["nvcc", "-O3", "kernel.cu"],
                backend="cuda",
                arch="sm_120",
            )
            updated = self.compiler_evidence.update_manifest(
                evidence_dir,
                discovered={"sass": sass},
            )
            persisted = json.loads(
                (evidence_dir / "manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(updated["compile_command"], initial["compile_command"])
        self.assertEqual(updated["binary"], initial["binary"])
        self.assertEqual(updated["sass"]["status"], "available")
        self.assertEqual(persisted, updated)

    def test_update_manifest_revalidates_artifacts_that_disappeared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "kernel.cu"
            binary = root / "kernel.so"
            sass = root / "sass.txt"
            source.write_text("source", encoding="utf-8")
            binary.write_bytes(b"binary")
            sass.write_text("SASS", encoding="utf-8")
            evidence_dir = root / "compiler_evidence"
            self.compiler_evidence.update_manifest(
                evidence_dir, source=source, binary=binary, backend="cuda"
            )
            binary.unlink()

            updated = self.compiler_evidence.update_manifest(
                evidence_dir, discovered={"sass": sass}
            )

        self.assertEqual(updated["binary"]["status"], "unavailable")
        self.assertEqual(updated["sass"]["status"], "unavailable")

    def test_merge_marks_content_tampered_non_overridden_stage_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "kernel.cu"
            sass = root / "sass.txt"
            source.write_text("original", encoding="utf-8")
            sass.write_text("sass", encoding="utf-8")
            evidence_dir = root / "compiler_evidence"
            original = self.compiler_evidence.update_manifest(
                evidence_dir, source=source, backend="cuda", arch="sm_120"
            )
            source.write_text("tampered", encoding="utf-8")

            updated = self.compiler_evidence.update_manifest(
                evidence_dir, discovered={"sass": sass}
            )

        self.assertNotEqual(
            original["source"]["sha256"],
            self.compiler_evidence.collect(source=source)["source"]["sha256"],
        )
        self.assertEqual(updated["source"]["status"], "unavailable")

    def test_manifest_schema_rejects_extra_fields_and_incoherent_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "kernel.cu"
            source.write_text("source", encoding="utf-8")
            evidence_dir = root / "compiler_evidence"
            self.compiler_evidence.update_manifest(
                evidence_dir, source=source, backend="cuda", arch="sm_120"
            )
            manifest_path = evidence_dir / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["unexpected"] = True
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "top-level"):
                self.compiler_evidence.update_manifest(evidence_dir)

            payload.pop("unexpected")
            payload["source"]["extra"] = "forged"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source"):
                self.compiler_evidence.update_manifest(evidence_dir)

    def test_collect_rejects_forged_compile_metadata(self) -> None:
        for kwargs in (
            {"compile_command": ["nvcc", ""]},
            {"backend": "forged"},
            {"arch": []},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.compiler_evidence.collect(**kwargs)

    def test_fresh_manifest_overwrites_corrupt_previous_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "kernel.cu"
            source.write_text("source", encoding="utf-8")
            evidence_dir = root / "compiler_evidence"
            evidence_dir.mkdir()
            (evidence_dir / "manifest.json").write_text(
                "{not-json", encoding="utf-8"
            )

            result = self.compiler_evidence.write_fresh_manifest(
                evidence_dir,
                source=source,
                compile_command=["nvcc"],
                backend="cuda",
                arch="sm_120",
            )

        self.assertEqual(result["source"]["status"], "available")

    def test_parent_symlink_switch_during_hashing_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_dir = root / "first"
            second_dir = root / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            (first_dir / "kernel.ptx").write_text("first", encoding="utf-8")
            (second_dir / "kernel.ptx").write_text("second", encoding="utf-8")
            linked_dir = root / "active"
            try:
                linked_dir.symlink_to(first_dir, target_is_directory=True)
            except OSError:
                self.skipTest("symlinks are unavailable")
            candidate = linked_dir / "kernel.ptx"
            real_open = os.open
            switched = False

            def switching_open(path, flags, *args, **kwargs):
                nonlocal switched
                if not switched:
                    switched = True
                    linked_dir.unlink()
                    linked_dir.symlink_to(second_dir, target_is_directory=True)
                return real_open(path, flags, *args, **kwargs)

            with mock.patch.object(
                self.compiler_evidence.os, "open", side_effect=switching_open
            ):
                result = self.compiler_evidence.collect(
                    discovered={"ptx": candidate}, backend="triton"
                )

        self.assertEqual(result["ptx"]["status"], "unavailable")

    def test_failed_atomic_replace_leaves_previous_manifest_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence_dir = root / "compiler_evidence"
            source = root / "kernel.cu"
            source.write_text("source", encoding="utf-8")
            first = self.compiler_evidence.update_manifest(
                evidence_dir, source=source, backend="cuda"
            )

            with mock.patch.object(
                self.compiler_evidence.os,
                "replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    self.compiler_evidence.update_manifest(
                        evidence_dir, arch="sm_120"
                    )

            persisted = json.loads(
                (evidence_dir / "manifest.json").read_text(encoding="utf-8")
            )
            leftovers = list(evidence_dir.glob(".*.tmp"))

        self.assertEqual(persisted, first)
        self.assertEqual(leftovers, [])

    def test_refuses_symlink_evidence_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "outside"
            outside.mkdir()
            linked = root / "compiler_evidence"
            try:
                linked.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("symlinks are unavailable")

            with self.assertRaisesRegex(ValueError, "symlink"):
                self.compiler_evidence.update_manifest(linked, backend="cuda")

    def test_triton_cache_discovery_only_returns_new_real_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            stale = cache / "stale.ttir"
            stale.write_text("old", encoding="utf-8")
            before = self.compiler_evidence.snapshot_cache(cache)
            ptx = cache / "hash" / "kernel.ptx"
            ttgir = cache / "hash" / "kernel.ttgir"
            ptx.parent.mkdir()
            ptx.write_text("ptx", encoding="utf-8")
            ttgir.write_text("ttgir", encoding="utf-8")
            ignored = cache / "hash" / "metadata.json"
            ignored.write_text("{}", encoding="utf-8")

            discovered = self.compiler_evidence.discover_triton_cache(cache, before)

        self.assertEqual(discovered["ptx"], ptx.resolve())
        self.assertEqual(discovered["ttgir"], ttgir.resolve())
        self.assertNotIn("ttir", discovered)
        self.assertNotIn("metadata", discovered)

    def test_triton_cache_selection_is_deterministic_for_same_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            first = cache / "a" / "kernel.ptx"
            second = cache / "b" / "kernel.ptx"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            timestamp_ns = 1_700_000_000_000_000_000
            os.utime(first, ns=(timestamp_ns, timestamp_ns))
            os.utime(second, ns=(timestamp_ns, timestamp_ns))

            discovered = self.compiler_evidence.discover_triton_cache(cache, {})

        self.assertEqual(discovered["ptx"], second.resolve())
        self.assertEqual(
            self.compiler_evidence.discover_triton_cache(cache / "missing", {}),
            {},
        )

    def test_triton_cache_never_combines_stages_from_different_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            ttir = cache / "unit-a" / "kernel.ttir"
            ptx = cache / "unit-b" / "kernel.ptx"
            ttir.parent.mkdir()
            ptx.parent.mkdir()
            ttir.write_text("ttir", encoding="utf-8")
            ptx.write_text("ptx", encoding="utf-8")

            discovered = self.compiler_evidence.discover_triton_cache(cache, {})

        self.assertEqual(len(discovered), 1)
        self.assertNotEqual(set(discovered), {"ttir", "ptx"})

    def test_triton_snapshot_detects_same_size_same_mtime_content_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            ptx = cache / "unit" / "kernel.ptx"
            ptx.parent.mkdir()
            ptx.write_text("first", encoding="utf-8")
            timestamp_ns = 1_700_000_000_000_000_000
            os.utime(ptx, ns=(timestamp_ns, timestamp_ns))
            before = self.compiler_evidence.snapshot_cache(cache)
            ptx.write_text("other", encoding="utf-8")
            os.utime(ptx, ns=(timestamp_ns, timestamp_ns))

            discovered = self.compiler_evidence.discover_triton_cache(cache, before)

        self.assertEqual(discovered["ptx"], ptx.resolve())

    def test_durable_triton_stage_names_are_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence_dir = root / "evidence"
            first = root / "first" / "kernel.ptx"
            second = root / "second" / "kernel.ptx"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_text("first ptx", encoding="utf-8")
            second.write_text("second ptx", encoding="utf-8")

            first_published = self.compiler_evidence.publish_triton_stages(
                evidence_dir, {"ptx": first}
            )["ptx"]
            second_published = self.compiler_evidence.publish_triton_stages(
                evidence_dir, {"ptx": second}
            )["ptx"]

            self.assertNotEqual(first_published, second_published)
            self.assertEqual(first_published.read_text(encoding="utf-8"), "first ptx")
            self.assertEqual(second_published.read_text(encoding="utf-8"), "second ptx")
            self.assertRegex(first_published.name, r"^ptx-[0-9a-f]{64}\.ptx$")

    def test_rejects_unknown_stage_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown compiler stage"):
            self.compiler_evidence.collect(discovered={"made_up": "x"})


class CompilerEvidenceIntegrationTests(unittest.TestCase):
    def test_compile_cu_records_exact_command_backend_arch_and_binary(self) -> None:
        benchmark = _load("benchmark")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "kernel.cu"
            binary = root / "kernel.so"
            source.write_text(
                'extern "C" void solve(float *out) {}', encoding="utf-8"
            )

            commands = []
            attempt_identity = []
            private_dirs = []

            def fake_run(command, **_kwargs):
                commands.append(list(command))
                attempt = Path(command[command.index("-o") + 1])
                private_dirs.append(attempt.parent)
                self.assertEqual(attempt.parent.parent, binary.parent)
                self.assertEqual(stat.S_IMODE(attempt.parent.stat().st_mode), 0o700)
                self.assertEqual(attempt.parent.stat().st_dev, binary.parent.stat().st_dev)
                self.assertNotEqual(attempt, binary)
                self.assertTrue(attempt.is_file())
                self.assertEqual(attempt.stat().st_size, 0)
                attempt_identity.append((attempt.stat().st_dev, attempt.stat().st_ino))
                with attempt.open("r+b") as stream:
                    stream.write(b"elf")
                    stream.truncate()
                self.assertEqual(
                    (attempt.stat().st_dev, attempt.stat().st_ino),
                    attempt_identity[-1],
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(benchmark.subprocess, "run", side_effect=fake_run):
                benchmark.compile_cu(
                    str(source), str(binary), "sm_120", "nvcc", backend="cuda"
                )

            manifest = json.loads(
                (root / "compiler_evidence" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            binary_bytes = binary.read_bytes()
            private_dirs_removed = [not path.exists() for path in private_dirs]

        self.assertEqual(manifest["backend"], "cuda")
        self.assertEqual(manifest["arch"], "sm_120")
        self.assertEqual(manifest["binary"]["status"], "available")
        self.assertEqual(manifest["compile_command"][0], "nvcc")
        self.assertNotIn("-lineinfo", manifest["compile_command"])
        self.assertEqual(binary_bytes, b"elf")
        self.assertEqual(private_dirs_removed, [True])

    def test_private_compile_directory_blocks_shared_parent_symlink_attack(self) -> None:
        benchmark = _load("benchmark")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "kernel.cu"
            binary = root / "kernel.so"
            victim = root / "victim.so"
            source.write_text('extern "C" void solve(float *out) {}', encoding="utf-8")
            victim.write_bytes(b"SAFE")
            private_dirs = []
            decoys = []

            def fake_compiler_open(command, **_kwargs):
                attempt = Path(command[command.index("-o") + 1])
                self.assertNotEqual(attempt.parent, binary.parent)
                self.assertEqual(stat.S_IMODE(attempt.parent.stat().st_mode), 0o700)
                private_dirs.append(attempt.parent)
                decoy = binary.parent / attempt.name
                decoys.append(decoy)
                try:
                    decoy.symlink_to(victim)
                except OSError:
                    self.skipTest("symlinks are unavailable")
                with attempt.open("r+b") as stream:
                    stream.write(b"elf")
                    stream.truncate()
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            try:
                with mock.patch.object(
                    benchmark.subprocess, "run", side_effect=fake_compiler_open
                ):
                    benchmark.compile_cu(
                        str(source), str(binary), "sm_120", "nvcc", backend="cuda"
                    )

                self.assertEqual(victim.read_bytes(), b"SAFE")
                self.assertEqual(binary.read_bytes(), b"elf")
                self.assertTrue(all(not path.exists() for path in private_dirs))
            finally:
                for decoy in decoys:
                    decoy.unlink(missing_ok=True)

    def test_compile_rejects_attempt_path_replacement_without_following_symlink(self) -> None:
        benchmark = _load("benchmark")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "kernel.cu"
            binary = root / "kernel.so"
            victim = root / "victim.so"
            source.write_text('extern "C" void solve(float *out) {}', encoding="utf-8")
            victim.write_bytes(b"do-not-touch")

            def replace_attempt_with_symlink(command, **_kwargs):
                attempt = Path(command[command.index("-o") + 1])
                self.assertTrue(attempt.is_file())
                attempt.unlink()
                try:
                    attempt.symlink_to(victim)
                except OSError:
                    self.skipTest("symlinks are unavailable")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(
                benchmark.subprocess,
                "run",
                side_effect=replace_attempt_with_symlink,
            ):
                with self.assertRaises(SystemExit):
                    benchmark.compile_cu(
                        str(source), str(binary), "sm_120", "nvcc", backend="cuda"
                    )

            manifest = json.loads(
                (root / "compiler_evidence" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(victim.read_bytes(), b"do-not-touch")
            self.assertFalse(binary.exists())
            self.assertEqual(manifest["binary"]["status"], "unavailable")

    def test_publish_directory_fsync_failure_removes_visible_binary(self) -> None:
        benchmark = _load("benchmark")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "kernel.cu"
            binary = root / "kernel.so"
            source.write_text('extern "C" void solve(float *out) {}', encoding="utf-8")

            def fake_run(command, **_kwargs):
                Path(command[command.index("-o") + 1]).write_bytes(b"elf")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(
                benchmark.subprocess, "run", side_effect=fake_run
            ), mock.patch.object(
                benchmark,
                "_fsync_directory",
                side_effect=[OSError("publish fsync failed"), None],
            ):
                with self.assertRaisesRegex(OSError, "publish fsync failed"):
                    benchmark.compile_cu(
                        str(source), str(binary), "sm_120", "nvcc", backend="cuda"
                    )

            manifest = json.loads(
                (root / "compiler_evidence" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(binary.exists())
            self.assertEqual(manifest["binary"]["status"], "unavailable")

    def test_compile_failure_invalidates_stale_binary_evidence(self) -> None:
        benchmark = _load("benchmark")
        compiler_evidence = _load("compiler_evidence")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "kernel.cu"
            binary = root / "kernel.so"
            source.write_text(
                'extern "C" void solve(float *out) {}', encoding="utf-8"
            )
            binary.write_bytes(b"stale-elf")
            compiler_evidence.update_manifest(
                root / "compiler_evidence",
                source=source,
                binary=binary,
                compile_command=["old-nvcc"],
                backend="cuda",
                arch="sm_90",
            )

            failed = SimpleNamespace(returncode=1, stdout="", stderr="compile failed")
            with mock.patch.object(benchmark.subprocess, "run", return_value=failed):
                with self.assertRaises(SystemExit):
                    benchmark.compile_cu(
                        str(source), str(binary), "sm_120", "nvcc", backend="cuda"
                    )

            manifest = json.loads(
                (root / "compiler_evidence" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(manifest["binary"]["status"], "unavailable")
        self.assertFalse(binary.exists())
        self.assertEqual(manifest["source"]["status"], "available")
        self.assertEqual(manifest["arch"], "sm_120")
        self.assertEqual(manifest["compile_command"][0], "nvcc")

    def test_recompile_invalidates_stale_downstream_stage_evidence(self) -> None:
        benchmark = _load("benchmark")
        compiler_evidence = _load("compiler_evidence")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "kernel.cu"
            binary = root / "kernel.so"
            sass = root / "old.sass"
            ptx = root / "old.ptx"
            source.write_text(
                'extern "C" void solve(float *out) {}', encoding="utf-8"
            )
            binary.write_bytes(b"old-elf")
            sass.write_text("old sass", encoding="utf-8")
            ptx.write_text("old ptx", encoding="utf-8")
            compiler_evidence.update_manifest(
                root / "compiler_evidence",
                source=source,
                binary=binary,
                discovered={"sass": sass, "ptx": ptx},
                compile_command=["old-nvcc"],
                backend="cuda",
                arch="sm_90",
            )

            def fake_run(command, **_kwargs):
                Path(command[command.index("-o") + 1]).write_bytes(b"new-elf")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(benchmark.subprocess, "run", side_effect=fake_run):
                benchmark.compile_cu(
                    str(source), str(binary), "sm_120", "nvcc", backend="cuda"
                )

            manifest = json.loads(
                (root / "compiler_evidence" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(manifest["binary"]["status"], "available")
        self.assertEqual(manifest["sass"]["status"], "unavailable")
        self.assertEqual(manifest["ptx"]["status"], "unavailable")

    def test_zero_exit_without_output_binary_is_a_compile_failure(self) -> None:
        benchmark = _load("benchmark")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "kernel.cu"
            binary = root / "kernel.so"
            source.write_text(
                'extern "C" void solve(float *out) {}', encoding="utf-8"
            )
            binary.write_bytes(b"stale-elf")
            succeeded_without_output = SimpleNamespace(
                returncode=0, stdout="", stderr=""
            )

            with mock.patch.object(
                benchmark.subprocess, "run", return_value=succeeded_without_output
            ):
                with self.assertRaises(SystemExit):
                    benchmark.compile_cu(
                        str(source), str(binary), "sm_120", "nvcc", backend="cuda"
                    )

            manifest = json.loads(
                (root / "compiler_evidence" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(manifest["binary"]["status"], "unavailable")
        self.assertFalse(binary.exists())

    def test_cutlass_header_early_exit_invalidates_old_output_and_evidence(self) -> None:
        benchmark = _load("benchmark")
        compiler_evidence = _load("compiler_evidence")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "kernel.cu"
            binary = root / "kernel.so"
            sass = root / "old.sass"
            source.write_text(
                'extern "C" void solve(float *out) {}', encoding="utf-8"
            )
            binary.write_bytes(b"stale")
            sass.write_text("stale", encoding="utf-8")
            compiler_evidence.update_manifest(
                root / "compiler_evidence",
                source=source,
                binary=binary,
                discovered={"sass": sass},
                compile_command=["old-nvcc"],
                backend="cuda",
                arch="sm_90",
            )

            with mock.patch.object(benchmark, "find_cutlass_include_dir", return_value=""):
                with self.assertRaises(SystemExit):
                    benchmark.compile_cu(
                        str(source), str(binary), "sm_120", "nvcc", backend="cutlass"
                    )

            manifest = json.loads(
                (root / "compiler_evidence" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertFalse(binary.exists())
        self.assertEqual(manifest["backend"], "cutlass")
        self.assertEqual(manifest["arch"], "sm_120")
        self.assertEqual(manifest["binary"]["status"], "unavailable")
        self.assertEqual(manifest["sass"]["status"], "unavailable")

    def test_preprocess_failure_invalidates_old_output_and_records_attempt(self) -> None:
        benchmark = _load("benchmark")
        compiler_evidence = _load("compiler_evidence")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "missing.cu"
            binary = root / "kernel.so"
            old_source = root / "old.cu"
            old_source.write_text("old", encoding="utf-8")
            binary.write_bytes(b"stale")
            compiler_evidence.update_manifest(
                root / "compiler_evidence",
                source=old_source,
                binary=binary,
                compile_command=["old-nvcc"],
                backend="cuda",
                arch="sm_90",
            )

            with self.assertRaises(OSError):
                benchmark.compile_cu(
                    str(source), str(binary), "sm_120", "nvcc", backend="cuda"
                )

            manifest = json.loads(
                (root / "compiler_evidence" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertFalse(binary.exists())
        self.assertEqual(manifest["backend"], "cuda")
        self.assertEqual(manifest["arch"], "sm_120")
        self.assertEqual(manifest["source"]["status"], "unavailable")
        self.assertEqual(manifest["binary"]["status"], "unavailable")

    def test_spawn_failure_invalidates_old_output(self) -> None:
        benchmark = _load("benchmark")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "kernel.cu"
            binary = root / "kernel.so"
            source.write_text(
                'extern "C" void solve(float *out) {}', encoding="utf-8"
            )
            binary.write_bytes(b"stale")

            with mock.patch.object(
                benchmark.subprocess, "run", side_effect=OSError("spawn failed")
            ):
                with self.assertRaises(SystemExit):
                    benchmark.compile_cu(
                        str(source), str(binary), "sm_120", "nvcc", backend="cuda"
                    )

            manifest = json.loads(
                (root / "compiler_evidence" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertFalse(binary.exists())
        self.assertEqual(manifest["binary"]["status"], "unavailable")

    def test_preprocessed_compile_records_canonical_source_not_temporary_copy(self) -> None:
        benchmark = _load("benchmark")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "kernel.cu"
            binary = root / "kernel.so"
            source.write_text(
                "#include <__clang_cuda_runtime_wrapper.h>\n"
                'extern "C" void solve(float *out) {}',
                encoding="utf-8",
            )

            def fake_run(command, **_kwargs):
                Path(command[command.index("-o") + 1]).write_bytes(b"elf")
                self.assertTrue(command[-1].endswith(".nvcc_clean.cu"))
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(benchmark.subprocess, "run", side_effect=fake_run):
                benchmark.compile_cu(
                    str(source), str(binary), "sm_120", "nvcc", backend="cuda"
                )

            manifest = json.loads(
                (root / "compiler_evidence" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(manifest["source"]["path"], str(source.resolve()))
        self.assertNotIn("nvcc_clean", manifest["source"]["path"])
        self.assertTrue(manifest["compile_command"][-1].endswith(".nvcc_clean.cu"))

    def test_triton_records_only_cache_files_created_by_kernel_call(self) -> None:
        benchmark = _load("benchmark")

        class FakeTensor:
            pass

        benchmark.torch = SimpleNamespace(Tensor=FakeTensor)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            (cache / "unrelated.ptx").write_text("old", encoding="utf-8")
            source = root / "kernel.py"
            source.write_text(
                "import os\n"
                "from pathlib import Path\n"
                "def setup(**kwargs):\n"
                "    return {'inputs': {'n': kwargs['n']}, 'outputs': []}\n"
                "def run_kernel(**kwargs):\n"
                "    root = Path(os.environ['TRITON_CACHE_DIR']) / 'hash'\n"
                "    root.mkdir(parents=True, exist_ok=True)\n"
                "    (root / 'kernel.ttir').write_text('ttir')\n"
                "    (root / 'kernel.ptx').write_text('ptx')\n",
                encoding="utf-8",
            )

            with mock.patch.dict(
                os.environ, {"TRITON_CACHE_DIR": str(cache)}, clear=False
            ):
                state = benchmark._setup_triton(
                    str(source), {"n": 1}, seed=1, arch="sm_120"
                )
                state["callable"]()
                benchmark._record_triton_compiler_evidence(state)

            manifest = json.loads(
                (root / "compiler_evidence" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            durable_paths = {
                stage: Path(manifest[stage]["path"])
                for stage in ("ttir", "ptx")
            }
            benchmark.cleanup_solution(state)

            for path in durable_paths.values():
                self.assertTrue(path.is_file())
                self.assertTrue(
                    path.is_relative_to((root / "compiler_evidence" / "stages").resolve())
                )
            durable_contents = {
                stage: path.read_text(encoding="utf-8")
                for stage, path in durable_paths.items()
            }

        self.assertEqual(manifest["backend"], "triton")
        self.assertEqual(manifest["ttir"]["status"], "available")
        self.assertEqual(manifest["ptx"]["status"], "available")
        self.assertEqual(durable_contents["ttir"], "ttir")
        self.assertEqual(durable_contents["ptx"], "ptx")
        self.assertEqual(manifest["binary"]["status"], "unavailable")

    def test_triton_missing_interface_cleans_cache_without_masking_error(self) -> None:
        benchmark = _load("benchmark")
        real_temporary_directory = tempfile.TemporaryDirectory
        owners = []

        class CleanupFailingOwner:
            def __init__(self, *args, **kwargs):
                self.inner = real_temporary_directory(*args, **kwargs)
                self.name = self.inner.name
                self.cleanup_calls = 0

            def cleanup(self):
                self.cleanup_calls += 1
                self.inner.cleanup()
                raise OSError("cleanup broke")

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "kernel.py"
            source.write_text(
                "def run_kernel(**kwargs):\n"
                "    return None\n",
                encoding="utf-8",
            )

            def tracked_owner(*args, **kwargs):
                owner = CleanupFailingOwner(*args, **kwargs)
                owners.append(owner)
                return owner

            try:
                with mock.patch.object(
                    benchmark.tempfile, "TemporaryDirectory", side_effect=tracked_owner
                ):
                    try:
                        benchmark._setup_triton(str(source), {}, seed=1, arch="sm_120")
                    except AttributeError as error:
                        captured = error
                        frames = traceback.extract_tb(error.__traceback__)
                    else:
                        self.fail("missing setup() must raise AttributeError")

                self.assertIn("must define a `setup", str(captured))
                self.assertIn("_setup_triton", [frame.name for frame in frames])
                self.assertEqual(frames[-1].name, "_setup_triton_impl")
                self.assertEqual(owners[0].cleanup_calls, 1)
                self.assertFalse(Path(owners[0].name).exists())
            finally:
                for owner in owners:
                    owner.inner.cleanup()

    def test_triton_setup_exception_cleans_cache_and_preserves_traceback(self) -> None:
        benchmark = _load("benchmark")
        real_temporary_directory = tempfile.TemporaryDirectory
        owners = []

        def tracked_owner(*args, **kwargs):
            owner = real_temporary_directory(*args, **kwargs)
            owners.append(owner)
            return owner

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "kernel.py"
            source.write_text(
                "def setup(**kwargs):\n"
                "    raise RuntimeError('setup exploded')\n"
                "def run_kernel(**kwargs):\n"
                "    return None\n",
                encoding="utf-8",
            )
            try:
                with mock.patch.object(
                    benchmark.tempfile, "TemporaryDirectory", side_effect=tracked_owner
                ):
                    try:
                        benchmark._setup_triton(str(source), {}, seed=1, arch="sm_120")
                    except RuntimeError as error:
                        captured = error
                        frames = traceback.extract_tb(error.__traceback__)
                    else:
                        self.fail("setup exception must be preserved")

                self.assertEqual(str(captured), "setup exploded")
                self.assertEqual(frames[-1].name, "setup")
                self.assertFalse(Path(owners[0].name).exists())
            finally:
                for owner in owners:
                    owner.cleanup()

    def test_triton_uses_isolated_cache_for_import_setup_and_launch(self) -> None:
        benchmark = _load("benchmark")

        class FakeTensor:
            pass

        benchmark.torch = SimpleNamespace(Tensor=FakeTensor)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ambient = root / "ambient-cache"
            ambient.mkdir()
            (ambient / "unrelated.ptx").write_text("ambient", encoding="utf-8")
            evidence_dir = root / "evidence"
            source = root / "kernel.py"
            source.write_text(
                "import os\n"
                "from pathlib import Path\n"
                "IMPORT_CACHE = os.environ.get('TRITON_CACHE_DIR')\n"
                "def setup(**kwargs):\n"
                "    global SETUP_CACHE\n"
                "    SETUP_CACHE = os.environ.get('TRITON_CACHE_DIR')\n"
                "    return {'inputs': {'n': kwargs['n']}, 'outputs': []}\n"
                "def run_kernel(**kwargs):\n"
                "    cache = os.environ.get('TRITON_CACHE_DIR')\n"
                "    if cache != IMPORT_CACHE or cache != SETUP_CACHE:\n"
                "        raise RuntimeError('cache changed across lifecycle')\n"
                "    unit = Path(cache) / 'compile-key'\n"
                "    unit.mkdir(parents=True, exist_ok=True)\n"
                "    (unit / 'kernel.ttir').write_text('ttir')\n"
                "    (unit / 'kernel.ptx').write_text('ptx')\n"
                "    return cache\n",
                encoding="utf-8",
            )

            with mock.patch.dict(
                os.environ, {"TRITON_CACHE_DIR": str(ambient)}, clear=False
            ):
                state = benchmark._setup_triton(
                    str(source),
                    {"n": 1},
                    seed=1,
                    arch="sm_120",
                    evidence_dir=evidence_dir,
                )
                isolated = Path(
                    state["_compiler_evidence_runtime"]["cache_dir"]
                ).resolve()
                self.assertNotEqual(isolated, ambient.resolve())
                self.assertTrue(isolated.exists())
                self.assertEqual(os.environ["TRITON_CACHE_DIR"], str(ambient))
                self.assertEqual(Path(state["callable"]()).resolve(), isolated)
                self.assertEqual(os.environ["TRITON_CACHE_DIR"], str(ambient))
                benchmark._record_triton_compiler_evidence(state)

            try:
                manifest = json.loads(
                    (evidence_dir / "manifest.json").read_text(encoding="utf-8")
                )
                durable_ptx = Path(manifest["ptx"]["path"])
                self.assertTrue(
                    durable_ptx.is_relative_to((evidence_dir / "stages").resolve())
                )
                self.assertFalse(durable_ptx.is_relative_to(isolated))
                self.assertNotEqual(
                    durable_ptx, ambient / "unrelated.ptx"
                )
                self.assertTrue(isolated.exists())
            finally:
                benchmark.cleanup_solution(state)
            self.assertFalse(isolated.exists())
            self.assertEqual(durable_ptx.read_text(encoding="utf-8"), "ptx")
            reloaded = benchmark.compiler_evidence.load_manifest(evidence_dir)
            self.assertEqual(reloaded["ptx"], manifest["ptx"])
            durable_identity = benchmark.compiler_evidence.artifact_identity(
                durable_ptx
            )
            self.assertIsNotNone(durable_identity)
            self.assertEqual(
                durable_identity["sha256"], manifest["ptx"]["sha256"]
            )

    def test_benchmark_run_keeps_triton_evidence_after_finally_cleanup(self) -> None:
        benchmark = _load("benchmark")

        class FakeTensor:
            pass

        fake_cuda = SimpleNamespace(
            current_device=lambda: 0,
            get_device_name=lambda _index: "fake-gpu",
            synchronize=lambda: None,
        )
        benchmark.torch = SimpleNamespace(Tensor=FakeTensor, cuda=fake_cuda)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence_dir = root / "evidence"
            result_path = root / "result.json"
            source = root / "kernel.py"
            source.write_text(
                "import os\n"
                "from pathlib import Path\n"
                "def setup(**kwargs):\n"
                "    return {'inputs': {'n': kwargs['n']}, 'outputs': []}\n"
                "def run_kernel(**kwargs):\n"
                "    unit = Path(os.environ['TRITON_CACHE_DIR']) / 'unit'\n"
                "    unit.mkdir(parents=True, exist_ok=True)\n"
                "    (unit / 'kernel.ptx').write_text('durable ptx')\n",
                encoding="utf-8",
            )
            state = benchmark._setup_triton(
                str(source), {"n": 1}, seed=None, arch="sm_120",
                evidence_dir=evidence_dir,
            )
            isolated = Path(state["_compiler_evidence_runtime"]["cache_dir"])

            with mock.patch.object(
                benchmark, "_setup_backend", return_value=state
            ), mock.patch.object(
                benchmark, "_time_iterations", return_value=[1.0]
            ):
                benchmark.run(
                    str(source), "", {"n": 1}, 0, 1, 0, "sm_120",
                    1e-4, 1e-3, 42, json_out=str(result_path), backend="triton",
                )

            result = json.loads(result_path.read_text(encoding="utf-8"))
            manifest = json.loads(
                (evidence_dir / "manifest.json").read_text(encoding="utf-8")
            )
            durable_ptx = Path(manifest["ptx"]["path"])

            self.assertFalse(isolated.exists())
            self.assertTrue(durable_ptx.is_file())
            self.assertEqual(durable_ptx.read_text(encoding="utf-8"), "durable ptx")
            self.assertEqual(
                result["compiler_evidence"]["manifest"]["ptx"], manifest["ptx"]
            )

    def test_readonly_triton_source_does_not_block_benchmark_evidence(self) -> None:
        benchmark = _load("benchmark")

        class FakeTensor:
            pass

        benchmark.torch = SimpleNamespace(Tensor=FakeTensor)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "readonly"
            source_dir.mkdir()
            source = source_dir / "kernel.py"
            source.write_text(
                "def setup(**kwargs):\n"
                "    return {'inputs': {'n': kwargs['n']}, 'outputs': []}\n"
                "def run_kernel(**kwargs):\n"
                "    return None\n",
                encoding="utf-8",
            )
            source_dir.chmod(0o555)
            try:
                state = benchmark._setup_triton(
                    str(source), {"n": 1}, seed=1, arch="sm_120"
                )
                state["callable"]()
                benchmark._record_triton_compiler_evidence(state)
            finally:
                source_dir.chmod(0o755)

        self.assertEqual(state["compiler_evidence"]["status"], "unavailable")
        self.assertTrue(state["compiler_evidence"]["error"])
        benchmark.cleanup_solution(state)

    def test_zero_warmup_does_not_finalize_triton_cache_before_first_launch(self) -> None:
        benchmark = _load("benchmark")
        benchmark.torch = SimpleNamespace(
            cuda=SimpleNamespace(synchronize=lambda: None)
        )
        state = {"callable": lambda: None, "compiler_evidence": {}}
        with mock.patch.object(benchmark, "_record_triton_compiler_evidence") as record:
            benchmark.warm_solution(state, 0)

        record.assert_not_called()

    def test_sass_dump_is_persisted_and_merged_into_compiler_manifest(self) -> None:
        sass_check = _load("sass_check")
        compiler_evidence = _load("compiler_evidence")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            iter_dir = root / "iterv1"
            iter_dir.mkdir()
            source = iter_dir / "kernel.cu"
            binary = iter_dir / "kernel.so"
            source.write_text("source", encoding="utf-8")
            binary.write_bytes(b"elf")
            compiler_evidence.write_fresh_manifest(
                iter_dir / "compiler_evidence",
                source=source,
                binary=binary,
                compile_command=["nvcc", "kernel.cu"],
                backend="cuda",
                arch="sm_120",
            )
            (iter_dir / "methods.json").write_text(
                json.dumps({"methods": [{"id": "vectorize"}]}), encoding="utf-8"
            )
            state = root / "state.json"
            state.write_text(json.dumps({"run_dir": str(root)}), encoding="utf-8")
            signatures = root / "signatures.json"
            signatures.write_text(
                json.dumps(
                    {
                        "methods": {
                            "vectorize": {
                                "sass_patterns": ["LDG"],
                                "require_any": True,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(
                sass_check, "_dump_sass", return_value="/*0000*/ LDG.E R1, [R2];"
            ):
                result = sass_check.run(str(state), 1, str(signatures))

            evidence_dir = iter_dir / "compiler_evidence"
            sass_text = (evidence_dir / "sass.txt").read_text(encoding="utf-8")
            manifest = json.loads(
                (evidence_dir / "manifest.json").read_text(encoding="utf-8")
            )

        self.assertTrue(result["checks"][0]["verified"])
        self.assertIn("LDG.E", sass_text)
        self.assertEqual(manifest["source"]["status"], "available")
        self.assertEqual(manifest["binary"]["status"], "available")
        self.assertEqual(manifest["sass"]["status"], "available")
        self.assertEqual(result["binary_sha256"], manifest["binary"]["sha256"])
        self.assertEqual(manifest["binary_sha256"], manifest["binary"]["sha256"])
        self.assertEqual(result["checks"][0]["status"], "passed")

    def test_sass_method_status_is_tri_state_and_never_vacuously_verified(self) -> None:
        sass_check = _load("sass_check")
        signatures = {
            "methods": {
                "empty": {"sass_patterns": [], "require_any": True},
                "required": {"sass_patterns": ["LDG"], "require_any": True},
            }
        }

        unknown = sass_check.check_method_sass("unknown", "LDG", signatures)
        not_applicable = sass_check.check_method_sass("empty", "LDG", signatures)
        failed = sass_check.check_method_sass("required", "NOP", signatures)
        passed = sass_check.check_method_sass("required", "LDG", signatures)

        self.assertEqual((unknown["status"], unknown["verified"]), ("unavailable", False))
        self.assertEqual(
            (not_applicable["status"], not_applicable["verified"]),
            ("not_applicable", False),
        )
        self.assertEqual((failed["status"], failed["verified"]), ("failed", False))
        self.assertEqual((passed["status"], passed["verified"]), ("passed", True))

    def test_sass_aggregate_does_not_hide_unknown_method_as_passed(self) -> None:
        sass_check = _load("sass_check")
        passed = {
            "method_id": "known",
            "status": "passed",
            "verified": True,
        }
        unknown = {
            "method_id": "unknown",
            "status": "unavailable",
            "verified": False,
        }
        skipped = {
            "method_id": "skipped",
            "status": "not_applicable",
            "verified": False,
        }

        self.assertEqual(sass_check._checks_status([passed, unknown]), "unavailable")
        self.assertEqual(sass_check._checks_status([passed, skipped]), "passed")
        self.assertEqual(
            sass_check._checks_status([passed, unknown, {"status": "failed"}]),
            "failed",
        )

    def test_sass_result_json_refuses_symlink_output(self) -> None:
        sass_check = _load("sass_check")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            iter_dir = root / "iterv1"
            iter_dir.mkdir()
            victim = root / "victim.json"
            victim.write_text("sentinel", encoding="utf-8")
            output = iter_dir / "sass_check.json"
            try:
                output.symlink_to(victim)
            except OSError:
                self.skipTest("symlinks are unavailable")

            with self.assertRaisesRegex(ValueError, "symlink"):
                sass_check._write_result(str(iter_dir), {"status": "passed"})

            self.assertEqual(victim.read_text(encoding="utf-8"), "sentinel")

    def test_sass_requires_current_bound_binary_manifest(self) -> None:
        sass_check = _load("sass_check")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            iter_dir = root / "iterv1"
            iter_dir.mkdir()
            (iter_dir / "kernel.cu").write_text("source", encoding="utf-8")
            (iter_dir / "kernel.so").write_bytes(b"elf")
            (iter_dir / "methods.json").write_text(
                json.dumps({"methods": [{"id": "vectorize"}]}), encoding="utf-8"
            )
            state = root / "state.json"
            state.write_text(json.dumps({"run_dir": str(root)}), encoding="utf-8")

            with mock.patch.object(sass_check, "_dump_sass") as dump:
                result = sass_check.run(str(state), 1)

        dump.assert_not_called()
        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["checks"][0]["verified"])

    def test_sass_rejects_symlink_binary_without_dumping(self) -> None:
        sass_check = _load("sass_check")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            iter_dir = root / "iterv1"
            iter_dir.mkdir()
            source = iter_dir / "kernel.cu"
            source.write_text("source", encoding="utf-8")
            real_binary = root / "real.so"
            real_binary.write_bytes(b"elf")
            linked_binary = iter_dir / "kernel.so"
            try:
                linked_binary.symlink_to(real_binary)
            except OSError:
                self.skipTest("symlinks are unavailable")
            (iter_dir / "methods.json").write_text(
                json.dumps({"methods": [{"id": "vectorize"}]}), encoding="utf-8"
            )
            state = root / "state.json"
            state.write_text(json.dumps({"run_dir": str(root)}), encoding="utf-8")

            with mock.patch.object(sass_check, "_dump_sass") as dump:
                result = sass_check.run(str(state), 1)

        dump.assert_not_called()
        self.assertEqual(result["status"], "unavailable")

    def test_sass_fails_closed_when_binary_is_replaced_during_dump(self) -> None:
        sass_check = _load("sass_check")
        compiler_evidence = _load("compiler_evidence")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            iter_dir = root / "iterv1"
            iter_dir.mkdir()
            source = iter_dir / "kernel.cu"
            binary = iter_dir / "kernel.so"
            source.write_text("source", encoding="utf-8")
            binary.write_bytes(b"original-elf")
            compiler_evidence.write_fresh_manifest(
                iter_dir / "compiler_evidence",
                source=source,
                binary=binary,
                compile_command=["nvcc"],
                backend="cuda",
                arch="sm_120",
            )
            (iter_dir / "methods.json").write_text(
                json.dumps({"methods": [{"id": "vectorize"}]}), encoding="utf-8"
            )
            state = root / "state.json"
            state.write_text(json.dumps({"run_dir": str(root)}), encoding="utf-8")

            def replace_during_dump(_path):
                replacement = iter_dir / "replacement.so"
                replacement.write_bytes(b"replacement-elf")
                os.replace(replacement, binary)
                return "LDG.E"

            with mock.patch.object(
                sass_check, "_dump_sass", side_effect=replace_during_dump
            ):
                result = sass_check.run(str(state), 1)

            manifest = compiler_evidence.load_manifest(iter_dir / "compiler_evidence")

        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["checks"][0]["verified"])
        self.assertEqual(manifest["sass"]["status"], "unavailable")

    def test_triton_sass_is_not_applicable_and_not_verified(self) -> None:
        sass_check = _load("sass_check")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            iter_dir = root / "iterv1"
            iter_dir.mkdir()
            (iter_dir / "kernel.py").write_text("# triton", encoding="utf-8")
            (iter_dir / "methods.json").write_text(
                json.dumps({"methods": [{"id": "triton_method"}]}), encoding="utf-8"
            )
            state = root / "state.json"
            state.write_text(json.dumps({"run_dir": str(root)}), encoding="utf-8")

            result = sass_check.run(str(state), 1)

        self.assertEqual(result["status"], "not_applicable")
        self.assertEqual(result["checks"][0]["status"], "not_applicable")
        self.assertFalse(result["checks"][0]["verified"])

    def test_unavailable_cuobjdump_is_not_reported_as_verified(self) -> None:
        sass_check = _load("sass_check")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            iter_dir = root / "iterv1"
            iter_dir.mkdir()
            (iter_dir / "kernel.cu").write_text("source", encoding="utf-8")
            (iter_dir / "kernel.so").write_bytes(b"elf")
            (iter_dir / "methods.json").write_text(
                json.dumps({"methods": [{"id": "vectorize"}]}), encoding="utf-8"
            )
            state = root / "state.json"
            state.write_text(json.dumps({"run_dir": str(root)}), encoding="utf-8")

            with mock.patch.object(
                sass_check, "_dump_sass", return_value="ERROR: cuobjdump not found"
            ):
                result = sass_check.run(str(state), 1)

            sass_path = iter_dir / "compiler_evidence" / "sass.txt"

        self.assertFalse(result["checks"][0]["verified"])
        self.assertEqual(result["checks"][0]["status"], "unavailable")
        self.assertFalse(sass_path.exists())

    def test_cuobjdump_nonzero_exit_is_reported_as_unavailable(self) -> None:
        sass_check = _load("sass_check")
        failed = SimpleNamespace(returncode=1, stdout="", stderr="bad object")
        with mock.patch.object(sass_check.subprocess, "run", return_value=failed):
            result = sass_check._dump_sass("kernel.so")

        self.assertTrue(result.startswith("ERROR:"), result)
        self.assertIn("bad object", result)

    def test_cuobjdump_empty_success_is_reported_as_unavailable(self) -> None:
        sass_check = _load("sass_check")
        empty = SimpleNamespace(returncode=0, stdout="", stderr="")
        with mock.patch.object(sass_check.subprocess, "run", return_value=empty):
            result = sass_check._dump_sass("kernel.so")

        self.assertTrue(result.startswith("ERROR:"), result)
        self.assertIn("no SASS", result)


if __name__ == "__main__":
    unittest.main()
