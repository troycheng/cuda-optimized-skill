from __future__ import annotations

import importlib.util
import json
import os
import tempfile
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
        self.assertEqual(updated["sass"]["status"], "available")

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

            def fake_run(command, **_kwargs):
                Path(command[command.index("-o") + 1]).write_bytes(b"elf")
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

        self.assertEqual(manifest["backend"], "cuda")
        self.assertEqual(manifest["arch"], "sm_120")
        self.assertEqual(manifest["binary"]["status"], "available")
        self.assertEqual(manifest["compile_command"][0], "nvcc")
        self.assertNotIn("-lineinfo", manifest["compile_command"])

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

        self.assertEqual(manifest["backend"], "triton")
        self.assertEqual(manifest["ttir"]["status"], "available")
        self.assertEqual(manifest["ptx"]["status"], "available")
        self.assertTrue(manifest["ptx"]["path"].endswith("hash/kernel.ptx"))
        self.assertEqual(manifest["binary"]["status"], "unavailable")

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
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            iter_dir = root / "iterv1"
            iter_dir.mkdir()
            source = iter_dir / "kernel.cu"
            binary = iter_dir / "kernel.so"
            source.write_text("source", encoding="utf-8")
            binary.write_bytes(b"elf")
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
