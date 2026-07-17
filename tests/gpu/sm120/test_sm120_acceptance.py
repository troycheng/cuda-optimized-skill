from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
BENCHMARK = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "benchmark.py"
CHECK_ENV = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "check_env.py"
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"
RUNNER = Path(__file__).resolve().parent / "remote" / "run_lane.sh"
ARTIFACTS = Path(os.environ.get("CUDA_E2E_ARTIFACTS", "/tmp/cuda-sm120-acceptance"))
WORKLOAD_CONTROLLER = SCRIPTS / "workload_controller.py"


@dataclass(frozen=True)
class Case:
    name: str
    solution: str
    reference: str
    dims: tuple[str, ...]
    extra: tuple[str, ...] = ()


CASES = (
    Case(
        "triton_vector",
        "triton_vector.py",
        "triton_vector_ref.py",
        ("--N=1048576",),
    ),
    Case(
        "cuda_reduction",
        "cuda_reduction.cu",
        "cuda_reduction_ref.py",
        ("--N=1048576",),
        ("--ptr-size", "1048576"),
    ),
    Case(
        "cutlass_gemm",
        "cutlass_gemm.cu",
        "cutlass_gemm_ref.py",
        ("--M=512", "--N=512", "--K=512"),
        ("--ptr-size", "262144"),
    ),
)


def _run(command: list[str], json_path: Path) -> dict:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        old = json_path.lstat()
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISLNK(old.st_mode):
            raise AssertionError(f"refusing symlink JSON output: {json_path}")
        if not stat.S_ISREG(old.st_mode):
            raise AssertionError(f"refusing non-regular JSON output: {json_path}")
        json_path.unlink()
    started_ns = time.time_ns()
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(json_path, flags)
    except OSError as error:
        raise AssertionError(f"fresh JSON output is missing or unsafe: {json_path}") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise AssertionError(f"fresh JSON output is not regular: {json_path}")
        if opened.st_mtime_ns < started_ns:
            raise AssertionError(f"JSON output is not fresh: {json_path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if any(getattr(opened, field) != getattr(after, field) for field in identity_fields):
            raise AssertionError(f"JSON output changed while reading: {json_path}")
    finally:
        os.close(descriptor)
    try:
        return json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AssertionError(f"fresh JSON output is invalid: {json_path}") from error


def _load_script(name: str):
    scripts = str(SCRIPTS)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    module = importlib.import_module(name)
    expected = (SCRIPTS / f"{name}.py").resolve(strict=True)
    actual = Path(module.__file__).resolve(strict=True)
    if actual != expected:
        raise ImportError(f"canonical module {name} resolved outside optimizer scripts")
    return module


def _fixture_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage_fixture_workspace(
    name: str, *, artifacts=ARTIFACTS, fixtures=FIXTURES
) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or any(
            not (character.isalnum() or character in "._-")
            for character in name
        )
    ):
        raise ValueError("workspace name must contain only letters, digits, ., _, or -")
    artifact_root = Path(artifacts).expanduser().resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_root = artifact_root.resolve(strict=True)
    fixture_root = Path(fixtures).expanduser().resolve(strict=True)
    if not fixture_root.is_dir():
        raise ValueError("fixtures must be a directory")
    case_root = artifact_root / name
    if case_root.is_symlink():
        raise ValueError("artifact case directory must not be a symlink")
    case_root.mkdir(parents=True, exist_ok=True)
    if case_root.resolve(strict=True).parent != artifact_root:
        raise ValueError("artifact case directory escapes the artifact root")
    workspace = case_root / "workspace"
    if workspace.is_symlink():
        raise ValueError("artifact workspace must not be a symlink")
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(
        fixture_root,
        workspace,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.pyo", "*.so", "compiler_evidence", ".*"
        ),
    )
    return workspace


class Sm120AcceptanceHelperTests(unittest.TestCase):
    def test_fixture_workspace_copies_relative_dependencies_under_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixtures = root / "source-fixtures"
            fixtures.mkdir()
            (fixtures / "kernel.cu").write_text("// source", encoding="utf-8")
            (fixtures / "reference.py").write_text("# reference", encoding="utf-8")
            (fixtures / "nested").mkdir()
            (fixtures / "nested" / "helper.py").write_text(
                "VALUE = 1", encoding="utf-8"
            )
            artifacts = root / "artifacts"

            workspace = _stage_fixture_workspace(
                "cuda_case", artifacts=artifacts, fixtures=fixtures
            )
            evidence = workspace / "compiler_evidence" / "manifest.json"
            evidence.parent.mkdir()
            evidence.write_text("{}", encoding="utf-8")

            self.assertEqual(
                workspace, artifacts.resolve() / "cuda_case" / "workspace"
            )
            self.assertEqual((workspace / "kernel.cu").read_text("utf-8"), "// source")
            self.assertTrue((workspace / "nested" / "helper.py").is_file())
            self.assertTrue(evidence.is_file())
            self.assertFalse((fixtures / "compiler_evidence").exists())

    def test_script_loader_reuses_canonical_workload_spec_class(self) -> None:
        workload_adapter = _load_script("workload_adapter")
        workload_evaluate = _load_script("workload_evaluate")

        self.assertIs(workload_adapter, sys.modules["workload_adapter"])
        self.assertIs(workload_evaluate, sys.modules["workload_evaluate"])
        self.assertIs(
            workload_adapter.WorkloadSpec, workload_evaluate.WorkloadSpec
        )

    def test_workload_acceptance_uses_production_outer_evidence_seam(self) -> None:
        source = inspect.getsource(
            Sm120AcceptanceTests.test_user_workload_smoke_produces_separate_outer_raw_evidence
        )

        self.assertIn("evaluate_outer_candidate", source)
        self.assertIn("workload_paired_samples", source)
        self.assertNotIn("write_paired_samples", source)

    def test_run_never_reuses_a_stale_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            output.write_text('{"stale":true}', encoding="utf-8")

            with self.assertRaisesRegex(AssertionError, "fresh|missing"):
                _run([sys.executable, "-c", "pass"], output)

            self.assertFalse(output.exists())

    def test_run_rejects_symlink_output_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "outside.json"
            output = root / "result.json"
            target.write_text('{"outside":true}', encoding="utf-8")
            try:
                output.symlink_to(target)
            except OSError:
                self.skipTest("symlinks are unavailable")

            with self.assertRaisesRegex(AssertionError, "symlink"):
                _run([sys.executable, "-c", "pass"], output)

            self.assertEqual(target.read_text("utf-8"), '{"outside":true}')

    def test_runner_contract_is_fail_closed_and_uses_immutable_image(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")

        self.assertNotIn("|| true", source)
        self.assertGreaterEqual(source.count("assert_gpu_idle"), 3)
        self.assertIn("resolved_image_id", source)
        self.assertIn("requested_ref", source)
        self.assertIn('"$resolved_image_id"', source)
        self.assertIn("must be empty", source)

    def test_runner_restricts_cutlass_to_the_dedicated_checkout(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn(
            '/data/tcheng/cuda-skill-e2e/deps/cutlass', source
        )
        self.assertIn("include/cutlass/cutlass.h", source)
        self.assertIn("include/cutlass/version.h", source)
        self.assertIn('expected_cutlass_version="4.6.1"', source)
        self.assertIn("vllm-opt", source)
        self.assertIn('-v "$repo_root:$repo_root:ro"', source)

    def test_v2_4_lane_uses_the_production_workload_controller(self) -> None:
        source = inspect.getsource(
            Sm120AcceptanceTests.test_workload_controller_promotes_real_gpu_candidate
        )
        for marker in ("start_run", "register_change", "evaluate_change"):
            self.assertIn(marker, source)
        for fixture in (
            "triton_vector_slow.py",
            "workload_probe.py",
            "workload_smoke.py",
        ):
            self.assertTrue((FIXTURES / fixture).is_file(), fixture)
        adapter = (FIXTURES / "workload_smoke.py").read_text(encoding="utf-8")
        self.assertIn("Mapping", adapter)
        self.assertIn('candidate["path"]', adapter)

@unittest.skipUnless(
    os.environ.get("CUDA_SM120_E2E") == "1",
    "set CUDA_SM120_E2E=1 on an SM120 CUDA host",
)
class Sm120AcceptanceTests(unittest.TestCase):
    def test_environment_and_fixture_matrix(self) -> None:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        env_path = ARTIFACTS / "env.json"
        _run([sys.executable, str(CHECK_ENV), "--out", str(env_path)], env_path)
        env = json.loads(env_path.read_text(encoding="utf-8"))
        self.assertEqual(env["primary_sm_arch"], "sm_120")

        for case in CASES:
            with self.subTest(case=case.name):
                workspace = _stage_fixture_workspace(case.name)
                result_path = ARTIFACTS / case.name / "bench.json"
                command = [
                    sys.executable,
                    str(BENCHMARK),
                    str(workspace / case.solution),
                    "--ref",
                    str(workspace / case.reference),
                    "--warmup",
                    "3",
                    "--repeat",
                    "8",
                    "--json-out",
                    str(result_path),
                    *case.extra,
                    *case.dims,
                ]
                result = _run(command, result_path)
                self.assertTrue(result["correctness"]["passed"])
                self.assertIn("samples_ms", result["kernel"])
                self.assertGreater(len(result["kernel"]["samples_ms"]), 4)
                self.assertIn("median_ms", result["kernel"])
                self.assertIn("p95_ms", result["kernel"])
                self.assertIn("stddev_ms", result["kernel"])
                self.assertIn("cv_pct", result["kernel"])

    def test_paired_noop_is_inconclusive_with_recomputable_raw_pairs(self) -> None:
        paired_benchmark = _load_script("paired_benchmark")
        paired_stats = _load_script("paired_stats")
        artifact_store = _load_script("artifact_store")
        pair_count = int(os.environ.get("CUDA_E2E_NOOP_PAIRS", "30"))
        workspace = _stage_fixture_workspace("paired_noop")
        solution = workspace / "triton_vector.py"
        classifier = {
            "direction": "lower",
            "min_effect_pct": 0.5,
            "confidence": 0.95,
            "bootstrap_samples": 2000,
            "seed": 17,
        }

        raw = paired_benchmark.run_paired(
            str(solution),
            str(solution),
            backend="triton",
            dims={"N": 1_048_576},
            ptr_size=0,
            arch="sm_120",
            nvcc_bin="nvcc",
            seed=classifier["seed"],
            blocks=pair_count,
            warmup=5,
        )
        statistics = paired_stats.classify_pairs(raw["pairs"], **classifier)
        evidence = artifact_store.write_paired_samples(
            ARTIFACTS / "paired_noop" / "paired_samples.jsonl",
            raw["pairs"],
            kind="kernel",
            input_hash=_fixture_hash(solution),
            iteration=1,
            candidate_id="triton-identical-noop",
            candidate_file=solution,
            classifier_config=classifier,
        )
        artifact_store.atomic_write_json(
            ARTIFACTS / "paired_noop" / "statistics.json", statistics
        )

        records = artifact_store.ArtifactStore(ARTIFACTS).read_jsonl(
            "paired_noop/paired_samples.jsonl"
        )
        recomputed = paired_stats.classify_pairs(
            [record["pair"] for record in records], **classifier
        )
        self.assertEqual(statistics["status"], "inconclusive")
        self.assertEqual(recomputed, statistics)
        self.assertEqual(len(records), pair_count)
        self.assertGreater(statistics["valid_pairs"], 0)
        self.assertTrue(
            all(record["kind"] == "kernel" for record in records)
        )
        self.assertTrue(
            all(record["pair"]["order"] in {"AB", "BA"} for record in records)
        )

    def test_user_workload_smoke_produces_separate_outer_raw_evidence(self) -> None:
        orchestrate = _load_script("orchestrate")
        case_root = ARTIFACTS / "workload_smoke"
        iter_dir = case_root / "iterv1"
        workspace = _stage_fixture_workspace("iterv1", artifacts=case_root)
        solution = workspace / "triton_vector.py"
        objective_path = workspace / "objective.json"
        workload_path = workspace / "workload_smoke.py"
        objective = orchestrate.workload_evaluate.validate_objective(
            json.loads(objective_path.read_text(encoding="utf-8"))
        )
        workload = orchestrate.normalize_workload(workload=str(workload_path))
        self.assertIsNotNone(workload)
        self.assertEqual(workload.objective, objective)
        kernel_pairs = [
            {"baseline": 100.0, "candidate": 98.0, "valid": True}
            for _ in range(20)
        ]
        kernel_statistics = orchestrate.workload_evaluate.paired_stats.classify_pairs(
            kernel_pairs,
            direction="lower",
            min_effect_pct=0.5,
            confidence=0.95,
            bootstrap_samples=2000,
            seed=23,
        )
        self.assertEqual(kernel_statistics["status"], "confirmed_win")
        candidate = {
            "id": "triton-workload-noop",
            "status": "confirmed_win",
            "candidate_file": str(solution),
            "statistics": kernel_statistics,
        }
        pair_count = int(os.environ.get("CUDA_E2E_WORKLOAD_PAIRS", "8"))
        policy = orchestrate.resolve_budget(
            "custom",
            max_seconds=900,
            max_rounds=1,
            branches=1,
            min_pairs=pair_count,
            max_pairs=pair_count,
            outer_candidates=1,
            reserve_seconds=30,
        )
        input_hash = hashlib.sha256(
            f"{workload.source_hash}:{_fixture_hash(solution)}".encode("utf-8")
        ).hexdigest()

        terminal = orchestrate.evaluate_outer_candidate(
            candidate,
            mode="full",
            workload_spec=workload,
            baseline=str(solution),
            policy=policy,
            confidence=0.95,
            candidate_root=iter_dir,
            input_hash=input_hash,
            iteration=1,
            retries=0,
            seed=23,
        )
        evidence = terminal["workload_paired_samples"]
        expected_path = (
            iter_dir
            / "workload"
            / terminal["candidate_sha256"][:16]
            / "paired_samples.jsonl"
        )
        self.assertEqual(Path(evidence["path"]), expected_path.resolve(strict=True))
        records = orchestrate.ArtifactStore(iter_dir).read_jsonl(
            str(expected_path.relative_to(iter_dir))
        )
        classifier = evidence["classifier"]
        recomputed = orchestrate.workload_evaluate.classify_recorded_pairs(
            classifier["objective"],
            [record["pair"] for record in records],
            confidence=classifier["confidence"],
            bootstrap_samples=classifier["bootstrap_samples"],
            seed=classifier["seed"],
        )
        self.assertEqual(terminal["status"], "kernel_only_win")
        self.assertNotEqual(terminal["status"], "end_to_end_win")
        self.assertEqual(terminal["workload_status"], "evaluated")
        self.assertEqual(terminal["workload_statistics"]["status"], "inconclusive")
        self.assertEqual(recomputed["primary"], terminal["workload_statistics"])
        self.assertEqual(recomputed["constraints"], terminal["constraints"])
        terminal_status, promote_global = orchestrate.state_manager._promotion_for(
            terminal["status"], "full"
        )
        self.assertEqual(terminal_status, "kernel_only_win")
        self.assertFalse(promote_global)
        self.assertGreater(len(records), 0)
        self.assertTrue(
            all(record["kind"] == "workload" for record in records)
        )
        self.assertTrue(
            all(
                "latency_ms" in record["pair"]["baseline_metrics"]
                and "latency_ms" in record["pair"]["candidate_metrics"]
                for record in records
            )
        )

    def test_ncu_target_profile_records_success_or_explicit_degradation(self) -> None:
        profile_root = ARTIFACTS / "ncu_target"
        workspace = _stage_fixture_workspace("ncu_target")
        env_path = profile_root / "env.json"
        state_path = profile_root / "state.json"
        _run([sys.executable, str(CHECK_ENV), "--out", str(env_path)], env_path)
        env = json.loads(env_path.read_text(encoding="utf-8"))
        state = {
            "run_dir": str(profile_root.resolve()),
            "best_file": str((workspace / "triton_vector.py").resolve()),
            "env": env,
            "env_path": str(env_path.resolve()),
            "dims": {"N": 1_048_576},
            "ptr_size": 0,
            "ncu_num": 5,
        }
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state), encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "profile_ncu.py"),
                "--state",
                str(state_path),
                "--iter",
                "1",
                "--which",
                "best_input",
                "--benchmark",
                str(BENCHMARK),
            ],
            text=True,
            capture_output=True,
            check=True,
        )

        top = json.loads(
            (profile_root / "iterv1" / "ncu_top.json").read_text(encoding="utf-8")
        )
        recorded = json.loads(state_path.read_text(encoding="utf-8"))
        ncu = recorded["env"]["ncu"]
        if top["degraded"] is False:
            self.assertIs(ncu.get("can_read_counters"), True)
            self.assertGreater(top.get("metric_count_collected", 0), 0)
            self.assertGreater(
                sum(len(top.get(axis, [])) for axis in ("compute", "memory", "latency")),
                0,
            )
        elif top["degraded"] is True:
            self.assertIs(ncu.get("can_read_counters"), False)
            self.assertEqual(
                ncu.get("counter_access_error"), "ERR_NVGPUCTRPERM"
            )
        else:
            self.fail("ncu_top.degraded must be a literal boolean")

    def test_workload_controller_promotes_real_gpu_candidate(self) -> None:
        controller = _load_script("workload_controller")
        case_root = ARTIFACTS / "workload_controller"
        workspace = _stage_fixture_workspace("workspace", artifacts=case_root)
        baseline = workspace / "baseline.py"
        candidate = workspace / "candidate.py"
        shutil.copy2(workspace / "triton_vector_slow.py", baseline)
        shutil.copy2(workspace / "triton_vector_slow.py", candidate)
        workload_manifest = workspace / "controller_workload.json"
        workload_manifest.write_text(
            json.dumps(
                {
                    "kind": "python",
                    "source": "workload_smoke.py",
                    "objective": json.loads(
                        (workspace / "objective.json").read_text("utf-8")
                    ),
                    "cases": [{"N": 1_048_576}],
                }
            ),
            encoding="utf-8",
        )
        control = {
            "schema_version": "cuda-workload-optimizer/control-v1",
            "project_root": str(workspace),
            "workload_manifest": str(workload_manifest),
            "baseline_candidate": {
                "name": "slow-baseline",
                "revision": "fixture-slow",
                "path": str(baseline),
            },
            "budget": "fast",
            "mutation": {
                "project_paths": ["candidate.py"],
                "environment_root": str(case_root / "isolated-environment"),
                "host_policy": "recommend_only",
            },
            "probes": [
                {
                    "id": "timeline",
                    "kind": "timeline",
                    "argv": [sys.executable, str(workspace / "workload_probe.py")],
                    "timeout_seconds": 60,
                }
            ],
        }
        run_dir = case_root / "run"
        state = controller.start_run(control, run_dir)
        self.assertEqual(state["next_action"], "register_change")
        probe = json.loads((run_dir / "probes" / "timeline.json").read_text("utf-8"))
        diagnosis = json.loads((run_dir / "diagnosis.json").read_text("utf-8"))
        self.assertEqual(probe["schema_version"], "cuda-workload-optimizer/probe-v1")
        self.assertEqual(
            diagnosis["schema_version"], "cuda-workload-optimizer/diagnosis-v1"
        )

        change = {
            "schema_version": "cuda-workload-optimizer/change-v1",
            "id": "round-1-remove-redundant-launches",
            "hypothesis": "Two redundant launches dominate this fixture workload.",
            "diagnosis_ids": diagnosis["diagnosis_ids"] or ["kernel:fixture"],
            "scope": "project",
            "candidate": {
                "name": "single-launch",
                "revision": "fixture-fast",
                "path": str(candidate),
            },
            "paths": ["candidate.py"],
            "commands": [[sys.executable, "-m", "py_compile", str(candidate)]],
            "rollback": "restore_frozen_snapshot",
            "expected_metrics": ["gpu_busy_pct", "latency_ms", "output_checksum"],
        }
        controller.register_change(control, run_dir, change)
        slow_hash = _fixture_hash(candidate)
        shutil.copy2(workspace / "triton_vector.py", candidate)
        self.assertNotEqual(_fixture_hash(candidate), slow_hash)

        decision = controller.evaluate_change(run_dir)

        self.assertEqual(decision["status"], "promoted")
        evaluation = json.loads((run_dir / "evaluation.json").read_text("utf-8"))
        review = json.loads((run_dir / "review.json").read_text("utf-8"))
        self.assertEqual(evaluation["primary"]["status"], "confirmed_win")
        self.assertTrue(
            all(item["status"] == "passed" for item in evaluation["constraints"])
        )
        self.assertEqual(review["status"], "skipped")
        self.assertEqual(controller.evaluate_change(run_dir), decision)
        self.assertEqual(controller.resume_run(run_dir)["next_action"], "done")
        self.assertIn(
            "No host mutation was executed",
            (run_dir / "host_recommendations.md").read_text("utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
