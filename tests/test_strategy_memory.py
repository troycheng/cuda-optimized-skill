from __future__ import annotations

import copy
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import multiprocessing
import os
import stat
import sys
import tempfile
import textwrap
import time
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


def _load_script(name: str):
    module_name = f"cuda_optimizer_strategy_memory_{name}_test"
    sys.modules.pop(module_name, None)
    path = MODULE_PATH.with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


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
        self.orchestrate = _load_script("orchestrate")
        self.state = _load_script("state")
        self.decision = _load_script("decision")
        self.paired_stats = _load_script("paired_stats")

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

    def _record(self, marker: str = "1", *, scope: dict | None = None) -> dict:
        scope = copy.deepcopy(self._scope() if scope is None else scope)
        candidate_sha = marker if len(marker) == 64 else marker * 64
        run_root = "/tmp/cuda-strategy-memory-test-run"
        identities = {
            "state": "1" * 64, "checkpoint": "e" * 64,
            "decision": "d" * 64, "methods": "2" * 64,
            "candidate": candidate_sha, "kernel_paired_samples": "3" * 64,
        }
        evidence = [
            {"role": role, "path": f"{run_root}/iterv1/{role}.json", "sha256": digest}
            for role, digest in identities.items()
        ]
        statistics = {
            "status": "inconclusive",
            "statistic": "median_paired_improvement_pct", "direction": "lower",
            "min_effect_pct": 1.0, "confidence": 0.95,
            "estimate_pct": 0.0, "ci_low_pct": -1.0, "ci_high_pct": 1.0,
            "valid_pairs": 1, "invalid_pairs": 0, "improvements_pct": [0.0],
        }
        decision_evidence = next(item for item in evidence if item["role"] == "decision")
        return {
            "schema_version": self.memory.RECORD_SCHEMA,
            "input_hash": scope["input_hash"], "candidate_sha256": candidate_sha,
            "decision_sha256": "d" * 64, "checkpoint_identity": "e" * 64,
            "run_root": {"path": run_root, "device": 1, "inode": 1},
            "scope": scope, "completed_at": 1.0,
            "terminal": {
                "status": "inconclusive", "mode": "kernel-only",
                "reason": "kernel evidence does not contain a confirmed inner-loop win",
                "statistics": statistics, "workload_status": None,
                "workload_statistics": None, "workload_failure": None,
                "constraints": [], "pareto": None,
            },
            "bundle": {
                "outcome": "inconclusive", "method_ids": [],
                "promotion_authority": False, "decision_evidence": decision_evidence,
            },
            "methods": {"performance": {}, "implementation": {}},
            "evidence": evidence,
        }

    def _suggestion_record(
        self,
        marker: str,
        method_id: str,
        *,
        performance: str | None = None,
        completed_at: float = 1.0,
        scope: dict | None = None,
        implementation: str | None = None,
    ) -> dict:
        record = self._record(marker, scope=scope)
        record["completed_at"] = completed_at
        record["bundle"]["method_ids"] = [method_id]
        if performance is not None:
            champion_ms, ablated_ms = (
                (10.0, 12.0) if performance == "positive" else (12.0, 10.0)
            )
            attribution_ms = ablated_ms - champion_ms
            evidence = []
            for role in (
                "attribution",
                "champion_bench",
                f"ablation_kernel:{method_id}",
                f"ablation_bench:{method_id}",
            ):
                identity = {
                    "role": role,
                    "path": str(
                        Path(record["run_root"]["path"])
                        / "iterv1"
                        / f"{marker}-{role.replace(':', '-')}.json"
                    ),
                    "sha256": hashlib.sha256(
                        f"{marker}:{role}".encode("utf-8")
                    ).hexdigest(),
                }
                evidence.append(identity)
                record["evidence"].append(identity)
            record["methods"]["performance"][method_id] = {
                "outcome": performance,
                "champion_ms": champion_ms,
                "ablated_ms": ablated_ms,
                "attribution_ms": attribution_ms,
                "attribution_pct": attribution_ms / champion_ms * 100.0,
                "evidence_quality": "diagnostic_unpaired_ablation",
                "promotion_authority": False,
                "evidence": evidence,
            }
        if implementation is not None:
            identity = {
                "role": "sass_check",
                "path": str(
                    Path(record["run_root"]["path"])
                    / "iterv1"
                    / f"{marker}-sass-check.json"
                ),
                "sha256": hashlib.sha256(
                    f"{marker}:sass_check".encode("utf-8")
                ).hexdigest(),
            }
            record["evidence"].append(identity)
            record["methods"]["implementation"][method_id] = {
                "status": implementation,
                "evidence": identity,
            }
        return self.memory._validate_record(record)

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

    def _completed_run(
        self,
        root: Path,
        status: str = "kernel_only_win",
        *,
        with_ablation: bool = False,
        workload_failed: bool = False,
        method_id: str = "vectorize",
        champion_ms: float = 10.0,
        ablated_ms: float = 12.0,
    ) -> Path:
        """Build a real terminal snapshot through the production validators."""
        run = root.resolve() / "run"
        iteration = run / "iterv1"
        iteration.mkdir(parents=True)
        baseline = root / "baseline.py"
        reference = root / "reference.py"
        baseline.write_text("# baseline\n", encoding="utf-8")
        reference.write_text("# reference\n", encoding="utf-8")
        candidate = iteration / "kernel.py"
        candidate.write_text("# candidate\n", encoding="utf-8")
        candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
        mode = "full" if status == "end_to_end_win" or workload_failed else "kernel-only"
        workload = self._python_workload(root, "-run") if mode == "full" else None
        budget = {"elapsed_seconds": 1.0, "remaining_seconds": 9.0}
        inputs = {
            "baseline": {
                "path": str(baseline.resolve()),
                "sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
                "size_bytes": baseline.stat().st_size,
            },
            "ref": {
                "path": str(reference.resolve()),
                "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                "size_bytes": reference.stat().st_size,
            },
        }
        manifest = {
            "schema_version": 2,
            "inputs": inputs,
            "environment": {"primary_sm_arch": "sm_120"},
            "backend": "triton",
            "dims": {"M": 128},
            "ptr_size": 8,
            "mode": mode,
            "workload": workload,
            "budget": budget,
            "confidence": 0.95,
            "min_effect_pct": 1.0,
            "started_at": 1.0,
        }
        manifest["input_hash"] = self.orchestrate._frozen_input_hash(
            {"inputs": inputs}, workload=workload, dims=manifest["dims"],
            backend=manifest["backend"], budget=budget, confidence=0.95,
            min_effect_pct=1.0, ptr_size=8,
        )
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        candidate_value = {
            "kernel_only_win": 97.0,
            "end_to_end_win": 97.0,
            "confirmed_loss": 103.0,
            "inconclusive": 100.0,
        }[status]
        kernel_pairs = [
            {"block": index, "baseline": 100.0, "candidate": candidate_value, "valid": True}
            for index in range(3)
        ]
        kernel_classifier = {
            "direction": "lower", "min_effect_pct": 1.0,
            "confidence": 0.95, "bootstrap_samples": 20, "seed": 0,
        }
        kernel_stats = self.paired_stats.classify_pairs(
            kernel_pairs, direction="lower", min_effect_pct=1.0,
            confidence=0.95, bootstrap_samples=20, seed=0,
        )
        kernel_artifact = self.orchestrate.write_paired_samples(
            iteration / "kernel" / "paired_samples.jsonl", kernel_pairs,
            kind="kernel", input_hash=manifest["input_hash"], iteration=1,
            candidate_id="b1", candidate_file=candidate,
            classifier_config=kernel_classifier,
        )
        kernel_evidence = {"status": kernel_stats["status"], "statistics": kernel_stats}
        workload_evidence = None
        workload_artifact = None
        if mode == "full":
            objective = workload["objective"]
            raw_workload_pairs = [
                {"block": index, "order": "AB", "case": None,
                 "baseline_metrics": {"latency_ms": 100.0},
                 "candidate_metrics": {"latency_ms": 97.0},
                 "valid": not workload_failed,
                 "attempts": {"baseline": 1, "candidate": 1},
                 "attempt_records": {"baseline": [], "candidate": []}}
                for index in range(3)
            ]
            workload_classifier = {
                "objective": objective,
                "objective_sha256": hashlib.sha256(json.dumps(
                    objective, sort_keys=True, separators=(",", ":")
                ).encode()).hexdigest(),
                "confidence": 0.95, "bootstrap_samples": 20, "seed": 0,
            }
            workload_artifact = self.orchestrate.write_paired_samples(
                iteration / "workload" / "paired_samples.jsonl", raw_workload_pairs,
                kind="workload", input_hash=manifest["input_hash"], iteration=1,
                candidate_id="b1", candidate_file=candidate,
                classifier_config=workload_classifier,
            )
            if workload_failed:
                workload_evidence = self.orchestrate.workload_evaluate.classify_recorded_pairs(
                    objective, raw_workload_pairs, confidence=0.95,
                    bootstrap_samples=20, seed=0,
                )
            else:
                workload_stats = self.paired_stats.classify_pairs(
                    [{"baseline": 100.0, "candidate": 97.0, "valid": True}] * 3,
                    direction="lower", min_effect_pct=1.0, confidence=0.95,
                    bootstrap_samples=20, seed=0,
                )
                workload_evidence = {
                    "status": "evaluated", "objective": objective,
                    "primary": workload_stats, "constraints": [],
                }
        decision = self.decision.decide(
            mode=mode, kernel=kernel_evidence, workload=workload_evidence,
            constraints=[], pareto=None,
        )
        self.assertEqual(decision["status"], status)
        decision.update({
            "candidate_id": "b1", "candidate_file": str(candidate.resolve()),
            "candidate_sha256": candidate_sha,
        })
        decision["kernel_paired_samples"] = kernel_artifact
        if status in {"confirmed_loss", "inconclusive"}:
            decision["statistics"] = kernel_stats
        if workload_artifact is not None:
            decision["workload_paired_samples"] = workload_artifact
        decision_path = iteration / "decision.json"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")
        methods = iteration / "methods.json"
        methods.write_text(json.dumps({"methods": [{"id": method_id}]}), encoding="utf-8")
        champion_bench = iteration / "bench.json"
        champion_bench.write_text(json.dumps({
            "correctness": {"passed": True}, "kernel": {"average_ms": champion_ms},
            "reference": {"average_ms": 20.0},
        }), encoding="utf-8")
        sass = iteration / "sass_check.json"
        sass.write_text(json.dumps({
            "status": "passed", "checks": [{
                "method_id": method_id, "status": "passed", "verified": True,
                "patterns_missing": [],
            }],
        }), encoding="utf-8")
        attribution_path = None
        if with_ablation:
            ablation = iteration / "ablations" / method_id.replace(".", "_")
            ablation.mkdir(parents=True)
            ablated_kernel = ablation / "kernel.py"
            ablated_kernel.write_text("# ablated\n", encoding="utf-8")
            ablated_bench = ablation / "bench.json"
            ablated_bench.write_text(json.dumps({
                "correctness": {"passed": True}, "kernel": {"average_ms": ablated_ms},
            }), encoding="utf-8")
            attribution_path = iteration / "attribution.json"
            attribution_path.write_text(json.dumps({
                "iter": 1, "attributions": [{
                    "method_id": method_id,
                    "champion_bench": str(champion_bench.resolve()),
                    "champion_bench_sha256": hashlib.sha256(champion_bench.read_bytes()).hexdigest(),
                    "ablated_kernel": str(ablated_kernel.resolve()),
                    "ablated_kernel_sha256": hashlib.sha256(ablated_kernel.read_bytes()).hexdigest(),
                    "ablated_bench": str(ablated_bench.resolve()),
                    "ablated_bench_sha256": hashlib.sha256(ablated_bench.read_bytes()).hexdigest(),
                    "champion_ms": round(champion_ms, 4),
                    "ablated_ms": round(ablated_ms, 4),
                    "attribution_ms": round(ablated_ms - champion_ms, 4),
                    "attribution_pct": round((ablated_ms - champion_ms) / champion_ms * 100.0, 2),
                    "contributed": True,
                }],
            }), encoding="utf-8")

        state_payload = {
            "schema_version": 2, "run_dir": str(run), "input_hash": manifest["input_hash"],
            "budget": budget, "candidates": {}, "mode": mode, "confidence": 0.95,
            "min_effect_pct": 1.0, "bootstrap_samples": 20, "seed": 0,
            "workload": workload, "best_file": str(baseline.resolve()),
            "best_metric_ms": 20.0, "best_kernel_statistics": None,
            "best_workload_statistics": None, "noise_threshold_pct": 2.0,
            "selected_methods": [], "effective_methods": [], "ineffective_methods": [],
            "implementation_failed_methods": [], "history": [], "roofline_history": [],
            "frontier": [], "ref_file": str(reference.resolve()), "dims": {"M": 128},
            "ptr_size": 8,
        }
        state_path = run / "state.json"
        state_path.write_text(json.dumps(state_payload), encoding="utf-8")
        checkpoint = {
            "schema_version": 2, "input_hash": manifest["input_hash"], "iteration": 1,
            "stage": "decision", "stage_index": self.orchestrate.STAGES.index("decision"),
            "status": "in_progress", "candidate_id": "b1", "candidate_status": status,
            "candidate_file": str(candidate.resolve()), "candidate_sha256": candidate_sha,
            "budget": budget, "updated_at": 2.0, "stage_evidence": {},
            "run_dir": str(run),
        }
        checkpoint_path = run / "checkpoint.json"
        self.orchestrate.ArtifactStore(run).write_checkpoint(checkpoint)
        args = argparse.Namespace(
            state=str(state_path), iter=1, kernel=str(candidate), bench=str(champion_bench),
            methods_json=str(methods), attribution=(str(attribution_path) if attribution_path else None),
            sass_check=str(sass), retries=0, skip_validation=True,
            allow_ineffective=False, decision=str(decision_path),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.state.cmd_update(args)
        current = json.loads(checkpoint_path.read_text("utf-8"))
        current = self.orchestrate.transition_checkpoint(
            current, "decision", status="stage_complete", updated_at=3.0,
        )
        self.orchestrate.ArtifactStore(run).write_checkpoint(current)
        self.state.persist_checkpoint_snapshot(state_path, current, checkpoint_path)
        complete = self.orchestrate.transition_checkpoint(
            current, "complete", status="complete", updated_at=4.0,
        )
        self.orchestrate.ArtifactStore(run).write_checkpoint(complete)
        self.state.persist_checkpoint_snapshot(state_path, complete, checkpoint_path)
        return run

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
                scoped_record = self._record(scope=scope)
                self.memory.append_run(memory, scope, scoped_record)
            before = memory.read_bytes()
            overflow = self._scope("f")
            overflow["input_hash"] = "f" * 64
            overflow_record = self._record(scope=overflow)
            with self.assertRaisesRegex(ValueError, "scope capacity"):
                self.memory.append_run(memory, overflow, overflow_record)
            self.assertEqual(memory.read_bytes(), before)

            run_memory = root / "run-cap.json"
            scope = self._scope()
            for index in range(self.memory.MAX_RUNS_PER_SCOPE):
                unique = self._record(f"{index:064x}")
                self.memory.append_run(run_memory, scope, unique)
            duplicate = self._record(f"{0:064x}")
            before_duplicate = run_memory.read_bytes()
            self.assertFalse(self.memory.append_run(run_memory, scope, duplicate))
            self.assertEqual(run_memory.read_bytes(), before_duplicate)
            new_record = self._record("f" * 64)
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
            with self.assertRaisesRegex(ValueError, "input_hash.*scope|scope.*input_hash"):
                self.memory.append_run(memory, scope, mismatch)
            self.assertEqual(memory.read_bytes(), original)

            corrupt = json.loads(original)
            entry = next(iter(corrupt["scopes"].values()))
            entry["runs"][0]["input_hash"] = "f" * 64
            memory.write_text(json.dumps(corrupt), encoding="utf-8")
            corrupt_bytes = memory.read_bytes()
            with self.assertRaisesRegex(ValueError, "input_hash.*scope|scope.*input_hash"):
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

    def test_completed_v22_runs_are_replayed_and_detached(self) -> None:
        for status in (
            "kernel_only_win", "end_to_end_win", "confirmed_loss", "inconclusive"
        ):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                run = self._completed_run(Path(tmp), status)
                record = self.memory.load_completed_run(run)
                self.assertEqual(record["schema_version"], self.memory.RECORD_SCHEMA)
                self.assertEqual(record["bundle"]["outcome"], status)
                self.assertEqual(record["input_hash"], record["scope"]["input_hash"])
                self.assertNotIn("environment", json.dumps(record))
                self.assertNotIn("# candidate", json.dumps(record))
                self.assertEqual(record["methods"]["performance"], {})

    def test_checkpoint_candidate_file_and_hash_must_match_terminal(self) -> None:
        for field in ("candidate_file", "candidate_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run = self._completed_run(root)
                alternate = run / "iterv1" / "alternate.py"
                alternate.write_text("# alternate valid candidate\n", encoding="utf-8")
                alternate_sha = hashlib.sha256(alternate.read_bytes()).hexdigest()
                checkpoint = run / "checkpoint.json"
                self._mutate_json(
                    checkpoint,
                    lambda value: value.update(**{
                        "candidate_file": str(alternate.resolve()),
                        "candidate_sha256": alternate_sha,
                    }),
                )
                # Keep the terminal resume binding valid: state intentionally does
                # not snapshot checkpoint file/hash, which is the gap under test.
                with self.assertRaisesRegex(ValueError, "checkpoint.*candidate"):
                    self.memory.record_run(root / "memory.json", run, root / "record.json")
                self.assertFalse((root / "memory.json").exists())

    def test_loss_and_inconclusive_require_bound_raw_kernel_pairs(self) -> None:
        for status in ("confirmed_loss", "inconclusive"):
            for mutation in ("missing", "tampered"):
                with self.subTest(status=status, mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    run = self._completed_run(root, status)
                    paired = run / "iterv1" / "kernel" / "paired_samples.jsonl"
                    if mutation == "missing":
                        paired.unlink()
                    else:
                        paired.write_text("{}\n", encoding="utf-8")
                    with self.assertRaises(ValueError):
                        self.memory.record_run(root / "memory.json", run, root / "record.json")
                    self.assertFalse((root / "memory.json").exists())
                    self.assertFalse((root / "record.json").exists())

    def test_completed_run_uses_all_authoritative_validators(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._completed_run(Path(tmp))
            with mock.patch.object(
                self.memory.orchestrate, "_load_and_verify_manifest",
                wraps=self.memory.orchestrate._load_and_verify_manifest,
            ) as manifest_validator, mock.patch.object(
                self.memory.state, "validate_state", wraps=self.memory.state.validate_state,
            ) as state_validator, mock.patch.object(
                self.memory.orchestrate, "_validate_checkpoint",
                wraps=self.memory.orchestrate._validate_checkpoint,
            ) as checkpoint_validator, mock.patch.object(
                self.memory.orchestrate, "_verify_state_candidates",
                wraps=self.memory.orchestrate._verify_state_candidates,
            ) as candidate_validator, mock.patch.object(
                self.memory.decision, "decide", wraps=self.memory.decision.decide,
            ) as replay:
                self.memory.load_completed_run(run)
            for called in (
                manifest_validator, state_validator, checkpoint_validator,
                candidate_validator, replay,
            ):
                self.assertTrue(called.called)

    def test_tamper_incomplete_and_legacy_fail_before_memory_or_output(self) -> None:
        mutators = {
            "paired": lambda run: (run / "iterv1" / "kernel" / "paired_samples.jsonl").write_text(
                "{}\n", encoding="utf-8"
            ),
            "decision": lambda run: (run / "iterv1" / "decision.json").write_text(
                "{}", encoding="utf-8"
            ),
            "candidate": lambda run: (run / "iterv1" / "kernel.py").write_text(
                "# drift\n", encoding="utf-8"
            ),
            "checkpoint": lambda run: self._mutate_json(
                run / "checkpoint.json", lambda value: value.update(stage="decision", stage_index=6, status="stage_complete")
            ),
            "legacy": lambda run: self._mutate_json(
                run / "state.json", lambda value: value.update(schema_version=1)
            ),
            "missing": lambda run: (run / "checkpoint.json").unlink(),
            "symlink": self._symlink_decision,
        }
        for name, mutate in mutators.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run = self._completed_run(root)
                memory = root / "memory.json"
                output = root / "record.json"
                mutate(run)
                with self.assertRaises(ValueError):
                    self.memory.record_run(memory, run, output)
                self.assertFalse(memory.exists())
                self.assertFalse(output.exists())

    @staticmethod
    def _symlink_decision(run: Path) -> None:
        path = run / "iterv1" / "decision.json"
        external = run.parent / "external-decision.json"
        external.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(external)

    def test_raw_pair_statistic_contradiction_is_recomputed_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._completed_run(root)
            paired = run / "iterv1" / "kernel" / "paired_samples.jsonl"
            records = [json.loads(line) for line in paired.read_text("utf-8").splitlines()]
            records[0]["pair"]["candidate"] = 120.0
            paired.write_text(
                "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in records),
                encoding="utf-8",
            )
            new_sha = hashlib.sha256(paired.read_bytes()).hexdigest()
            self._mutate_json(
                run / "state.json",
                lambda value: value["terminal_decision"]["kernel_paired_samples"].update(sha256=new_sha),
            )
            with self.assertRaisesRegex(ValueError, "recompute|statistics"):
                self.memory.record_run(root / "memory.json", run, root / "record.json")
            self.assertFalse((root / "memory.json").exists())

    @staticmethod
    def _mutate_json(path: Path, mutate) -> None:
        value = json.loads(path.read_text("utf-8"))
        mutate(value)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_decision_without_replay_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._completed_run(Path(tmp))
            decision_path = run / "iterv1" / "decision.json"
            self._mutate_json(decision_path, lambda value: value.pop("evidence"))
            decision_sha = hashlib.sha256(decision_path.read_bytes()).hexdigest()
            self._mutate_json(
                run / "state.json",
                lambda value: value["terminal_decision"].update(decision_sha256=decision_sha),
            )
            with self.assertRaisesRegex(ValueError, "evidence|replay"):
                self.memory.load_completed_run(run)

    def test_valid_ablation_is_diagnostic_and_sass_is_implementation_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._completed_run(Path(tmp), with_ablation=True)
            record = self.memory.load_completed_run(run)
            performance = record["methods"]["performance"]["vectorize"]
            self.assertEqual(performance["outcome"], "positive")
            self.assertEqual(performance["evidence_quality"], "diagnostic_unpaired_ablation")
            self.assertFalse(performance["promotion_authority"])
            implementation = record["methods"]["implementation"]["vectorize"]
            self.assertEqual(implementation["status"], "passed")
            self.assertNotIn("outcome", implementation)

    def test_producer_rounded_dotted_method_ablation_imports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._completed_run(
                Path(tmp), with_ablation=True, method_id="memory.coalesce",
                champion_ms=10.123456, ablated_ms=12.654321,
            )
            record = self.memory.load_completed_run(run)
            performance = record["methods"]["performance"]["memory.coalesce"]
            self.assertEqual(performance["outcome"], "positive")
            self.assertEqual(performance["champion_ms"], 10.123456)
            self.assertEqual(performance["ablated_ms"], 12.654321)

    def test_sass_file_must_match_terminal_snapshot(self) -> None:
        for mutation in ("missing", "changed"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                run = self._completed_run(Path(tmp))
                sass = run / "iterv1" / "sass_check.json"
                if mutation == "missing":
                    sass.unlink()
                else:
                    self._mutate_json(sass, lambda value: value.update(status="failed"))
                with self.assertRaisesRegex(ValueError, "SASS|sass"):
                    self.memory.load_completed_run(run)

    def test_terminal_reason_must_match_replayed_reason_when_snapshotted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._completed_run(Path(tmp))
            self._mutate_json(
                run / "state.json",
                lambda value: value["terminal_decision"].update(reason="tampered reason"),
            )
            with self.assertRaisesRegex(ValueError, "reason|replay"):
                self.memory.load_completed_run(run)

    def test_legacy_four_hash_records_are_rejected_from_append_and_load(self) -> None:
        legacy = {
            "input_hash": "a" * 64, "candidate_sha256": "1" * 64,
            "decision_sha256": "d" * 64, "checkpoint_identity": "e" * 64,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = root / "memory.json"
            with self.assertRaises(ValueError):
                self.memory.append_run(memory, self._scope(), legacy)
            payload = {
                "schema_version": self.memory.MEMORY_SCHEMA,
                "scopes": {self.memory.scope_key(self._scope()): {
                    "scope": self._scope(), "runs": [legacy],
                    "methods": {}, "bundles": {},
                }},
            }
            memory.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                self.memory.load_memory(memory)

    def test_invalid_ablation_never_creates_performance_evidence(self) -> None:
        for field in (
            "ablated_bench_sha256", "attribution_pct", "correctness",
            "champion_correctness_false", "champion_correctness_missing",
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                run = self._completed_run(Path(tmp), with_ablation=True)
                attribution = run / "iterv1" / "attribution.json"
                if field == "correctness":
                    bench = run / "iterv1" / "ablations" / "vectorize" / "bench.json"
                    self._mutate_json(bench, lambda value: value["correctness"].update(passed=False))
                    self._mutate_json(
                        attribution,
                        lambda value: value["attributions"][0].update(
                            ablated_bench_sha256=hashlib.sha256(bench.read_bytes()).hexdigest()
                        ),
                    )
                elif field.startswith("champion_correctness"):
                    bench = run / "iterv1" / "bench.json"
                    if field.endswith("false"):
                        self._mutate_json(
                            bench, lambda value: value["correctness"].update(passed=False)
                        )
                    else:
                        self._mutate_json(
                            bench, lambda value: value.pop("correctness")
                        )
                    self._mutate_json(
                        attribution,
                        lambda value: value["attributions"][0].update(
                            champion_bench_sha256=hashlib.sha256(bench.read_bytes()).hexdigest()
                        ),
                    )
                elif field == "ablated_bench_sha256":
                    self._mutate_json(
                        attribution,
                        lambda value: value["attributions"][0].pop(field),
                    )
                else:
                    self._mutate_json(
                        attribution,
                        lambda value: value["attributions"][0].update(attribution_pct=99.0),
                    )
                record = self.memory.load_completed_run(run)
                self.assertEqual(record["methods"]["performance"], {})

    def test_verified_record_and_store_indices_reject_fabricated_evidence(self) -> None:
        mutations = {
            "bundle-outcome": lambda value: value["bundle"].update(outcome="confirmed_loss"),
            "promotion": lambda value: value["bundle"].update(promotion_authority=True),
            "method-order": lambda value: value["bundle"].update(method_ids=["z", "a"]),
            "decision-hash": lambda value: value["bundle"]["decision_evidence"].update(sha256="0" * 64),
            "evidence-extra": lambda value: value["evidence"][0].update(extra=True),
            "performance-zero": lambda value: value["methods"]["performance"]["vectorize"].update(
                attribution_ms=0.0, attribution_pct=0.0
            ),
            "performance-quality": lambda value: value["methods"]["performance"]["vectorize"].update(
                evidence_quality="claimed"
            ),
            "implementation-performance": lambda value: value["methods"]["implementation"]["vectorize"].update(
                outcome="positive"
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                run = self._completed_run(Path(tmp), with_ablation=True)
                record = self.memory.load_completed_run(run)
                mutate(record)
                with self.assertRaises(ValueError):
                    self.memory._validate_record(record)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._completed_run(root, with_ablation=True)
            memory = root / "memory.json"
            self.memory.record_run(memory, run, root / "record.json")
            payload = json.loads(memory.read_text("utf-8"))
            entry = next(iter(payload["scopes"].values()))
            next(iter(entry["bundles"].values()))["records"][0]["outcome"] = "inconclusive"
            memory.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "index|bundle"):
                self.memory.load_memory(memory)

    def test_verified_record_core_evidence_is_unique_and_hash_bound(self) -> None:
        singleton_roles = (
            "state", "checkpoint", "decision", "methods", "candidate",
            "kernel_paired_samples",
        )
        cases = []
        for role in singleton_roles:
            cases.append((f"missing-{role}", lambda value, role=role: value["evidence"].__setitem__(
                slice(None), [item for item in value["evidence"] if item["role"] != role]
            )))
            cases.append((f"duplicate-{role}", lambda value, role=role: self._duplicate_evidence_role(value, role)))
        cases.extend(
            [
                ("candidate-sha", lambda value: next(
                    item for item in value["evidence"] if item["role"] == "candidate"
                ).update(sha256="0" * 64)),
                ("checkpoint-sha", lambda value: next(
                    item for item in value["evidence"] if item["role"] == "checkpoint"
                ).update(sha256="0" * 64)),
                ("unexpected-workload", lambda value: value["evidence"].append({
                    "role": "workload_paired_samples",
                    "path": str(Path(value["run_root"]["path"]) / "iterv1" / "extra-workload.jsonl"),
                    "sha256": "1" * 64,
                })),
            ]
        )
        for name, mutate in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                record = self.memory.load_completed_run(
                    self._completed_run(Path(tmp), with_ablation=True)
                )
                mutate(record)
                with self.assertRaises(ValueError):
                    self.memory._validate_record(record)

    def test_failed_full_workload_retains_exactly_one_bound_paired_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._completed_run(
                root, "kernel_only_win", workload_failed=True
            )
            record = self.memory.load_completed_run(run)
            self.assertEqual(record["terminal"]["mode"], "full")
            self.assertEqual(record["terminal"]["workload_status"], "workload_failed")
            self.assertEqual(
                sum(item["role"] == "workload_paired_samples" for item in record["evidence"]),
                1,
            )
            for mutation in ("missing", "duplicate"):
                corrupt = copy.deepcopy(record)
                if mutation == "missing":
                    corrupt["evidence"] = [
                        item for item in corrupt["evidence"]
                        if item["role"] != "workload_paired_samples"
                    ]
                else:
                    self._duplicate_evidence_role(corrupt, "workload_paired_samples")
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    self.memory._validate_record(corrupt)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._completed_run(
                root, "kernel_only_win", workload_failed=True
            )
            (run / "iterv1" / "workload" / "paired_samples.jsonl").unlink()
            with self.assertRaises(ValueError):
                self.memory.record_run(root / "memory.json", run, root / "record.json")
            self.assertFalse((root / "memory.json").exists())

        for mutation in ("missing", "duplicate"):
            with self.subTest(workload=mutation), tempfile.TemporaryDirectory() as tmp:
                record = self.memory.load_completed_run(
                    self._completed_run(Path(tmp), "end_to_end_win")
                )
                if mutation == "missing":
                    record["evidence"] = [
                        item for item in record["evidence"]
                        if item["role"] != "workload_paired_samples"
                    ]
                else:
                    self._duplicate_evidence_role(record, "workload_paired_samples")
                with self.assertRaises(ValueError):
                    self.memory._validate_record(record)

    @staticmethod
    def _duplicate_evidence_role(record: dict, role: str) -> None:
        original = next(item for item in record["evidence"] if item["role"] == role)
        duplicate = dict(original)
        duplicate["path"] = str(Path(record["run_root"]["path"]) / "iterv1" / f"duplicate-{role}")
        duplicate["sha256"] = "1" * 64
        record["evidence"].append(duplicate)

    def test_record_writes_memory_before_output_and_retry_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._completed_run(root)
            memory = root / "memory.json"
            output = root / "record.json"
            with mock.patch.object(
                self.memory, "_atomic_write_output", side_effect=OSError("output failed")
            ), self.assertRaisesRegex(OSError, "output failed"):
                self.memory.record_run(memory, run, output)
            stored = self.memory.load_memory(memory)
            entry = next(iter(stored["scopes"].values()))
            self.assertEqual(len(entry["runs"]), 1)
            result = self.memory.record_run(memory, run, output)
            self.assertFalse(result["inserted"])
            self.assertEqual(len(next(iter(self.memory.load_memory(memory)["scopes"].values()))["runs"]), 1)
            self.assertEqual(json.loads(output.read_text("utf-8"))["schema_version"], self.memory.RECORD_SCHEMA)

    def test_suggestion_uses_only_exact_scope_and_traceable_ablation_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            manifest = self._manifest(root)
            scope = self.memory.scope_document(manifest)
            memory = root / "memory.json"
            records = [
                self._suggestion_record("1", "preferred", performance="positive", completed_at=4.0, scope=scope),
                self._suggestion_record("2", "preferred", performance="positive", completed_at=2.0, scope=scope),
                self._suggestion_record("3", "negative", performance="negative", completed_at=3.0, scope=scope),
                self._suggestion_record("4", "conflicting", performance="positive", completed_at=5.0, scope=scope),
                self._suggestion_record("5", "conflicting", performance="negative", completed_at=6.0, scope=scope),
                self._suggestion_record("6", "inconclusive", completed_at=7.0, scope=scope, implementation="unavailable"),
                self._suggestion_record("7", "bundle-only", completed_at=1.0, scope=scope),
            ]
            for record in records:
                self.memory.append_run(memory, scope, record)

            other_scope = self._scope("f")
            for marker in ("8", "9"):
                self.memory.append_run(
                    memory,
                    other_scope,
                    self._suggestion_record(
                        marker,
                        "other-scope",
                        performance="positive",
                        completed_at=8.0,
                        scope=other_scope,
                    ),
                )

            output = root / "suggestion.json"
            suggestion = self.memory.suggest_strategies(memory, manifest, output)
            self.assertEqual(
                suggestion["schema_version"],
                "cuda-kernel-optimizer/strategy-suggestion-v1",
            )
            self.assertEqual(suggestion["status"], "available")
            self.assertEqual(suggestion["preferred_method_ids"], ["preferred"])
            self.assertEqual(
                suggestion["caution_method_ids"], ["conflicting", "negative"]
            )
            self.assertNotIn("inconclusive", suggestion["caution_method_ids"])
            self.assertNotIn("bundle-only", suggestion["method_evidence"])
            self.assertNotIn("other-scope", suggestion["method_evidence"])

            preferred = suggestion["method_evidence"]["preferred"]
            self.assertEqual(preferred["count"], 2)
            self.assertEqual(preferred["positive_count"], 2)
            self.assertEqual(preferred["negative_count"], 0)
            for item in preferred["records"]:
                self.assertEqual(item["count"], 1)
                self.assertRegex(item["record_identity"], r"^[0-9a-f]{64}$")
                self.assertEqual(item["decision_evidence"]["role"], "decision")
                self.assertRegex(
                    item["decision_evidence"]["sha256"], r"^[0-9a-f]{64}$"
                )
                self.assertEqual(
                    {entry["role"] for entry in item["ablation_evidence"]},
                    {
                        "attribution",
                        "champion_bench",
                        "ablation_kernel:preferred",
                        "ablation_bench:preferred",
                    },
                )
                for evidence in item["ablation_evidence"]:
                    self.assertTrue(os.path.isabs(evidence["path"]))
                    self.assertRegex(evidence["sha256"], r"^[0-9a-f]{64}$")

            conflicting = suggestion["method_evidence"]["conflicting"]
            self.assertEqual(conflicting["positive_count"], 1)
            self.assertEqual(conflicting["negative_count"], 1)
            self.assertEqual(
                [item["outcome"] for item in conflicting["records"]],
                ["positive", "negative"],
            )
            bundles = suggestion["prior_bundles"]
            self.assertEqual(
                [(item["completed_at"], item["record_identity"]) for item in bundles],
                sorted(
                    (item["completed_at"], item["record_identity"])
                    for item in bundles
                ),
            )
            self.assertEqual(len(bundles), len(records))
            self.assertTrue(any(item["method_ids"] == ["bundle-only"] for item in bundles))
            for item in bundles:
                self.assertEqual(item["count"], 1)
                self.assertEqual(item["decision_evidence"]["role"], "decision")
                self.assertRegex(item["decision_evidence"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(json.loads(output.read_text("utf-8")), suggestion)

    def test_suggestion_is_read_only_and_has_fixed_advisory_guardrail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            manifest = self._manifest(root, mode="full")
            scope = self.memory.scope_document(manifest)
            memory = root / "memory.json"
            self.memory.append_run(
                memory,
                scope,
                self._suggestion_record(
                    "1", "vectorize", performance="positive", scope=scope
                ),
            )
            run_files = [
                manifest,
                root / "state.json",
                root / "checkpoint.json",
                root / "decision.json",
                root / "candidate.py",
            ]
            for path in run_files[1:]:
                path.write_bytes(f"do-not-read-or-change:{path.name}".encode("utf-8"))
            watched = [memory, *run_files]
            before = {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in watched
            }
            output = root / "suggestion.json"
            real_read = self.memory.artifact_store.read_regular_bytes

            def read_manifest_only(path):
                if os.path.abspath(path) != os.path.abspath(manifest):
                    raise AssertionError(f"suggest read outside manifest/memory: {path}")
                return real_read(path)

            with mock.patch.object(
                self.memory.orchestrate,
                "main",
                side_effect=AssertionError("suggest must not call orchestrator"),
            ), mock.patch.object(
                self.memory.artifact_store,
                "read_regular_bytes",
                side_effect=read_manifest_only,
            ):
                suggestion = self.memory.suggest_strategies(memory, manifest, output)
            after = {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in watched
            }
            self.assertEqual(after, before)
            self.assertTrue(suggestion["advisory"])
            self.assertEqual(suggestion["guardrail"], self.memory.ADVISORY_GUARDRAIL)
            for required in (
                "delete or prune branches",
                "profiler",
                "budget",
                "correctness",
                "sanitizer",
                "paired",
                "workload",
                "decision gate",
                "promotion",
            ):
                self.assertIn(required, suggestion["guardrail"])

    def test_missing_memory_publishes_unavailable_advisory_and_cli_is_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            manifest = self._manifest(root)
            memory = root / "missing-memory.json"
            output = root / "suggestion.json"
            exit_code = self.memory.main(
                [
                    "suggest",
                    "--memory",
                    str(memory),
                    "--manifest",
                    str(manifest),
                    "--out",
                    str(output),
                ]
            )
            self.assertNotEqual(exit_code, 0)
            suggestion = json.loads(output.read_text("utf-8"))
            self.assertEqual(suggestion["status"], "unavailable")
            self.assertTrue(suggestion["advisory"])
            self.assertEqual(suggestion["preferred_method_ids"], [])
            self.assertEqual(suggestion["caution_method_ids"], [])
            self.assertEqual(suggestion["prior_bundles"], [])

    def test_parser_has_only_explicit_record_and_suggest_paths(self) -> None:
        parser = self.memory.build_parser()
        action = next(
            item
            for item in parser._actions
            if isinstance(item, argparse._SubParsersAction)
        )
        self.assertEqual(set(action.choices), {"record", "suggest"})
        for command, required in (
            ("record", {"memory", "run_dir", "out"}),
            ("suggest", {"memory", "manifest", "out"}),
        ):
            subparser = action.choices[command]
            options = {
                item.dest: item.required
                for item in subparser._actions
                if item.dest != "help"
            }
            self.assertEqual(set(options), required)
            self.assertTrue(all(options.values()))
            self.assertNotIn("orchestrator", subparser.format_help().lower())

    def test_suggestion_output_rejects_symlink_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            manifest = self._manifest(root)
            scope = self.memory.scope_document(manifest)
            memory = root / "memory.json"
            self.memory.append_run(memory, scope, self._record(scope=scope))
            target = root / "target.json"
            target.write_text("unchanged", encoding="utf-8")
            output = root / "suggestion.json"
            output.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink|unsafe"):
                self.memory.suggest_strategies(memory, manifest, output)
            self.assertEqual(target.read_text("utf-8"), "unchanged")

    @staticmethod
    def _tree_snapshot(root: Path) -> dict[str, bytes]:
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    def test_suggest_rejects_output_aliases_without_mutating_inputs(self) -> None:
        for alias in (
            "manifest-exact",
            "manifest-relative",
            "manifest-hardlink",
            "memory-exact",
            "missing-memory-exact",
        ):
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                manifest = self._manifest(root)
                scope = self.memory.scope_document(manifest)
                memory = root / "memory.json"
                if alias != "missing-memory-exact":
                    self.memory.append_run(memory, scope, self._record(scope=scope))
                if alias == "manifest-exact":
                    output = manifest
                elif alias == "manifest-relative":
                    output = Path(os.path.relpath(manifest, Path.cwd()))
                elif alias == "manifest-hardlink":
                    output = root / "manifest-hardlink.json"
                    try:
                        os.link(manifest, output)
                    except OSError as error:
                        self.skipTest(f"hardlinks unavailable: {error}")
                else:
                    output = memory
                before = self._tree_snapshot(root)
                with self.assertRaisesRegex(ValueError, "output.*alias|collision"):
                    self.memory.suggest_strategies(memory, manifest, output)
                self.assertEqual(self._tree_snapshot(root), before)

    def test_record_rejects_memory_output_alias_before_append(self) -> None:
        for alias in ("missing-exact", "existing-hardlink"):
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                run = self._completed_run(root / "fixture")
                memory = root / "memory.json"
                if alias == "missing-exact":
                    output = memory
                else:
                    self.memory.append_run(
                        memory, self._scope(), self._record()
                    )
                    output = root / "memory-hardlink.json"
                    try:
                        os.link(memory, output)
                    except OSError as error:
                        self.skipTest(f"hardlinks unavailable: {error}")
                before = self._tree_snapshot(root)
                with self.assertRaisesRegex(ValueError, "output.*alias|collision"):
                    self.memory.record_run(memory, run, output)
                self.assertEqual(self._tree_snapshot(root), before)

    def test_record_rejects_every_verified_artifact_output_before_append(self) -> None:
        roles = (
            "manifest",
            "state",
            "checkpoint",
            "decision",
            "methods",
            "candidate",
            "kernel_paired_samples",
            "sass_check",
            "attribution",
            "champion_bench",
            "ablation_kernel:vectorize",
            "ablation_bench:vectorize",
        )
        for role in roles:
            with self.subTest(role=role), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                run = self._completed_run(root / "fixture", with_ablation=True)
                record = self.memory.load_completed_run(run)
                if role == "manifest":
                    output = run / "manifest.json"
                else:
                    output = Path(
                        next(
                            item["path"]
                            for item in record["evidence"]
                            if item["role"] == role
                        )
                    )
                memory = root / "memory.json"
                before = self._tree_snapshot(root)
                with self.assertRaisesRegex(ValueError, "output.*alias|collision"):
                    self.memory.record_run(memory, run, output)
                self.assertEqual(self._tree_snapshot(root), before)

    def test_record_rejects_hardlink_to_verified_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            run = self._completed_run(root / "fixture")
            decision_path = run / "iterv1" / "decision.json"
            output = root / "decision-hardlink.json"
            try:
                os.link(decision_path, output)
            except OSError as error:
                self.skipTest(f"hardlinks unavailable: {error}")
            memory = root / "memory.json"
            before = self._tree_snapshot(root)
            with self.assertRaisesRegex(ValueError, "output.*alias|collision"):
                self.memory.record_run(memory, run, output)
            self.assertEqual(self._tree_snapshot(root), before)


if __name__ == "__main__":
    unittest.main()
