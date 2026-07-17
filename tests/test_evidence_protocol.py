from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"
MODULE_PATH = SCRIPT_DIR / "evidence_protocol.py"


def _load_module():
    name = "cuda_optimizer_evidence_protocol_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _guard_policy() -> dict:
    return {
        "schema_version": "cuda-evidence/guard-policy-v1",
        "formal": True,
        "sample_interval_ms": 100,
        "max_sample_gap_ms": 150,
        "joint_clean_window_ms": 200,
        "gpus": {
            "target": {"uuid": "GPU-target", "pci_bus_id": "0000:01:00.0"},
            "peers": [{"uuid": "GPU-peer", "pci_bus_id": "0000:02:00.0"}],
            "siblings": [
                {"uuid": "GPU-sibling", "pci_bus_id": "0000:03:00.0"}
            ],
        },
        "cpu": {"cpus": [0, 1], "numa_nodes": [0]},
        "allowlist": {"pids": [111], "containers": ["allowed-container"]},
        "limits": {
            "min_sm_clock_mhz": 1000,
            "max_temperature_c": 80,
            "max_power_w": 400,
            "forbidden_throttle_reasons": ["thermal", "power"],
            "max_swap_used_bytes": 0,
            "max_memory_pressure_pct": 10,
            "max_foreign_cpu_pct": 5,
            "max_foreign_gpu_pct": 1,
        },
        "phase_requirements": {
            "correctness": "required",
            "sanitizer": "not_applicable",
            "diagnostic": "required",
            "timing": "required",
        },
        "not_applicable_reasons": {"sanitizer": "no applicable sanitizer"},
    }


def _sample(monotonic_ms: int) -> dict:
    roles = (
        ("target", "GPU-target", "0000:01:00.0"),
        ("peer", "GPU-peer", "0000:02:00.0"),
        ("sibling", "GPU-sibling", "0000:03:00.0"),
    )
    return {
        "monotonic_ms": monotonic_ms,
        "gpus": [
            {
                "role": role,
                "uuid": uuid,
                "pci_bus_id": pci,
                "sm_clock_mhz": 2100,
                "temperature_c": 60,
                "power_w": 250,
                "throttle_reasons": [],
                "foreign_gpu_pct": 0,
                "processes": [],
            }
            for role, uuid, pci in roles
        ],
        "cpu": {"cpus": [0, 1], "numa_nodes": [0], "foreign_cpu_pct": 0},
        "memory": {"swap_used_bytes": 0, "pressure_pct": 0},
        "contamination_markers": [],
    }


def _clean_samples() -> list[dict]:
    return [_sample(timestamp) for timestamp in range(100, 1200, 100)]


def _phase_markers() -> list[dict]:
    return [
        {
            "phase": "correctness",
            "watcher_ready_ms": 300,
            "start_ms": 300,
            "end_ms": 500,
        },
        {
            "phase": "diagnostic",
            "watcher_ready_ms": 600,
            "start_ms": 600,
            "end_ms": 800,
        },
        {
            "phase": "timing",
            "watcher_ready_ms": 900,
            "start_ms": 900,
            "end_ms": 1100,
        },
    ]


def _sha(character: str) -> str:
    return character * 64


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _execution_path() -> dict:
    return {
        "schema_version": "cuda-evidence/execution-path-v1",
        "expected_cases": ["c1", "c2"],
        "case_hits": [
            {"case_id": "c1", "hit_count": 4},
            {"case_id": "c2", "hit_count": 2},
        ],
        "proof_kind": "dispatch_counter",
        "trace_sha256": _sha("a"),
        "diagnostic_binary": {
            "sha256": _sha("b"),
            "source_sha256": _sha("c"),
            "build_config_sha256": _sha("d"),
            "diagnostic_features": ["dispatch_counter"],
        },
        "timed_binary": {
            "sha256": _sha("e"),
            "source_sha256": _sha("c"),
            "build_config_sha256": _sha("d"),
            "diagnostic_features": [],
        },
        "rebuilt_after_diagnostics": True,
        "timed_binary_bound": True,
    }


def _serving_experiment(protocols=("http", "grpc")) -> dict:
    metrics = [
        "qps",
        "avg_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "server_input_ms",
        "server_infer_ms",
        "server_output_ms",
    ]
    strata = []
    for protocol in protocols:
        for concurrency in (1, 2, 4, 8, 12):
            strata.append(
                {
                    "id": f"{protocol}-c{concurrency}",
                    "protocol": protocol,
                    "concurrency": concurrency,
                    "warmup_requests": 20,
                    "measured_requests": 200,
                    "metrics": list(metrics),
                    "must_pass": {
                        "relative": [
                            {
                                "metric": "p99_latency_ms",
                                "comparison": "max_regression",
                                "direction": "lower",
                                "limit_pct": 2.0,
                            }
                        ],
                        "absolute": [
                            {"metric": "error_rate_pct", "operator": "<=", "limit": 0.1}
                        ],
                    },
                }
            )
    return {
        "schema_version": "cuda-evidence/serving-experiment-v1",
        "protocols": list(protocols),
        "fresh_process_per_role": True,
        "request_corpus_sha256": _sha("f"),
        "strata": strata,
    }


def _artifact_identities() -> dict:
    return {
        "schema_version": "cuda-evidence/artifact-identities-v1",
        "source": {"sha256": _sha("1")},
        "binary": {
            "sha256": _sha("2"),
            "source_sha256": _sha("1"),
            "build_config_sha256": _sha("3"),
        },
        "plugin": {
            "sha256": _sha("4"),
            "source_sha256": _sha("5"),
            "compiler_version": "nvcc 13.3",
            "abi": "TensorRT-10",
        },
        "engine": {
            "sha256": _sha("6"),
            "plugin_sha256": _sha("4"),
            "builder_version": "TensorRT 10.12",
            "runtime_version": "TensorRT 10.12",
            "tactic_digest": _sha("7"),
            "timing_cache_digest": _sha("8"),
        },
        "backend": {"sha256": _sha("9"), "version": "1.0", "abi": "triton-v1"},
        "server": {"sha256": _sha("a"), "version": "25.07", "abi": "http-grpc-v2"},
        "image": {"digest": "sha256:" + _sha("b"), "tag": "server:latest"},
    }


def _profiler_bundle() -> dict:
    target = _sha("2")
    return {
        "schema_version": "cuda-evidence/profiler-bundle-v1",
        "authority": "non_promotional",
        "target_binary_sha256": target,
        "reports": [
            {
                "tool": "nsys",
                "version": "2026.2",
                "status": "available",
                "report_sha256": _sha("c"),
                "command": ["nsys", "profile", "./server"],
                "target_binary_sha256": target,
            },
            {
                "tool": "ncu",
                "version": "2026.2.1",
                "status": "unavailable",
                "report_sha256": None,
                "command": ["ncu", "--target-processes", "all", "./server"],
                "target_binary_sha256": target,
            },
        ],
        "observations": ["timeline shows launch gaps"],
        "limitations": ["NCU counters unavailable; bundle is explanatory only"],
    }


def _formal_design() -> dict:
    return {
        "schema_version": "cuda-evidence/experiment-design-v1",
        "formal": True,
        "schedule": [
            {"pair_id": "pair-1", "order": "BA"},
            {"pair_id": "pair-2", "order": "AB"},
        ],
        "experimental_unit": "fresh_process_pair",
        "aggregation": "median_paired_improvement",
        "resampling_unit": "pair",
        "ci": {
            "method": "paired_bootstrap",
            "confidence": 0.95,
            "samples": 100,
            "seed": 7,
        },
        "min_valid_pairs": 2,
        "wins_required": 2,
        "guardrails": {
            "relative": [
                {
                    "metric": "p99_latency_ms",
                    "comparison": "max_regression",
                    "direction": "lower",
                    "limit_pct": 2.0,
                }
            ],
            "absolute": [
                {"metric": "error_rate_pct", "operator": "<=", "limit": 0.1}
            ],
        },
        "exclusion_policy": "no_exclusion",
        "retry_policy": {
            "role_retries": 0,
            "whole_pair_only": True,
            "allowed_reasons": ["pre_measurement_infrastructure_failure"],
        },
    }


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _build_attempt(
    root: Path,
    *,
    state: str = "valid",
    claim_layer: str = "serving_endpoint",
) -> Path:
    files = {
        "runner": ("runner.py", "print('runner')\n"),
        "guard": ("guard.py", "print('guard')\n"),
        "analysis": ("analysis.py", "print('analysis')\n"),
        "schedule": ("schedule.json", json.dumps(_formal_design()["schedule"])),
        "source": ("source.cu", "extern \"C\" __global__ void k() {}\n"),
        "diagnostic_binary": ("diagnostic-kernel.so", "diagnostic-binary\n"),
        "binary": ("kernel.so", "timed-binary\n"),
        "profiler_nsys": ("profile.nsys-rep", "nsys-report\n"),
        "plugin": ("plugin.so", "plugin\n"),
        "engine": ("engine.plan", "engine\n"),
        "backend": ("backend.so", "backend\n"),
        "server": ("server.bin", "server\n"),
        "image": ("image.txt", "sha256:" + _sha("b") + "\n"),
    }
    artifacts = []
    for kind, (name, contents) in files.items():
        if claim_layer != "serving_endpoint" and kind in {
            "plugin",
            "engine",
            "backend",
            "server",
            "image",
        }:
            continue
        (root / name).write_text(contents, encoding="utf-8")
        artifacts.append({"id": kind, "kind": kind, "path": name})

    execution_path = _execution_path()
    execution_path["diagnostic_binary"].update(
        sha256=_file_sha(root / "diagnostic-kernel.so"),
        source_sha256=_file_sha(root / "source.cu"),
    )
    execution_path["timed_binary"].update(
        sha256=_file_sha(root / "kernel.so"),
        source_sha256=_file_sha(root / "source.cu"),
    )
    identities = _artifact_identities()
    identities["source"]["sha256"] = _file_sha(root / "source.cu")
    identities["binary"].update(
        sha256=_file_sha(root / "kernel.so"),
        source_sha256=_file_sha(root / "source.cu"),
    )
    if claim_layer == "serving_endpoint":
        for kind in ("plugin", "engine", "backend", "server"):
            identities[kind]["sha256"] = _file_sha(root / files[kind][0])
        identities["engine"]["plugin_sha256"] = identities["plugin"]["sha256"]
    profiler = _profiler_bundle()
    profiler["target_binary_sha256"] = _file_sha(root / "kernel.so")
    for report in profiler["reports"]:
        report["target_binary_sha256"] = _file_sha(root / "kernel.so")
        if report["tool"] == "nsys":
            report["report_sha256"] = _file_sha(root / "profile.nsys-rep")

    json_artifacts = {
        "experiment_design": ("experiment.json", _formal_design()),
        "guard_policy": ("guard-policy.json", _guard_policy()),
        "phase_markers": ("phase-markers.json", _phase_markers()),
        "execution_path": ("execution-path.json", execution_path),
        "serving_experiment": ("serving.json", _serving_experiment()),
        "artifact_identities": ("identities.json", identities),
        "profiler_bundle": ("profiler.json", profiler),
        "performance_verdict": (
            "performance.json",
            {
                "schema_version": "cuda-evidence/performance-verdict-v1",
                "status": "confirmed_win",
                "promotional_eligible": True,
                "analysis_sha256": _file_sha(root / "analysis.py"),
                "experiment_design_sha256": None,
                "raw_rows_sha256": None,
            },
        ),
    }
    for kind, (name, payload) in json_artifacts.items():
        if claim_layer != "serving_endpoint" and kind in {
            "serving_experiment",
            "artifact_identities",
        }:
            continue
        _write_json(root / name, payload)
        artifacts.append({"id": kind, "kind": kind, "path": name})

    samples_path = root / "guard-samples.jsonl"
    samples_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in _clean_samples()),
        encoding="utf-8",
    )
    artifacts.append(
        {"id": "guard_samples", "kind": "guard_samples", "path": samples_path.name}
    )
    raw_path = root / "raw-rows.jsonl"
    raw_rows = [
        {
            "pair_id": "pair-1",
            "order": "BA",
            "valid": True,
            "attempts": {"baseline": 1, "candidate": 1},
        },
        {
            "pair_id": "pair-2",
            "order": "AB",
            "valid": True,
            "attempts": {"baseline": 1, "candidate": 1},
        },
    ]
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw_rows),
        encoding="utf-8",
    )
    artifacts.append({"id": "raw_rows", "kind": "raw_rows", "path": raw_path.name})

    verdict = json.loads((root / "performance.json").read_text())
    verdict["experiment_design_sha256"] = _file_sha(root / "experiment.json")
    verdict["raw_rows_sha256"] = _file_sha(raw_path)
    _write_json(root / "performance.json", verdict)

    manifest = {
        "schema_version": "cuda-evidence/attempt-v1",
        "attempt_id": "attempt-001",
        "state": state,
        "claim_layer": claim_layer,
        "artifacts": artifacts,
    }
    path = root / "attempt.json"
    _write_json(path, manifest)
    return path


class StrictInputTests(unittest.TestCase):
    def test_duplicate_keys_and_nonfinite_constants_are_rejected(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
            nonfinite = root / "nonfinite.json"
            nonfinite.write_text('{"a": NaN}\n', encoding="utf-8")

            for path in (duplicate, nonfinite):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    module.load_json_strict(path)


class SharedHostGuardTests(unittest.TestCase):
    def test_clean_continuous_phase_bound_attempt_passes(self) -> None:
        module = _load_module()

        result = module.audit_shared_host_guard(
            _guard_policy(), _clean_samples(), _phase_markers()
        )

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["evidence_integrity"], "PASS")
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["phase_results"]["sanitizer"]["status"], "not_applicable")
        for phase in ("correctness", "diagnostic", "timing"):
            self.assertEqual(result["phase_results"][phase]["status"], "PASS")

    def test_unknown_gap_and_watcher_fail_closed(self) -> None:
        module = _load_module()
        cases = []

        missing_metric = _clean_samples()
        del missing_metric[5]["memory"]["pressure_pct"]
        cases.append((missing_metric, _phase_markers(), "pressure_pct"))

        gap = _clean_samples()
        del gap[5]
        cases.append((gap, _phase_markers(), "sample_gap"))

        missing_ready = _phase_markers()
        del missing_ready[2]["watcher_ready_ms"]
        cases.append((_clean_samples(), missing_ready, "watcher_ready"))

        short_clean_window = _phase_markers()
        short_clean_window[0]["watcher_ready_ms"] = 200
        cases.append((_clean_samples(), short_clean_window, "joint_clean_window"))

        for samples, markers, reason in cases:
            with self.subTest(reason=reason):
                result = module.audit_shared_host_guard(
                    _guard_policy(), samples, markers
                )
                self.assertEqual(result["status"], "FAIL")
                self.assertTrue(any(reason in item for item in result["reasons"]))

    def test_identity_cpu_actor_and_contamination_drift_fail_closed(self) -> None:
        module = _load_module()
        mutators = (
            (lambda rows: rows[5]["gpus"][0].update(uuid="GPU-other"), "identity"),
            (lambda rows: rows[5]["gpus"].append(copy.deepcopy(rows[5]["gpus"][0])), "identity"),
            (lambda rows: rows[5]["cpu"].update(cpus=[0, 2]), "cpu_affinity"),
            (
                lambda rows: rows[5]["gpus"][0]["processes"].append(
                    {"pid": 999, "container_id": "foreign"}
                ),
                "foreign_process",
            ),
            (
                lambda rows: rows[5]["contamination_markers"].append("maintenance"),
                "contamination_marker",
            ),
        )

        for mutate, reason in mutators:
            samples = _clean_samples()
            mutate(samples)
            with self.subTest(reason=reason):
                result = module.audit_shared_host_guard(
                    _guard_policy(), samples, _phase_markers()
                )
                self.assertEqual(result["status"], "FAIL")
                self.assertTrue(any(reason in item for item in result["reasons"]))

    def test_allowlisted_actor_passes_but_every_resource_limit_is_enforced(self) -> None:
        module = _load_module()
        allowed = _clean_samples()
        allowed[5]["gpus"][0]["processes"].append(
            {"pid": 999, "container_id": "allowed-container"}
        )
        self.assertEqual(
            module.audit_shared_host_guard(
                _guard_policy(), allowed, _phase_markers()
            )["status"],
            "PASS",
        )

        mutators = (
            (lambda rows: rows[5]["gpus"][0].update(sm_clock_mhz=999), "clock"),
            (lambda rows: rows[5]["gpus"][0].update(temperature_c=81), "temperature"),
            (lambda rows: rows[5]["gpus"][0].update(power_w=401), "power"),
            (
                lambda rows: rows[5]["gpus"][0]["throttle_reasons"].append("thermal"),
                "throttle",
            ),
            (lambda rows: rows[5]["memory"].update(swap_used_bytes=1), "swap"),
            (lambda rows: rows[5]["memory"].update(pressure_pct=11), "memory_pressure"),
            (lambda rows: rows[5]["cpu"].update(foreign_cpu_pct=6), "foreign_cpu"),
            (lambda rows: rows[5]["gpus"][0].update(foreign_gpu_pct=2), "foreign_gpu"),
        )

        for mutate, reason in mutators:
            samples = _clean_samples()
            mutate(samples)
            with self.subTest(reason=reason):
                result = module.audit_shared_host_guard(
                    _guard_policy(), samples, _phase_markers()
                )
                self.assertEqual(result["status"], "FAIL")
                self.assertTrue(any(reason in item for item in result["reasons"]))

    def test_policy_is_closed_and_formal_unknown_cannot_be_downgraded(self) -> None:
        module = _load_module()
        for mutate in (
            lambda policy: policy.update(formal=False),
            lambda policy: policy.update(extra=True),
            lambda policy: policy["phase_requirements"].pop("timing"),
            lambda policy: policy["not_applicable_reasons"].clear(),
        ):
            policy = copy.deepcopy(_guard_policy())
            mutate(policy)
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                module.audit_shared_host_guard(
                    policy, _clean_samples(), _phase_markers()
                )


class ExecutionPathCoverageTests(unittest.TestCase):
    def test_complete_hits_and_rebuilt_residue_free_timed_binary_pass(self) -> None:
        module = _load_module()

        result = module.validate_execution_path(_execution_path())

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["timed_binary_sha256"], _sha("e"))

    def test_missing_hit_diagnostic_reuse_and_identity_drift_fail_closed(self) -> None:
        module = _load_module()
        mutators = (
            lambda item: item["case_hits"].pop(),
            lambda item: item["case_hits"][0].update(hit_count=0),
            lambda item: item["timed_binary"].update(sha256=_sha("b")),
            lambda item: item["timed_binary"].update(source_sha256=_sha("0")),
            lambda item: item["timed_binary"]["diagnostic_features"].append("trace"),
            lambda item: item.update(rebuilt_after_diagnostics=False),
            lambda item: item.update(timed_binary_bound=False),
        )
        for mutate in mutators:
            payload = _execution_path()
            mutate(payload)
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                module.validate_execution_path(payload)


class ServingIdentityAndProfilerTests(unittest.TestCase):
    def test_serving_experiment_requires_all_strata_and_metrics(self) -> None:
        module = _load_module()

        result = module.validate_serving_experiment(_serving_experiment())

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["strata_count"], 10)

    def test_serving_experiment_rejects_missing_strata_nonfresh_and_guardrail_gaps(self) -> None:
        module = _load_module()
        mutators = (
            lambda item: item["strata"].pop(),
            lambda item: item.update(fresh_process_per_role=False),
            lambda item: item["strata"][0].update(warmup_requests=0),
            lambda item: item["strata"][0]["metrics"].remove("p99_latency_ms"),
            lambda item: item["strata"][0]["must_pass"].update(relative=[]),
            lambda item: item["strata"][0]["must_pass"].update(absolute=[]),
        )
        for mutate in mutators:
            payload = _serving_experiment()
            mutate(payload)
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                module.validate_serving_experiment(payload)

    def test_serving_artifact_identity_is_complete_and_version_bound(self) -> None:
        module = _load_module()

        result = module.validate_artifact_identities(_artifact_identities())

        self.assertEqual(result["status"], "PASS")
        self.assertNotEqual(result["source_sha256"], result["binary_sha256"])

    def test_tag_only_or_missing_binary_engine_tactic_cache_identity_is_rejected(self) -> None:
        module = _load_module()
        mutators = (
            lambda item: item["image"].update(digest="server:latest"),
            lambda item: item["binary"].pop("sha256"),
            lambda item: item["engine"].pop("sha256"),
            lambda item: item["engine"].pop("tactic_digest"),
            lambda item: item["engine"].pop("timing_cache_digest"),
            lambda item: item["engine"].update(plugin_sha256=_sha("0")),
        )
        for mutate in mutators:
            payload = _artifact_identities()
            mutate(payload)
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                module.validate_artifact_identities(payload)

    def test_profiler_bundle_is_binary_bound_and_never_promotional(self) -> None:
        module = _load_module()

        result = module.validate_profiler_bundle(_profiler_bundle())

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["authority"], "non_promotional")

    def test_profiler_bundle_rejects_promotional_authority_or_incomplete_tools(self) -> None:
        module = _load_module()
        mutators = (
            lambda item: item.update(authority="promotion"),
            lambda item: item["reports"].pop(),
            lambda item: item["reports"][0].update(target_binary_sha256=_sha("0")),
            lambda item: item["reports"][0].update(report_sha256=None),
        )
        for mutate in mutators:
            payload = _profiler_bundle()
            mutate(payload)
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                module.validate_profiler_bundle(payload)


class AttemptLifecycleTests(unittest.TestCase):
    def test_isolated_operator_valid_attempt_does_not_require_serving_artifacts(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = _build_attempt(root, claim_layer="isolated_operator")

            module.seal_attempt(attempt, root / "seal.json")
            module.audit_seal(root / "seal.json", root / "audit.json")
            result = module.decide_attempt(
                root / "seal.json",
                root / "audit.json",
                root / "decision.json",
                root / "evidence-manifest.json",
            )

            self.assertEqual(result["decision"], "promote")

    def test_valid_attempt_closes_seal_audit_decision_manifest_without_conflation(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = _build_attempt(root)

            seal = module.seal_attempt(attempt, root / "seal.json")
            audit = module.audit_seal(root / "seal.json", root / "audit.json")
            decision = module.decide_attempt(
                root / "seal.json",
                root / "audit.json",
                root / "decision.json",
                root / "evidence-manifest.json",
            )

            self.assertEqual(seal["attempt_state"], "valid")
            self.assertEqual(seal["evidence_integrity"], "not_audited")
            self.assertEqual(audit["evidence_integrity"], "PASS")
            self.assertNotIn("performance_verdict", audit)
            self.assertEqual(decision["performance_verdict"], "confirmed_win")
            self.assertEqual(decision["evidence_integrity"], "PASS")
            self.assertEqual(decision["decision"], "promote")
            closure = json.loads((root / "evidence-manifest.json").read_text())
            self.assertEqual(set(closure["evidence_refs"]), {"seal", "audit", "decision"})
            for name in ("seal", "audit", "decision"):
                self.assertEqual(len(closure["evidence_refs"][name]["sha256"]), 64)

            with self.assertRaises(FileExistsError):
                module.seal_attempt(attempt, root / "seal.json")
            with self.assertRaises(FileExistsError):
                module.audit_seal(root / "seal.json", root / "audit.json")

    def test_tamper_after_seal_fails_integrity_and_can_never_promote(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = _build_attempt(root)
            module.seal_attempt(attempt, root / "seal.json")
            (root / "runner.py").write_text("tampered\n", encoding="utf-8")

            audit = module.audit_seal(root / "seal.json", root / "audit.json")
            decision = module.decide_attempt(
                root / "seal.json",
                root / "audit.json",
                root / "decision.json",
                root / "evidence-manifest.json",
            )

            self.assertEqual(audit["evidence_integrity"], "FAIL")
            self.assertEqual(decision["performance_verdict"], "confirmed_win")
            self.assertEqual(decision["evidence_integrity"], "FAIL")
            self.assertEqual(decision["decision"], "retain")

    def test_decision_recomputes_integrity_instead_of_trusting_forged_audit(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = _build_attempt(root)
            module.seal_attempt(attempt, root / "seal.json")
            (root / "runner.py").write_text("tampered\n", encoding="utf-8")
            _write_json(
                root / "forged-audit.json",
                {
                    "schema_version": "cuda-evidence/audit-v1",
                    "attempt_id": "attempt-001",
                    "attempt_state": "valid",
                    "seal_sha256": _file_sha(root / "seal.json"),
                    "evidence_integrity": "PASS",
                    "artifact_count": 21,
                    "reasons": [],
                },
            )

            result = module.decide_attempt(
                root / "seal.json",
                root / "forged-audit.json",
                root / "decision.json",
                root / "evidence-manifest.json",
            )

            self.assertEqual(result["decision"], "retain")
            self.assertEqual(result["evidence_integrity"], "FAIL")
            self.assertTrue(
                any("provided_audit_mismatch" in reason for reason in result["reasons"])
            )

    def test_audit_rejects_seal_fields_forged_away_from_attempt_manifest(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = _build_attempt(root, state="partial")
            module.seal_attempt(attempt, root / "seal.json")
            (root / "seal.json").chmod(0o600)
            seal = json.loads((root / "seal.json").read_text())
            seal["attempt_state"] = "valid"
            _write_json(root / "seal.json", seal)

            audit = module.audit_seal(root / "seal.json", root / "audit.json")

            self.assertEqual(audit["evidence_integrity"], "FAIL")
            self.assertIn("attempt_manifest_binding_mismatch", audit["reasons"])

    def test_audit_rejects_unknown_nested_seal_fields(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = _build_attempt(root, state="partial")
            module.seal_attempt(attempt, root / "seal.json")
            (root / "seal.json").chmod(0o600)
            seal = json.loads((root / "seal.json").read_text())
            seal["artifacts"][0]["unknown"] = True
            _write_json(root / "seal.json", seal)

            audit = module.audit_seal(root / "seal.json", root / "audit.json")

            self.assertEqual(audit["evidence_integrity"], "FAIL")
            self.assertTrue(
                any("artifact_record_invalid" in reason for reason in audit["reasons"])
            )

    def test_terminal_states_are_closed_and_only_valid_can_promote(self) -> None:
        module = _load_module()
        terminal_states = (
            "valid",
            "invalid_contaminated",
            "invalid_identity",
            "partial",
            "superseded",
        )
        for state in terminal_states:
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                attempt = _build_attempt(root, state=state)
                module.seal_attempt(attempt, root / "seal.json")
                module.audit_seal(root / "seal.json", root / "audit.json")
                result = module.decide_attempt(
                    root / "seal.json",
                    root / "audit.json",
                    root / "decision.json",
                    root / "evidence-manifest.json",
                )
                expected = "promote" if state == "valid" else "retain"
                self.assertEqual(result["decision"], expected)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = _build_attempt(root, state="running")
            with self.assertRaises(ValueError):
                module.seal_attempt(attempt, root / "seal.json")

    def test_valid_attempt_requires_all_evidence_and_no_role_retry_or_exclusion(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = _build_attempt(root)
            manifest = json.loads(attempt.read_text())
            manifest["artifacts"] = [
                item for item in manifest["artifacts"] if item["kind"] != "engine"
            ]
            _write_json(attempt, manifest)
            with self.assertRaises(ValueError):
                module.seal_attempt(attempt, root / "seal.json")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = _build_attempt(root)
            rows = [
                json.loads(line)
                for line in (root / "raw-rows.jsonl").read_text().splitlines()
            ]
            rows[0]["attempts"]["candidate"] = 2
            (root / "raw-rows.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                module.seal_attempt(attempt, root / "seal.json")

    def test_valid_attempt_rejects_cross_artifact_identity_mismatch(self) -> None:
        module = _load_module()
        mutators = (
            lambda root: (
                json.loads((root / "identities.json").read_text()),
                "identities.json",
                lambda item: (
                    item["source"].update(sha256=_sha("0")),
                    item["binary"].update(source_sha256=_sha("0")),
                ),
            ),
            lambda root: (
                json.loads((root / "execution-path.json").read_text()),
                "execution-path.json",
                lambda item: item["timed_binary"].update(sha256=_sha("0")),
            ),
            lambda root: (
                json.loads((root / "profiler.json").read_text()),
                "profiler.json",
                lambda item: (
                    item.update(target_binary_sha256=_sha("0")),
                    [
                        report.update(target_binary_sha256=_sha("0"))
                        for report in item["reports"]
                    ],
                ),
            ),
            lambda root: (
                json.loads((root / "profiler.json").read_text()),
                "profiler.json",
                lambda item: item["reports"][0].update(report_sha256=_sha("0")),
            ),
            lambda root: (
                json.loads((root / "schedule.json").read_text()),
                "schedule.json",
                lambda item: item.reverse(),
            ),
            lambda root: (
                json.loads((root / "performance.json").read_text()),
                "performance.json",
                lambda item: item.update(analysis_sha256=_sha("0")),
            ),
        )
        for prepare in mutators:
            with self.subTest(prepare=prepare), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                attempt = _build_attempt(root)
                payload, name, mutate = prepare(root)
                mutate(payload)
                _write_json(root / name, payload)

                with self.assertRaises(ValueError):
                    module.seal_attempt(attempt, root / "seal.json")

    def test_performance_verdict_cannot_claim_evidence_integrity(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = _build_attempt(root)
            verdict = json.loads((root / "performance.json").read_text())
            verdict["evidence_integrity"] = "PASS"
            _write_json(root / "performance.json", verdict)
            with self.assertRaises(ValueError):
                module.seal_attempt(attempt, root / "seal.json")

    def test_normal_closure_requires_one_evidence_directory(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "attempt"
            other = parent / "other"
            root.mkdir()
            other.mkdir()
            attempt = _build_attempt(root)

            with self.assertRaises(ValueError):
                module.seal_attempt(attempt, other / "seal.json")
            self.assertFalse((other / "seal.json").exists())

            module.seal_attempt(attempt, root / "seal.json")
            module.audit_seal(root / "seal.json", root / "audit.json")
            with self.assertRaises(ValueError):
                module.decide_attempt(
                    root / "seal.json",
                    root / "audit.json",
                    other / "decision.json",
                    other / "evidence-manifest.json",
                )

    def test_seal_rejects_symlink_and_traversal_artifact_paths(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "attempt"
            root.mkdir()
            attempt = _build_attempt(root)
            external = parent / "external.py"
            external.write_text("external\n", encoding="utf-8")
            (root / "runner.py").unlink()
            (root / "runner.py").symlink_to(external)
            with self.assertRaises(ValueError):
                module.seal_attempt(attempt, root / "seal.json")

        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "attempt"
            root.mkdir()
            attempt = _build_attempt(root)
            (parent / "outside.py").write_text("outside\n", encoding="utf-8")
            manifest = json.loads(attempt.read_text())
            next(item for item in manifest["artifacts"] if item["kind"] == "runner")[
                "path"
            ] = "../outside.py"
            _write_json(attempt, manifest)
            with self.assertRaises(ValueError):
                module.seal_attempt(attempt, root / "seal.json")


class ImportedServingAuditTests(unittest.TestCase):
    def test_sealed_import_is_rehashed_read_only_and_remains_non_promotional(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            imported = parent / "imported"
            output = parent / "audit-output"
            imported.mkdir()
            attempt = _build_attempt(imported)
            module.seal_attempt(attempt, imported / "seal.json")
            before = {
                path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in imported.iterdir()
                if path.is_file()
            }

            result = module.audit_imported_run(imported, output)

            self.assertEqual(result["status"], "sealed_import")
            self.assertEqual(result["evidence_integrity"], "PASS")
            self.assertEqual(result["promotional"], False)
            after = {
                path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in imported.iterdir()
                if path.is_file()
            }
            self.assertEqual(after, before)

    def test_legacy_v2_4_import_is_read_only_and_non_promotional(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            imported = parent / "imported"
            output = parent / "audit-output"
            imported.mkdir()
            manifest = imported / "manifest.json"
            _write_json(
                manifest,
                {"schema_version": "cuda-optimizer/v2", "input_hash": _sha("a")},
            )
            before = manifest.read_bytes()
            before_stat = manifest.stat()

            result = module.audit_imported_run(imported, output)

            self.assertEqual(result["status"], "legacy_unsealed")
            self.assertEqual(result["promotional"], False)
            self.assertEqual(result["evidence_integrity"], "UNKNOWN")
            self.assertEqual(manifest.read_bytes(), before)
            self.assertEqual(manifest.stat().st_mtime_ns, before_stat.st_mtime_ns)
            self.assertTrue((output / "import-audit.json").is_file())

    def test_import_output_must_be_outside_source_tree(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            imported = Path(tmp) / "imported"
            imported.mkdir()
            _write_json(imported / "manifest.json", {"schema_version": "cuda-optimizer/v2"})
            with self.assertRaises(ValueError):
                module.audit_imported_run(imported, imported / "audit")


if __name__ == "__main__":
    unittest.main()
