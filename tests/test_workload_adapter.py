from __future__ import annotations

import copy
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import textwrap
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"
TEMPLATE_DIR = ROOT / "skills" / "cuda-kernel-optimizer" / "templates"
WORKLOAD_ADAPTER_PATH = SCRIPT_DIR / "workload_adapter.py"
PREFLIGHT_PATH = SCRIPT_DIR / "preflight.py"


def _load_workload_adapter():
    module_name = "cuda_optimizer_workload_adapter_test"
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


def _objective(
    *,
    name: str = "p50_latency_ms",
    direction: str = "lower",
    min_effect_pct: float = 1.0,
) -> dict:
    return {
        "primary_metric": {"name": name, "direction": direction},
        "min_effect_pct": min_effect_pct,
        "constraints": [
            {"name": "p99_latency_ms", "max_regression_pct": 0.5}
        ],
    }


def _write_python_adapter(path: Path, *, objective: dict | None = None) -> None:
    objective = _objective() if objective is None else objective
    path.write_text(
        textwrap.dedent(
            f"""
            EVENTS = []

            def prepare(candidate):
                EVENTS.append(("prepare", candidate))
                return {{"prepared": candidate}}

            def validate(candidate):
                EVENTS.append(("validate", candidate))
                return True

            def benchmark(candidate):
                EVENTS.append(("benchmark", candidate))
                return {{"latency_ms": 1.25}}

            def metrics():
                EVENTS.append(("metrics", None))
                return {objective!r}

            def cleanup():
                EVENTS.append(("cleanup", None))
            """
        ),
        encoding="utf-8",
    )


class NormalizeWorkloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workloads = _load_workload_adapter()

    def test_zero_forms_is_kernel_only_and_forms_are_mutually_exclusive(self) -> None:
        self.assertIsNone(self.workloads.normalize_workload())
        combinations = (
            {"workload": "a.py", "workload_cmd": "run"},
            {"workload": "a.py", "workload_manifest": "m.json"},
            {"workload_cmd": "run", "workload_manifest": "m.json"},
            {
                "workload": "a.py",
                "workload_cmd": "run",
                "workload_manifest": "m.json",
            },
        )
        for kwargs in combinations:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValueError, "exactly one workload form"
            ):
                self.workloads.normalize_workload(**kwargs)

    def test_objective_without_workload_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "objective.*without.*workload"):
            self.workloads.normalize_workload(objective="objective.json")

    def test_python_adapter_requires_all_calls_and_lists_missing_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workload.py"
            path.write_text(
                "def prepare(candidate): pass\n"
                "def benchmark(candidate): pass\n"
                "cleanup = 42\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, r"validate, metrics, cleanup"
            ) as raised:
                self.workloads.load_python_adapter(path)

            message = str(raised.exception)
            for name in ("validate", "metrics", "cleanup"):
                self.assertEqual(message.count(name), 1)

    def test_python_form_normalizes_objective_hashes_source_and_freezes_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workload.py"
            _write_python_adapter(path)

            first = self.workloads.normalize_workload(workload=path)
            second = self.workloads.normalize_workload(workload=str(path))

            self.assertEqual(first.kind, "python")
            self.assertEqual(first.source, str(path.resolve()))
            self.assertEqual(first.objective, _objective())
            self.assertEqual(first.cases, ())
            self.assertEqual(first.source_hash, second.source_hash)
            self.assertEqual(len(first.source_hash), 64)
            with self.assertRaises(FrozenInstanceError):
                first.kind = "command"
            with self.assertRaises(TypeError):
                first.objective["min_effect_pct"] = 99
            with self.assertRaises(TypeError):
                first.objective["constraints"].append(
                    {"name": "new", "max_regression_pct": 1}
                )

            path.write_text(path.read_text("utf-8") + "\n# changed\n", "utf-8")
            changed = self.workloads.normalize_workload(workload=path)
            self.assertNotEqual(first.source_hash, changed.source_hash)

    def test_python_form_rejects_external_objective_as_conflicting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "workload.py"
            objective = root / "objective.json"
            _write_python_adapter(adapter)
            objective.write_text(json.dumps(_objective()), "utf-8")

            with self.assertRaisesRegex(ValueError, "conflicting objective"):
                self.workloads.normalize_workload(
                    workload=adapter, objective=objective
                )

    def test_command_requires_external_objective_and_uses_shlex(self) -> None:
        with self.assertRaisesRegex(ValueError, "command.*--objective"):
            self.workloads.normalize_workload(
                workload_cmd='./runner --label "two words"'
            )

        with tempfile.TemporaryDirectory() as tmp:
            objective_path = Path(tmp) / "objective.json"
            objective_path.write_text(json.dumps(_objective()), "utf-8")
            spec = self.workloads.normalize_workload(
                workload_cmd='./runner --label "two words"',
                objective=objective_path,
            )

        self.assertEqual(spec.kind, "command")
        self.assertEqual(spec.source, ["./runner", "--label", "two words"])
        self.assertEqual(spec.objective, _objective())

    def test_command_sequence_rejects_non_string_and_empty_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            objective_path = Path(tmp) / "objective.json"
            objective_path.write_text(json.dumps(_objective()), "utf-8")
            for command in ([], ["run", ""], ["run", 3], "   "):
                with self.subTest(command=command), self.assertRaisesRegex(
                    ValueError, "command"
                ):
                    self.workloads.normalize_workload(
                        workload_cmd=command, objective=objective_path
                    )


class ObjectiveValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workloads = _load_workload_adapter()

    def test_valid_objective_is_deep_copied_without_mutating_input(self) -> None:
        source = _objective()
        original = copy.deepcopy(source)
        result = self.workloads.validate_objective(source)

        self.assertEqual(result, original)
        self.assertEqual(source, original)
        self.assertIsNot(result, source)
        self.assertIsNot(result["primary_metric"], source["primary_metric"])
        self.assertIsNot(result["constraints"], source["constraints"])

        source["primary_metric"]["name"] = "changed"
        source["constraints"][0]["name"] = "changed_constraint"
        self.assertEqual(result, original)

    def test_objective_rejects_missing_unknown_and_bad_primary_fields(self) -> None:
        cases = []
        for missing in ("primary_metric", "min_effect_pct", "constraints"):
            value = _objective()
            del value[missing]
            cases.append((value, missing))
        value = _objective()
        value["weight"] = 0.5
        cases.append((value, "unknown"))
        for primary in (
            {},
            {"name": "", "direction": "lower"},
            {"name": "latency", "direction": "minimize"},
            {"name": "latency", "direction": "lower", "weight": 1},
        ):
            value = _objective()
            value["primary_metric"] = primary
            cases.append((value, "primary_metric"))

        for value, message in cases:
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, message
            ):
                self.workloads.validate_objective(value)

    def test_objective_rejects_invalid_numeric_values(self) -> None:
        invalid = (True, False, "1", -0.1, math.nan, math.inf, -math.inf)
        for field in ("min_effect_pct", "max_regression_pct"):
            for bad in invalid:
                value = _objective()
                if field == "min_effect_pct":
                    value[field] = bad
                else:
                    value["constraints"][0][field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaisesRegex(
                    ValueError, field
                ):
                    self.workloads.validate_objective(value)

    def test_constraints_must_be_objects_with_unique_names_and_known_fields(self) -> None:
        cases = (
            ({**_objective(), "constraints": {}}, "constraints"),
            ({**_objective(), "constraints": ["latency"]}, "constraints"),
            (
                {
                    **_objective(),
                    "constraints": [
                        {"name": "same", "max_regression_pct": 1},
                        {"name": "same", "max_regression_pct": 2},
                    ],
                },
                "duplicate",
            ),
            (
                {
                    **_objective(),
                    "constraints": [
                        {
                            "name": "p99",
                            "max_regression_pct": 1,
                            "weight": 3,
                        }
                    ],
                },
                "unknown",
            ),
        )
        for value, message in cases:
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, message
            ):
                self.workloads.validate_objective(value)

    def test_external_objective_must_exist_and_be_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, payload, message in (
                ("missing.json", None, "objective"),
                ("bad.json", "not json", "JSON"),
                ("array.json", "[]", "object"),
            ):
                path = root / name
                if payload is not None:
                    path.write_text(payload, "utf-8")
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, message
                ):
                    self.workloads.load_objective(path)


class PythonLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workloads = _load_workload_adapter()

    def _adapter(self, **overrides):
        events = []

        class Adapter:
            def prepare(self, candidate):
                events.append(("prepare", candidate))

            def validate(self, candidate):
                events.append(("validate", candidate))
                return True

            def benchmark(self, candidate):
                events.append(("benchmark", candidate))
                return {"latency_ms": 1.2}

            def metrics(self):
                events.append(("metrics", None))
                return _objective()

            def cleanup(self):
                events.append(("cleanup", None))

        adapter = Adapter()
        for name, value in overrides.items():
            setattr(adapter, name, value)
        return adapter, events

    def test_run_once_preserves_order_returns_raw_observation_and_inputs(self) -> None:
        adapter, events = self._adapter()
        candidate = {"path": "candidate.py"}
        case = {"batch": 4, "shape": [2, 8]}
        originals = copy.deepcopy((candidate, case))

        result = self.workloads.run_once(
            adapter, candidate=candidate, role="candidate", case=case
        )

        self.assertEqual(
            [name for name, _ in events],
            ["prepare", "validate", "benchmark", "metrics", "cleanup"],
        )
        self.assertEqual(result["benchmark"], {"latency_ms": 1.2})
        self.assertEqual(result["objective"], _objective())
        self.assertEqual(result["role"], "candidate")
        self.assertEqual(result["case"], case)
        self.assertEqual((candidate, case), originals)

    def test_validation_failure_still_cleans_up_exactly_once(self) -> None:
        adapter, events = self._adapter(validate=lambda candidate: False)
        original_cleanup = adapter.cleanup
        cleanup_count = 0

        def cleanup():
            nonlocal cleanup_count
            cleanup_count += 1
            original_cleanup()

        adapter.cleanup = cleanup
        with self.assertRaisesRegex(ValueError, "validation failed"):
            self.workloads.run_once(
                adapter, candidate="candidate.py", role="candidate", case={}
            )
        self.assertEqual(cleanup_count, 1)
        self.assertNotIn("benchmark", [name for name, _ in events])

    def test_benchmark_errors_and_base_exceptions_still_cleanup(self) -> None:
        for error in (
            RuntimeError("benchmark failed"),
            TimeoutError("timed out"),
            KeyboardInterrupt(),
            SystemExit(7),
        ):
            cleanup_count = 0

            def fail(candidate, error=error):
                raise error

            def cleanup():
                nonlocal cleanup_count
                cleanup_count += 1

            adapter, _ = self._adapter(benchmark=fail, cleanup=cleanup)
            with self.subTest(error=type(error).__name__), self.assertRaises(
                type(error)
            ):
                self.workloads.run_once(
                    adapter, candidate="candidate.py", role="candidate", case={}
                )
            self.assertEqual(cleanup_count, 1)

    def test_cleanup_error_propagates_or_does_not_mask_primary_error(self) -> None:
        def cleanup_error():
            raise RuntimeError("cleanup failed")

        adapter, _ = self._adapter(cleanup=cleanup_error)
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            self.workloads.run_once(
                adapter, candidate="candidate.py", role="candidate", case={}
            )

        def benchmark_error(candidate):
            raise ValueError("primary benchmark error")

        adapter, _ = self._adapter(
            benchmark=benchmark_error, cleanup=cleanup_error
        )
        with self.assertRaisesRegex(ValueError, "primary benchmark error") as raised:
            self.workloads.run_once(
                adapter, candidate="candidate.py", role="candidate", case={}
            )
        self.assertTrue(
            any("cleanup failed" in note for note in getattr(raised.exception, "__notes__", []))
        )


class CommandWorkloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workloads = _load_workload_adapter()

    def _spec(self, root: Path):
        objective_path = root / "objective.json"
        objective_path.write_text(json.dumps(_objective()), "utf-8")
        return self.workloads.normalize_workload(
            workload_cmd='./runner --label "two words"', objective=objective_path
        )

    def test_command_uses_argv_shell_false_environment_and_only_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(Path(tmp))
            case = {"batch": 8}
            original_case = copy.deepcopy(case)

            def completed(argv, **kwargs):
                self.assertEqual(argv, ["./runner", "--label", "two words"])
                self.assertIs(kwargs["shell"], False)
                self.assertTrue(kwargs["capture_output"])
                self.assertTrue(kwargs["text"])
                env = kwargs["env"]
                self.assertEqual(env["CUDA_OPTIMIZER_CANDIDATE"], "/tmp/candidate.py")
                self.assertEqual(env["CUDA_OPTIMIZER_ROLE"], "baseline")
                self.assertEqual(json.loads(env["CUDA_OPTIMIZER_CASE"]), case)
                output = Path(env["CUDA_OPTIMIZER_OUTPUT"])
                self.assertFalse(output.exists())
                output.write_text('{"latency_ms": 3.5}', "utf-8")
                return subprocess.CompletedProcess(
                    argv, 0, stdout='{"forged": true}', stderr="diagnostic"
                )

            with mock.patch.object(
                self.workloads.subprocess, "run", side_effect=completed
            ) as run:
                result = self.workloads.run_command_once(
                    spec,
                    candidate="/tmp/candidate.py",
                    role="baseline",
                    case=case,
                    timeout=12.5,
                )

            self.assertEqual(result, {"latency_ms": 3.5})
            self.assertEqual(run.call_args.kwargs["timeout"], 12.5)
            self.assertEqual(case, original_case)

    def test_command_failures_have_clear_bounded_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(Path(tmp))
            cases = (
                (FileNotFoundError("missing runner"), "not found"),
                (subprocess.TimeoutExpired(spec.source, 3), "timed out"),
                (
                    subprocess.CompletedProcess(
                        spec.source, 9, stdout="x" * 20000, stderr="failure"
                    ),
                    "exit 9",
                ),
            )
            for side_effect, message in cases:
                with self.subTest(message=message), mock.patch.object(
                    self.workloads.subprocess,
                    "run",
                    side_effect=side_effect
                    if isinstance(side_effect, BaseException)
                    else None,
                    return_value=side_effect
                    if isinstance(side_effect, subprocess.CompletedProcess)
                    else mock.DEFAULT,
                ), self.assertRaisesRegex(RuntimeError, message) as raised:
                    self.workloads.run_command_once(
                        spec,
                        candidate="candidate.py",
                        role="candidate",
                        case={},
                    )
                self.assertLess(len(str(raised.exception)), 9000)

    def test_command_rejects_missing_bad_and_non_object_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(Path(tmp))

            def result_with(payload):
                def run(argv, **kwargs):
                    if payload is not None:
                        Path(kwargs["env"]["CUDA_OPTIMIZER_OUTPUT"]).write_text(
                            payload, "utf-8"
                        )
                    return subprocess.CompletedProcess(argv, 0, "ignored", "")

                return run

            for payload, message in (
                (None, "output"),
                ("not json", "JSON"),
                ("[]", "object"),
                ('{"a": 1}{"b": 2}', "JSON"),
            ):
                with self.subTest(payload=payload), mock.patch.object(
                    self.workloads.subprocess,
                    "run",
                    side_effect=result_with(payload),
                ), self.assertRaisesRegex(RuntimeError, message):
                    self.workloads.run_command_once(
                        spec,
                        candidate="candidate.py",
                        role="candidate",
                        case={},
                    )

    def test_timeout_must_be_positive_finite_and_role_case_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(Path(tmp))
            for timeout in (0, -1, True, "3", math.nan, math.inf, -math.inf):
                with self.subTest(timeout=timeout), self.assertRaisesRegex(
                    ValueError, "timeout"
                ):
                    self.workloads.run_command_once(
                        spec,
                        candidate="candidate.py",
                        role="candidate",
                        case={},
                        timeout=timeout,
                    )
            with self.assertRaisesRegex(ValueError, "case"):
                self.workloads.run_command_once(
                    spec,
                    candidate="candidate.py",
                    role="candidate",
                    case=[],
                )


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workloads = _load_workload_adapter()

    def test_manifest_resolves_python_source_cases_and_hashes_all_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter.py"
            _write_python_adapter(adapter)
            cases = [{"batch": 1}, {"batch": 8, "shape": [2, 4]}]
            manifest = root / "workload.json"
            manifest.write_text(
                json.dumps(
                    {
                        "kind": "python",
                        "source": "adapter.py",
                        "objective": _objective(),
                        "cases": cases,
                    }
                ),
                "utf-8",
            )
            original_adapter_source = adapter.read_text("utf-8")
            original_cases = copy.deepcopy(cases)

            first = self.workloads.normalize_workload(
                workload_manifest=manifest
            )
            second = self.workloads.normalize_workload(
                workload_manifest=manifest
            )

            self.assertEqual(first.kind, "manifest")
            self.assertEqual(first.source, str(adapter.resolve()))
            self.assertEqual(first.objective, _objective())
            self.assertEqual(first.cases, tuple(cases))
            self.assertEqual(first.source_hash, second.source_hash)
            self.assertEqual(cases, original_cases)
            with self.assertRaises(TypeError):
                first.cases[0]["batch"] = 99

            adapter.write_text(original_adapter_source + "\n# edit\n", "utf-8")
            changed_source = self.workloads.normalize_workload(
                workload_manifest=manifest
            )
            self.assertNotEqual(first.source_hash, changed_source.source_hash)

            adapter.write_text(original_adapter_source, "utf-8")
            payload = json.loads(manifest.read_text("utf-8"))
            payload["cases"][0]["batch"] = 2
            manifest.write_text(json.dumps(payload), "utf-8")
            changed_case = self.workloads.normalize_workload(
                workload_manifest=manifest
            )
            self.assertNotEqual(first.source_hash, changed_case.source_hash)

    def test_manifest_hash_is_stable_across_json_key_order_and_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter.py"
            _write_python_adapter(adapter)
            manifest = root / "workload.json"
            payload = {
                "kind": "python",
                "source": "adapter.py",
                "objective": _objective(),
                "cases": [{"batch": 4, "shape": [2, 8]}],
            }
            manifest.write_text(json.dumps(payload), "utf-8")
            first = self.workloads.normalize_workload(
                workload_manifest=manifest
            )

            reordered = {
                "cases": payload["cases"],
                "objective": payload["objective"],
                "source": payload["source"],
                "kind": payload["kind"],
            }
            manifest.write_text(json.dumps(reordered, indent=4), "utf-8")
            second = self.workloads.normalize_workload(
                workload_manifest=manifest
            )

            self.assertEqual(first.source_hash, second.source_hash)

    def test_manifest_embedded_and_external_objective_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter.py"
            _write_python_adapter(adapter)
            manifest = root / "workload.json"
            manifest.write_text(
                json.dumps(
                    {
                        "kind": "python",
                        "source": "adapter.py",
                        "objective": _objective(),
                        "cases": [],
                    }
                ),
                "utf-8",
            )
            external = root / "objective.json"
            external.write_text(json.dumps(_objective()), "utf-8")

            with self.assertRaisesRegex(ValueError, "conflicting objective"):
                self.workloads.normalize_workload(
                    workload_manifest=manifest, objective=external
                )

    def test_manifest_requires_objective_cases_and_strict_known_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter.py"
            _write_python_adapter(adapter)
            base = {
                "kind": "python",
                "source": "adapter.py",
                "cases": [],
            }
            variants = (
                (base, "objective"),
                ({**base, "objective": _objective(), "cases": {}}, "cases"),
                (
                    {**base, "objective": _objective(), "cases": [1]},
                    "cases",
                ),
                ({**base, "objective": _objective(), "extra": True}, "unknown"),
                ({**base, "objective": _objective(), "kind": "binary"}, "kind"),
            )
            for index, (payload, message) in enumerate(variants):
                manifest = root / f"manifest-{index}.json"
                manifest.write_text(json.dumps(payload), "utf-8")
                with self.subTest(index=index), self.assertRaisesRegex(
                    ValueError, message
                ):
                    self.workloads.normalize_workload(
                        workload_manifest=manifest
                    )

    def test_command_manifest_resolves_relative_referenced_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = root / "run benchmark.sh"
            runner.write_text("#!/bin/sh\n", "utf-8")
            manifest = root / "workload.json"
            manifest.write_text(
                json.dumps(
                    {
                        "kind": "command",
                        "source": ["./run benchmark.sh", "--quick"],
                        "objective": _objective(),
                        "cases": [{"batch": 1}],
                    }
                ),
                "utf-8",
            )

            spec = self.workloads.normalize_workload(
                workload_manifest=manifest
            )

            self.assertEqual(spec.kind, "manifest")
            self.assertEqual(spec.source[0], str(runner.resolve()))
            self.assertEqual(spec.source[1], "--quick")


class TemplateAndPreflightTests(unittest.TestCase):
    def _valid_kernel_inputs(self, root: Path) -> tuple[Path, Path]:
        baseline = root / "baseline.cu"
        baseline.write_text(
            'extern "C" void solve(float* x, int n) { }\n', "utf-8"
        )
        ref = root / "ref.py"
        ref.write_text("def reference(**kwargs): return None\n", "utf-8")
        return baseline, ref

    def _preflight(self, *args: object, no_site: bool = False):
        command = [sys.executable]
        if no_site:
            command.append("-S")
        command.extend([str(PREFLIGHT_PATH), *(str(arg) for arg in args)])
        return subprocess.run(command, text=True, capture_output=True, check=False)

    def test_templates_are_strict_and_do_not_claim_a_real_workload(self) -> None:
        schema = json.loads((TEMPLATE_DIR / "objective.schema.json").read_text("utf-8"))
        self.assertEqual(schema["type"], "object")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            set(schema["required"]),
            {"primary_metric", "min_effect_pct", "constraints"},
        )
        template = (TEMPLATE_DIR / "workload.py").read_text("utf-8")
        for name in ("prepare", "validate", "benchmark", "metrics", "cleanup"):
            self.assertIn(f"def {name}(", template)
        self.assertIn("TODO", template)
        self.assertIn("NotImplementedError", template)

    def test_preflight_no_workload_is_kernel_only_and_preserves_old_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline, ref = self._valid_kernel_inputs(root)
            completed = self._preflight(
                "--baseline",
                baseline,
                "--ref",
                ref,
                "--dims",
                '{"n": 4}',
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertTrue(report["ok"])
            self.assertEqual(report["mode"], "kernel-only")
            self.assertIsNone(report["workload"])
            for old_field in ("baseline", "ref", "warnings", "errors"):
                self.assertIn(old_field, report)

    def test_preflight_python_workload_is_full_without_running_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline, ref = self._valid_kernel_inputs(root)
            adapter = root / "workload.py"
            _write_python_adapter(adapter)
            adapter.write_text(
                adapter.read_text("utf-8").replace(
                    'return {"latency_ms": 1.25}',
                    'raise RuntimeError("benchmark must not run in preflight")',
                ),
                "utf-8",
            )
            completed = self._preflight(
                "--baseline",
                baseline,
                "--ref",
                ref,
                "--dims",
                '{"n": 4}',
                "--workload",
                adapter,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertEqual(report["mode"], "full")
            self.assertEqual(report["workload"]["kind"], "python")
            self.assertEqual(report["workload"]["cases_count"], 0)
            self.assertEqual(len(report["workload"]["source_hash"]), 64)
            self.assertEqual(report["workload"]["objective"], _objective())

    def test_preflight_conflicts_fail_nonzero_with_precise_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline, ref = self._valid_kernel_inputs(root)
            adapter = root / "workload.py"
            _write_python_adapter(adapter)
            manifest = root / "manifest.json"
            manifest.write_text("{}", "utf-8")
            completed = self._preflight(
                "--baseline",
                baseline,
                "--ref",
                ref,
                "--dims",
                '{"n": 4}',
                "--workload",
                adapter,
                "--workload-manifest",
                manifest,
            )

            self.assertNotEqual(completed.returncode, 0)
            report = json.loads(completed.stdout)
            self.assertFalse(report["ok"])
            self.assertIn("exactly one workload form", " ".join(report["errors"]))

    def test_preflight_command_objective_and_help_without_site_packages(self) -> None:
        help_result = self._preflight("--help", no_site=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        for option in (
            "--workload",
            "--workload-cmd",
            "--workload-manifest",
            "--objective",
        ):
            self.assertIn(option, help_result.stdout)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline, ref = self._valid_kernel_inputs(root)
            without_objective = self._preflight(
                "--baseline",
                baseline,
                "--ref",
                ref,
                "--dims",
                '{"n": 4}',
                "--workload-cmd",
                "./run.sh",
            )
            self.assertNotEqual(without_objective.returncode, 0)
            report = json.loads(without_objective.stdout)
            self.assertIn("--objective", " ".join(report["errors"]))


if __name__ == "__main__":
    unittest.main()
