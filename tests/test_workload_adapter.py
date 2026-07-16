from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import types
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


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n" + textwrap.dedent(body),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


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
            root = Path(tmp)
            runner = _write_executable(root / "runner", "pass\n")
            objective_path = root / "objective.json"
            objective_path.write_text(json.dumps(_objective()), "utf-8")
            spec = self.workloads.normalize_workload(
                workload_cmd=f'{shlex.quote(str(runner))} --label "two words"',
                objective=objective_path,
            )

        self.assertEqual(spec.kind, "command")
        self.assertEqual(spec.source, [str(runner.resolve()), "--label", "two words"])
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
            {"name": "   ", "direction": "lower"},
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
                        {"name": "   ", "max_regression_pct": 1}
                    ],
                },
                "constraints",
            ),
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
            ["prepare", "validate", "benchmark", "cleanup", "metrics"],
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

    def test_validation_result_schema_errors_still_cleanup_exactly_once(self) -> None:
        for invalid in (None, "passed", {}, {"valid": "yes"}):
            cleanup_count = 0

            def validate(candidate, invalid=invalid):
                return invalid

            def cleanup():
                nonlocal cleanup_count
                cleanup_count += 1

            adapter, _ = self._adapter(validate=validate, cleanup=cleanup)
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "validation"
            ):
                self.workloads.run_once(
                    adapter,
                    candidate="candidate.py",
                    role="candidate",
                    case={},
                )
            self.assertEqual(cleanup_count, 1)

    def test_benchmark_must_be_json_object_and_schema_errors_cleanup(self) -> None:
        invalid_results = (None, True, 1, 1.5, "fast", [], [1, 2])
        for invalid in invalid_results:
            cleanup_count = 0

            def benchmark(candidate, invalid=invalid):
                return invalid

            def cleanup():
                nonlocal cleanup_count
                cleanup_count += 1

            adapter, _ = self._adapter(benchmark=benchmark, cleanup=cleanup)
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "benchmark"
            ):
                self.workloads.run_once(
                    adapter,
                    candidate="candidate.py",
                    role="candidate",
                    case={},
                )
            self.assertEqual(cleanup_count, 1)

    def test_observation_objects_are_json_safe_deep_copies(self) -> None:
        for benchmark, message in (
            ({1: "non-string key"}, "string keys"),
            ({"value": object()}, "JSON"),
            ({"value": math.nan}, "finite"),
        ):
            cleanup_count = 0

            def return_benchmark(candidate, benchmark=benchmark):
                return benchmark

            def cleanup():
                nonlocal cleanup_count
                cleanup_count += 1

            adapter, _ = self._adapter(
                benchmark=return_benchmark,
                cleanup=cleanup,
            )
            with self.subTest(benchmark=benchmark), self.assertRaisesRegex(
                ValueError, message
            ):
                self.workloads.run_once(
                    adapter,
                    candidate="candidate.py",
                    role="candidate",
                    case={},
                )
            self.assertEqual(cleanup_count, 1)

        validation = {"valid": True, "checks": ["shape"]}
        benchmark = {"latency": {"samples": [1.0, 1.1]}}
        adapter, _ = self._adapter(
            validate=lambda candidate: validation,
            benchmark=lambda candidate: benchmark,
        )
        result = self.workloads.run_once(
            adapter,
            candidate="candidate.py",
            role="candidate",
            case={},
        )
        self.assertEqual(result["validation"], validation)
        self.assertEqual(result["benchmark"], benchmark)
        self.assertIsNot(result["validation"], validation)
        self.assertIsNot(result["benchmark"], benchmark)
        validation["checks"].append("changed")
        benchmark["latency"]["samples"].append(99)
        self.assertEqual(result["validation"]["checks"], ["shape"])
        self.assertEqual(result["benchmark"]["latency"]["samples"], [1.0, 1.1])

    def test_schema_error_is_primary_when_cleanup_also_fails(self) -> None:
        def cleanup_error():
            raise RuntimeError("cleanup failed")

        adapter, _ = self._adapter(
            validate=lambda candidate: None,
            cleanup=cleanup_error,
        )
        with self.assertRaisesRegex(ValueError, "validation") as raised:
            self.workloads.run_once(
                adapter,
                candidate="candidate.py",
                role="candidate",
                case={},
            )
        self.assertTrue(
            any(
                "cleanup failed" in note
                for note in getattr(raised.exception, "__notes__", [])
            )
        )


class CommandWorkloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workloads = _load_workload_adapter()

    def _spec(self, root: Path, body: str | None = None):
        if body is None:
            body = """
                import json, os
                from pathlib import Path
                Path(os.environ["CUDA_OPTIMIZER_OUTPUT"]).write_text(json.dumps({
                    "validation": {"valid": True},
                    "benchmark": {"latency_ms": 3.5}
                }))
            """
        runner = _write_executable(root / "runner", body)
        objective_path = root / "objective.json"
        objective_path.write_text(json.dumps(_objective()), "utf-8")
        return self.workloads.normalize_workload(
            workload_cmd=[str(runner), "--label", "two words"],
            objective=objective_path,
        )

    def test_command_uses_argv_shell_false_environment_and_only_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(Path(tmp))
            case = {"batch": 8}
            original_case = copy.deepcopy(case)

            real_popen = subprocess.Popen
            with mock.patch.object(
                self.workloads.subprocess, "Popen", wraps=real_popen
            ) as popen:
                result = self.workloads.run_command_once(
                    spec,
                    candidate="/tmp/candidate.py",
                    role="baseline",
                    case=case,
                    timeout=12.5,
                )

            self.assertEqual(
                result,
                {
                    "validation": {"valid": True},
                    "benchmark": {"latency_ms": 3.5},
                },
            )
            self.assertEqual(popen.call_args.args[0], list(spec.source))
            self.assertIs(popen.call_args.kwargs["shell"], False)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            env = popen.call_args.kwargs["env"]
            self.assertEqual(env["CUDA_OPTIMIZER_CANDIDATE"], "/tmp/candidate.py")
            self.assertEqual(env["CUDA_OPTIMIZER_ROLE"], "baseline")
            self.assertEqual(json.loads(env["CUDA_OPTIMIZER_CASE"]), case)
            self.assertEqual(case, original_case)

    def test_command_failures_have_clear_bounded_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = (
                ("import time; time.sleep(30)\n", "timed out", 0.1),
                ('import sys; print("x" * 20000); print("failure", file=sys.stderr); raise SystemExit(9)\n', "exit 9", None),
            )
            for index, (body, message, timeout) in enumerate(cases):
                case_root = root / str(index)
                case_root.mkdir()
                spec = self._spec(case_root, body)
                with self.subTest(message=message), self.assertRaisesRegex(
                    RuntimeError, message
                ) as raised:
                    self.workloads.run_command_once(
                        spec,
                        candidate="candidate.py",
                        role="candidate",
                        case={},
                        timeout=timeout,
                    )
                self.assertLess(len(str(raised.exception)), 9000)

    def test_command_rejects_missing_bad_and_non_object_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, (payload, message) in enumerate((
                (None, "output"),
                ("not json", "JSON"),
                ("[]", "object"),
                ('{"a": 1}{"b": 2}', "JSON"),
                ('{"benchmark": {}}', "validation"),
                ('{"validation": true}', "benchmark"),
                (
                    '{"validation": "passed", "benchmark": {}}',
                    "validation",
                ),
                (
                    '{"validation": {}, "benchmark": {}}',
                    "validation",
                ),
                (
                    '{"validation": true, "benchmark": 1}',
                    "benchmark",
                ),
            )):
                case_root = root / str(index)
                case_root.mkdir()
                if payload is None:
                    body = "pass\n"
                else:
                    body = f'''\nfrom pathlib import Path\nimport os\nPath(os.environ["CUDA_OPTIMIZER_OUTPUT"]).write_text({payload!r})\n'''
                spec = self._spec(case_root, body)
                with self.subTest(payload=payload), self.assertRaisesRegex(
                    RuntimeError, message
                ):
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

            self.assertEqual(first.kind, "python")
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
            runner.chmod(0o755)
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

            self.assertEqual(spec.kind, "command")
            self.assertEqual(spec.source[0], str(runner.resolve()))
            self.assertEqual(spec.source[1], "--quick")

    def test_python_manifest_objective_must_match_adapter_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter.py"
            _write_python_adapter(
                adapter,
                objective=_objective(name="throughput", direction="higher"),
            )
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

            with self.assertRaisesRegex(ValueError, "conflicting objective"):
                self.workloads.normalize_workload(workload_manifest=manifest)

    def test_python_manifest_rejects_invalid_metrics_objective(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter.py"
            _write_python_adapter(adapter)
            adapter.write_text(
                adapter.read_text("utf-8").replace(
                    f"return {_objective()!r}", "return {'invalid': True}"
                ),
                "utf-8",
            )
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

            with self.assertRaisesRegex(ValueError, "objective"):
                self.workloads.normalize_workload(workload_manifest=manifest)


class UnifiedExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workloads = _load_workload_adapter()

    @staticmethod
    def _command_runner(path: Path, payload: dict) -> Path:
        return _write_executable(
            path,
            f'''\nimport os\nfrom pathlib import Path\nPath(os.environ["CUDA_OPTIMIZER_OUTPUT"]).write_text({json.dumps(payload)!r})\n''',
        )
        return subprocess.CompletedProcess(argv, 0, "ignored", "")

    def _assert_unified(self, result: dict, *, case: dict) -> None:
        self.assertEqual(
            set(result),
            {"role", "case", "validation", "benchmark", "objective"},
        )
        self.assertEqual(result["role"], "candidate")
        self.assertEqual(result["case"], case)
        self.assertIn("validation", result)
        self.assertIn("benchmark", result)
        self.assertEqual(result["objective"], _objective())

    def test_direct_python_and_command_share_run_spec_once_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter.py"
            _write_python_adapter(adapter)
            python_spec = self.workloads.normalize_workload(workload=adapter)
            objective_path = root / "objective.json"
            objective_path.write_text(json.dumps(_objective()), "utf-8")
            runner = self._command_runner(
                root / "user-runner",
                {
                    "validation": {"valid": True, "checks": 4},
                    "benchmark": {"latency_ms": 2.5},
                    "diagnostics": {"source": "user-workload"},
                },
            )
            command_spec = self.workloads.normalize_workload(
                workload_cmd=[str(runner), "--once"],
                objective=objective_path,
            )
            case = {"batch": 4}

            python_result = self.workloads.run_spec_once(
                python_spec,
                candidate="candidate.py",
                role="candidate",
                case=case,
            )
            command_result = self.workloads.run_spec_once(
                command_spec,
                candidate="candidate.py",
                role="candidate",
                case=case,
                timeout=10,
            )

            self._assert_unified(python_result, case=case)
            self._assert_unified(command_result, case=case)
            self.assertEqual(
                set(python_result),
                {"role", "case", "validation", "benchmark", "objective"},
            )
            self.assertEqual(
                set(command_result),
                {
                    "role",
                    "case",
                    "validation",
                    "benchmark",
                    "objective",
                },
            )

    def test_manifest_python_and_command_dispatch_through_same_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter.py"
            _write_python_adapter(adapter)
            runner = self._command_runner(
                root / "runner.sh",
                {
                    "validation": {"valid": True, "checks": 4},
                    "benchmark": {"latency_ms": 2.5},
                    "diagnostics": {"source": "user-workload"},
                },
            )
            cases = [{"batch": 8}]
            python_manifest = root / "python.json"
            python_manifest.write_text(
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
            command_manifest = root / "command.json"
            command_manifest.write_text(
                json.dumps(
                    {
                        "kind": "command",
                        "source": ["./runner.sh"],
                        "objective": _objective(),
                        "cases": cases,
                    }
                ),
                "utf-8",
            )
            python_spec = self.workloads.normalize_workload(
                workload_manifest=python_manifest
            )
            command_spec = self.workloads.normalize_workload(
                workload_manifest=command_manifest
            )

            self.assertEqual(python_spec.kind, "python")
            self.assertEqual(command_spec.kind, "command")
            python_result = self.workloads.run_spec_once(
                python_spec,
                candidate="candidate.py",
                role="candidate",
                case=cases[0],
            )
            command_result = self.workloads.run_spec_once(
                command_spec,
                candidate="candidate.py",
                role="candidate",
                case=cases[0],
            )

            self._assert_unified(python_result, case=cases[0])
            self._assert_unified(command_result, case=cases[0])

    def test_run_spec_rejects_source_or_metrics_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter.py"
            _write_python_adapter(adapter)
            source_spec = self.workloads.normalize_workload(workload=adapter)
            adapter.write_text(adapter.read_text("utf-8") + "\n# changed\n", "utf-8")
            with self.assertRaisesRegex(ValueError, "source_hash"):
                self.workloads.run_spec_once(
                    source_spec,
                    candidate="candidate.py",
                    role="candidate",
                )

            dynamic_objective = root / "dynamic-objective.json"
            dynamic_objective.write_text(json.dumps(_objective()), "utf-8")
            dynamic_adapter = root / "dynamic.py"
            dynamic_adapter.write_text(
                textwrap.dedent(
                    """
                    import json
                    from pathlib import Path

                    def prepare(candidate): return None
                    def validate(candidate): return True
                    def benchmark(candidate): return {"latency_ms": 1.0}
                    def metrics():
                        return json.loads(
                            Path(__file__).with_name("dynamic-objective.json").read_text()
                        )
                    def cleanup(): return None
                    """
                ),
                "utf-8",
            )
            metrics_spec = self.workloads.normalize_workload(
                workload=dynamic_adapter
            )
            dynamic_objective.write_text(
                json.dumps(_objective(name="throughput", direction="higher")),
                "utf-8",
            )
            with self.assertRaisesRegex(ValueError, "conflicting objective"):
                self.workloads.run_spec_once(
                    metrics_spec,
                    candidate="candidate.py",
                    role="candidate",
                )

    def test_run_spec_metrics_dynamic_import_uses_frozen_dependency_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module_name = "cuda_optimizer_dynamic_metrics_helper"
            helper = root / f"{module_name}.py"
            helper.write_text(f"OBJECTIVE = {_objective()!r}\n", "utf-8")
            adapter = root / "workload.py"
            adapter.write_text(
                textwrap.dedent(
                    f"""
                    WORKLOAD_DEPENDENCIES = ["{module_name}.py"]
                    def prepare(candidate): return None
                    def validate(candidate): return True
                    def benchmark(candidate): return {{"latency_ms": 1.0}}
                    def metrics():
                        if "CUDA_OPTIMIZER_CONTEXT" in globals():
                            raise AssertionError("metrics received observation context")
                        import {module_name} as helper
                        return helper.OBJECTIVE
                    def cleanup(): return None
                    """
                ),
                "utf-8",
            )
            previous = sys.modules.pop(module_name, None)
            try:
                spec = self.workloads.normalize_workload(workload=adapter)

                try:
                    result = self.workloads.run_spec_once(
                        spec,
                        candidate="candidate.py",
                        role="candidate",
                        case={"batch": 4},
                    )
                except ModuleNotFoundError as error:
                    self.fail(f"frozen metrics dependency was not installed: {error}")
            finally:
                if previous is not None:
                    sys.modules[module_name] = previous

            self.assertEqual(result["objective"], _objective())

    def test_command_validation_failure_is_not_treated_as_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            objective_path = root / "objective.json"
            objective_path.write_text(json.dumps(_objective()), "utf-8")
            runner = self._command_runner(
                root / "runner",
                {
                    "validation": {"valid": False, "reason": "wrong"},
                    "benchmark": {"latency_ms": 0.1},
                },
            )
            spec = self.workloads.normalize_workload(
                workload_cmd=[str(runner)], objective=objective_path
            )

            with self.assertRaisesRegex(ValueError, "validation failed"):
                self.workloads.run_spec_once(
                    spec,
                    candidate="candidate.py",
                    role="candidate",
                )


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
        primary_name = schema["properties"]["primary_metric"]["properties"]["name"]
        constraint_schema = schema["properties"]["constraints"]
        constraint_name = constraint_schema["items"]["properties"]["name"]
        self.assertIn("pattern", primary_name)
        self.assertIn("pattern", constraint_name)
        self.assertTrue(constraint_schema["uniqueItems"])
        self.assertEqual(constraint_schema["x-unique-by"], "name")
        self.assertIn("runtime", constraint_schema["$comment"].lower())
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

    def test_preflight_rejects_manifest_python_metrics_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline, ref = self._valid_kernel_inputs(root)
            adapter = root / "workload.py"
            _write_python_adapter(
                adapter,
                objective=_objective(name="throughput", direction="higher"),
            )
            manifest = root / "workload.json"
            manifest.write_text(
                json.dumps(
                    {
                        "kind": "python",
                        "source": "workload.py",
                        "objective": _objective(),
                        "cases": [],
                    }
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
                "--workload-manifest",
                manifest,
            )

            self.assertNotEqual(completed.returncode, 0)
            report = json.loads(completed.stdout)
            self.assertIn("conflicting objective", " ".join(report["errors"]))

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


class SourceIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workloads = _load_workload_adapter()

    def test_same_mtime_same_size_source_change_never_reuses_pyc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp) / "workload.py"
            _write_python_adapter(adapter, objective=_objective(name="p50_latency_ms"))
            first = self.workloads.normalize_workload(workload=adapter)
            original_stat = adapter.stat()
            original = adapter.read_text("utf-8")
            changed = original.replace("p50_latency_ms", "q50_latency_ms")
            self.assertEqual(len(original.encode()), len(changed.encode()))
            adapter.write_text(changed, "utf-8")
            os.utime(
                adapter,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )

            second = self.workloads.normalize_workload(workload=adapter)

            self.assertEqual(first.objective["primary_metric"]["name"], "p50_latency_ms")
            self.assertEqual(second.objective["primary_metric"]["name"], "q50_latency_ms")
            self.assertNotEqual(first.source_hash, second.source_hash)

    def test_declared_dependency_bytes_are_frozen_and_ignore_old_module_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module_name = "cuda_optimizer_declared_helper"
            helper = root / f"{module_name}.py"
            helper.write_text(f"OBJECTIVE = {_objective(name='p50_latency_ms')!r}\n", "utf-8")
            adapter = root / "workload.py"
            adapter.write_text(
                textwrap.dedent(
                    f"""
                    import {module_name} as helper
                    WORKLOAD_DEPENDENCIES = ["{module_name}.py"]
                    def prepare(candidate): return None
                    def validate(candidate): return True
                    def benchmark(candidate): return {{"latency_ms": 1.0}}
                    def metrics(): return helper.OBJECTIVE
                    def cleanup(): return None
                    """
                ),
                "utf-8",
            )
            stale = types.ModuleType(module_name)
            stale.OBJECTIVE = _objective(name="stale_metric")
            previous = sys.modules.get(module_name)
            sys.modules[module_name] = stale
            try:
                first = self.workloads.normalize_workload(workload=adapter)
                helper_stat = helper.stat()
                changed = helper.read_text("utf-8").replace(
                    "p50_latency_ms", "q50_latency_ms"
                )
                helper.write_text(changed, "utf-8")
                os.utime(
                    helper,
                    ns=(helper_stat.st_atime_ns, helper_stat.st_mtime_ns),
                )
                second = self.workloads.normalize_workload(workload=adapter)
            finally:
                if previous is None:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = previous

            self.assertEqual(first.objective["primary_metric"]["name"], "p50_latency_ms")
            self.assertEqual(second.objective["primary_metric"]["name"], "q50_latency_ms")
            self.assertNotEqual(first.source_hash, second.source_hash)

    def test_source_and_dependencies_must_be_local_regular_non_symlink_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real.py"
            _write_python_adapter(real)
            linked = root / "linked.py"
            linked.symlink_to(real)
            with self.assertRaisesRegex(ValueError, "symlink"):
                self.workloads.normalize_workload(workload=linked)

            outside = root.parent / f"{root.name}-outside.py"
            outside.write_text("VALUE = 1\n", "utf-8")
            try:
                for dependency in ("../" + outside.name, "linked-helper.py"):
                    helper_link = root / "linked-helper.py"
                    if dependency == "linked-helper.py" and not helper_link.exists():
                        helper_link.symlink_to(outside)
                    adapter = root / "adapter.py"
                    _write_python_adapter(adapter)
                    adapter.write_text(
                        f'WORKLOAD_DEPENDENCIES = ["{dependency}"]\n'
                        + adapter.read_text("utf-8"),
                        "utf-8",
                    )
                    with self.subTest(dependency=dependency), self.assertRaisesRegex(
                        ValueError, "dependency|symlink|escape"
                    ):
                        self.workloads.normalize_workload(workload=adapter)
            finally:
                outside.unlink(missing_ok=True)

    def test_dependency_rejects_symlink_directory_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter_root = root / "adapter-root"
            outside = root / "outside"
            adapter_root.mkdir()
            outside.mkdir()
            (outside / "helper.py").write_text("VALUE = 1\n", "utf-8")
            (adapter_root / "linked").symlink_to(outside, target_is_directory=True)
            adapter = adapter_root / "workload.py"
            _write_python_adapter(adapter)
            adapter.write_text(
                'WORKLOAD_DEPENDENCIES = ["linked/helper.py"]\n'
                + adapter.read_text("utf-8"),
                "utf-8",
            )

            with self.assertRaisesRegex(ValueError, "dependency|symlink|escape"):
                self.workloads.normalize_workload(workload=adapter)

    def test_dependency_allows_ordinary_nested_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "nested"
            nested.mkdir()
            (nested / "helper.py").write_text("VALUE = 1\n", "utf-8")
            adapter = root / "workload.py"
            _write_python_adapter(adapter)
            adapter.write_text(
                'WORKLOAD_DEPENDENCIES = ["nested/helper.py"]\n'
                + adapter.read_text("utf-8"),
                "utf-8",
            )

            spec = self.workloads.normalize_workload(workload=adapter)

            self.assertEqual(spec.kind, "python")
            self.assertEqual(len(spec.source_hash), 64)


class CommandSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workloads = _load_workload_adapter()

    @staticmethod
    def _objective_file(root: Path) -> Path:
        path = root / "objective.json"
        path.write_text(json.dumps(_objective()), "utf-8")
        return path

    def test_path_executable_is_absolute_frozen_and_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = _write_executable(
                root / "cuda-user-runner",
                """
                import json, os
                from pathlib import Path
                Path(os.environ["CUDA_OPTIMIZER_OUTPUT"]).write_text(json.dumps({
                    "validation": True,
                    "benchmark": {"latency_ms": 1.0}
                }))
                """,
            )
            with mock.patch.dict(os.environ, {"PATH": str(root)}):
                spec = self.workloads.normalize_workload(
                    workload_cmd=["cuda-user-runner"],
                    objective=self._objective_file(root),
                )
                self.assertEqual(spec.source[0], str(runner.resolve()))
                runner.write_text(
                    runner.read_text("utf-8").replace("1.0", "2.0"), "utf-8"
                )
                runner.chmod(0o755)
                with self.assertRaisesRegex(ValueError, "source_hash"):
                    self.workloads.run_spec_once(
                        spec, candidate="candidate.py", role="candidate"
                    )

    def test_path_symlink_executable_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = _write_executable(root / "real-runner", "pass\n")
            (root / "linked-runner").symlink_to(real)
            with mock.patch.dict(os.environ, {"PATH": str(root)}), self.assertRaisesRegex(
                ValueError, "symlink"
            ):
                self.workloads.normalize_workload(
                    workload_cmd="linked-runner",
                    objective=self._objective_file(root),
                )

    def test_output_is_bounded_redacted_and_secret_env_requires_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = _write_executable(
                root / "noisy-runner",
                """
                import os, sys
                sys.stdout.write("x" * 2_000_000)
                sys.stderr.write(" API_TOKEN=" + str(os.getenv("API_TOKEN")))
                raise SystemExit(9)
                """,
            )
            env = {
                "API_TOKEN": "top-secret-token",
                "CUDA_OPTIMIZER_PASS_ENV": "API_TOKEN",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                spec = self.workloads.normalize_workload(
                    workload_cmd=[str(runner)],
                    objective=self._objective_file(root),
                )
                with self.assertRaises(RuntimeError) as raised:
                    self.workloads.run_command_once(
                        spec,
                        candidate="candidate.py",
                        role="candidate",
                        case={},
                    )
            message = str(raised.exception)
            self.assertLess(len(message), 12000)
            self.assertNotIn("top-secret-token", message)
            self.assertIn("REDACTED", message)

    def test_timeout_terminates_grandchild_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pid_file = root / "child.pid"
            runner = _write_executable(
                root / "tree-runner",
                """
                import subprocess, sys, time
                from pathlib import Path
                child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
                Path(sys.argv[1]).write_text(str(child.pid))
                time.sleep(30)
                """,
            )
            spec = self.workloads.normalize_workload(
                workload_cmd=[str(runner), str(pid_file)],
                objective=self._objective_file(root),
            )
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                self.workloads.run_command_once(
                    spec,
                    candidate="candidate.py",
                    role="candidate",
                    case={},
                    timeout=0.5,
                )
            self.assertTrue(pid_file.exists())
            child_pid = int(pid_file.read_text("utf-8"))
            alive = True
            deadline = time.time() + 2
            while time.time() < deadline:
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    alive = False
                    break
                time.sleep(0.05)
            if alive:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    alive = False
            self.assertFalse(alive, "grandchild survived workload timeout")

    def test_timeout_kills_term_ignoring_grandchild_after_leader_exits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for attempt in range(3):
                pid_file = root / f"child-{attempt}.pid"
                ready_file = root / f"child-{attempt}.ready"
                runner = _write_executable(
                    root / f"tree-runner-{attempt}",
                    """
                    import signal, subprocess, sys, time
                    from pathlib import Path

                    def raise_exit():
                        raise SystemExit(0)

                    signal.signal(signal.SIGTERM, lambda *_: raise_exit())
                    child_code = (
                        "import signal, sys, time; from pathlib import Path; "
                        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                        "Path(sys.argv[1]).write_text('ready'); time.sleep(5)"
                    )
                    child = subprocess.Popen([sys.executable, "-c", child_code, sys.argv[2]])
                    while not Path(sys.argv[2]).exists():
                        time.sleep(0.005)
                    Path(sys.argv[1]).write_text(str(child.pid))
                    time.sleep(30)
                    """,
                )
                spec = self.workloads.normalize_workload(
                    workload_cmd=[str(runner), str(pid_file), str(ready_file)],
                    objective=self._objective_file(root),
                )
                started = time.monotonic()
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    self.workloads.run_command_once(
                        spec,
                        candidate="candidate.py",
                        role="candidate",
                        case={},
                        timeout=1.5,
                    )
                elapsed = time.monotonic() - started
                self.assertLess(elapsed, 3.0, f"attempt {attempt} exceeded cleanup bound")
                self.assertTrue(pid_file.exists())
                child_pid = int(pid_file.read_text("utf-8"))
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    try:
                        os.kill(child_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.01)
                else:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.fail(f"attempt {attempt} left grandchild {child_pid} alive")

    def test_reader_start_failure_still_cleans_process_group_and_pipes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = _write_executable(root / "sleep-runner", "import time; time.sleep(4)\n")
            spec = self.workloads.normalize_workload(
                workload_cmd=[str(runner)],
                objective=self._objective_file(root),
            )
            spawned = []
            real_popen = subprocess.Popen

            def record_process(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                spawned.append(process)
                return process

            try:
                with mock.patch.object(
                    self.workloads.subprocess, "Popen", side_effect=record_process
                ), mock.patch.object(
                    self.workloads.threading.Thread,
                    "start",
                    side_effect=RuntimeError("thread start failed"),
                ), self.assertRaisesRegex(RuntimeError, "thread start failed"):
                    self.workloads.run_command_once(
                        spec,
                        candidate="candidate.py",
                        role="candidate",
                        case={},
                        timeout=1,
                    )
                self.assertEqual(len(spawned), 1)
                process = spawned[0]
                deadline = time.monotonic() + 1.0
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertIsNotNone(process.poll(), "runner survived reader start failure")
                self.assertTrue(process.stdout.closed)
                self.assertTrue(process.stderr.closed)
            finally:
                for process in spawned:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=2)


class ContextAndStrictJsonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workloads = _load_workload_adapter()

    def test_python_lifecycle_receives_detached_normalized_context(self) -> None:
        events = []

        class Adapter:
            def prepare(self, candidate):
                events.append(("prepare", copy.deepcopy(self.CUDA_OPTIMIZER_CONTEXT)))
                self.CUDA_OPTIMIZER_CONTEXT["case"]["batch"] = 99

            def validate(self, candidate):
                events.append(("validate", copy.deepcopy(self.CUDA_OPTIMIZER_CONTEXT)))
                return True

            def benchmark(self, candidate):
                events.append(("benchmark", copy.deepcopy(self.CUDA_OPTIMIZER_CONTEXT)))
                return {"latency_ms": 1.0}

            def metrics(self):
                events.append(("metrics", hasattr(self, "CUDA_OPTIMIZER_CONTEXT")))
                return _objective()

            def cleanup(self):
                events.append(("cleanup", copy.deepcopy(self.CUDA_OPTIMIZER_CONTEXT)))

        adapter = Adapter()
        case = {"batch": 4, "shape": [2, 8]}
        result = self.workloads.run_once(
            adapter,
            candidate="candidate.py",
            role="  candidate  ",
            case=case,
        )

        self.assertEqual(result["role"], "candidate")
        self.assertEqual(result["case"], case)
        self.assertEqual(case["batch"], 4)
        self.assertEqual(
            [name for name, _ in events],
            ["prepare", "validate", "benchmark", "cleanup", "metrics"],
        )
        self.assertEqual(events[0][1]["case"]["batch"], 4)
        self.assertEqual(events[-2][1]["case"]["batch"], 99)
        self.assertFalse(events[-1][1], "metrics received observation context")
        self.assertFalse(hasattr(adapter, "CUDA_OPTIMIZER_CONTEXT"))

    def test_case_must_be_json_safe_mapping(self) -> None:
        adapter, _ = PythonLifecycleTests()._adapter()
        for invalid, message in (
            ([], "case"),
            ({1: "bad"}, "string keys"),
            ({"value": math.nan}, "finite"),
            ({"value": object()}, "JSON"),
        ):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, message
            ):
                self.workloads.run_once(
                    adapter,
                    candidate="candidate.py",
                    role="candidate",
                    case=invalid,
                )

    def test_template_documents_exact_runtime_contract(self) -> None:
        template = (TEMPLATE_DIR / "workload.py").read_text("utf-8")
        for text in (
            "WORKLOAD_DEPENDENCIES",
            "CUDA_OPTIMIZER_CONTEXT",
            "literal bool",
            "finite JSON",
            "environment",
            "cleanup",
        ):
            self.assertIn(text, template)
        self.assertIn(
            "prepare(), validate(), benchmark(), and cleanup()",
            template,
        )
        self.assertIn("normalization/preflight", template)
        self.assertIn("no candidate, role, or case context", template)
        self.assertIn("lightweight pure function", template)

    def test_objective_and_manifest_reject_duplicate_and_nonfinite_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, payload, call, message in (
                (
                    "duplicate-objective.json",
                    '{"primary_metric":{"name":"a","name":"b","direction":"lower"},"min_effect_pct":1,"constraints":[]}',
                    lambda path: self.workloads.load_objective(path),
                    "duplicate",
                ),
                (
                    "nan-objective.json",
                    '{"primary_metric":{"name":"a","direction":"lower"},"min_effect_pct":NaN,"constraints":[]}',
                    lambda path: self.workloads.load_objective(path),
                    "strict JSON|non-finite JSON",
                ),
                (
                    "duplicate-manifest.json",
                    '{"kind":"command","kind":"python","source":"x","cases":[],"objective":{}}',
                    lambda path: self.workloads.normalize_workload(workload_manifest=path),
                    "duplicate",
                ),
            ):
                path = root / name
                path.write_text(payload, "utf-8")
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                    call(path)


if __name__ == "__main__":
    unittest.main()
