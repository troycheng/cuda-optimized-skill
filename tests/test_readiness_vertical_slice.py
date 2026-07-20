from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / "skills" / "cuda-kernel-optimizer"
SCRIPT = SKILL_DIR / "scripts" / "workload_controller.py"
FIXTURES = ROOT / "tests" / "fixtures" / "readiness"


def _load_controller():
    name = "cuda_optimizer_workload_controller_vertical_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ReadinessVerticalSliceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = _load_controller()

    def _workspace(self, root: Path, statuses: tuple[str, ...]) -> tuple[dict, Path]:
        project = root / "project"
        environment = root / "environment"
        project.mkdir()
        environment.mkdir()
        (project / "src").mkdir()
        (project / "configs").mkdir()
        (project / "src" / "candidate.py").write_text("VALUE = 1\n", "utf-8")
        (project / "configs" / "run.json").write_text("{}\n", "utf-8")
        (project / "adapter.py").write_text(
            textwrap.dedent(
                """
                def prepare(candidate): return None
                def validate(candidate): return {"valid": True}
                def benchmark(candidate): return {"latency_ms": 1.0}
                def metrics():
                    return {"primary_metric": {"name": "latency_ms", "direction": "lower"},
                            "min_effect_pct": 1.0, "constraints": []}
                def cleanup(): return None
                """
            ),
            "utf-8",
        )
        (project / "workload.json").write_text(
            json.dumps(
                {
                    "kind": "python",
                    "source": "adapter.py",
                    "objective": {
                        "primary_metric": {"name": "latency_ms", "direction": "lower"},
                        "min_effect_pct": 1.0,
                        "constraints": [],
                    },
                    "cases": [],
                }
            ),
            "utf-8",
        )
        emitter = project / "emit_probe.py"
        shutil.copyfile(FIXTURES / "emit_probe.py", emitter)
        contract_template = (FIXTURES / "readiness-contract.json.in").read_text("utf-8")
        requirements = []
        for index, status in enumerate(statuses):
            requirements.append(
                {
                    "id": f"probe-{index}",
                    "necessity": "required" if index == 0 else "diagnostic",
                    "control_scope": "project",
                    "phase": "foundation" if index == 0 else "workload",
                    "kind": "gpu_execute" if index == 0 else "ncu_counters",
                    "max_age_seconds": 300,
                    "probe": {
                        "argv": [sys.executable, str(emitter), f"probe-{index}", status],
                        "timeout_seconds": 5,
                    },
                    "remediation": {"mode": "none"},
                }
            )
        contract_path = project / "readiness.json"
        contract_path.write_text(
            contract_template.replace("@REQUIREMENTS@", json.dumps(requirements)),
            "utf-8",
        )
        control_template = (FIXTURES / "control-v2.json.in").read_text("utf-8")
        replacements = {
            "@PROJECT_ROOT@": str(project),
            "@ENVIRONMENT_ROOT@": str(environment),
            "@PYTHON@": sys.executable,
            "@EMITTER@": str(emitter),
        }
        for marker, value in replacements.items():
            control_template = control_template.replace(marker, value)
        return json.loads(control_template), root / "run"

    def test_cli_help_and_sample_contract_are_valid(self) -> None:
        help_result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("run", help_result.stdout)

        with tempfile.TemporaryDirectory() as tmp:
            control, _run_dir = self._workspace(Path(tmp).resolve(), ("ready",))
            normalized = self.controller.validate_control_manifest(control)
            contract_module = self.controller._load_readiness_contract_module()
            contract = contract_module.validate_contract(
                contract_module.load_contract(normalized["readiness_contract"]),
                project_root=Path(normalized["project_root"]),
                environment_root=Path(normalized["mutation"]["environment_root"]),
            )
            self.assertEqual(contract["requested_claim"], "workload")

    def test_blocked_readiness_never_loads_baseline_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            control, run_dir = self._workspace(Path(tmp).resolve(), ("failed",))
            with mock.patch.object(self.controller, "_load_evaluate_module") as evaluate:
                state = self.controller.start_run(control, run_dir)

            self.assertEqual(state["next_action"], "readiness_action")
            evaluate.assert_not_called()
            self.assertFalse((run_dir / "baseline" / "observation.json").exists())

    def test_degraded_diagnostic_readiness_can_enter_mocked_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            control, run_dir = self._workspace(
                Path(tmp).resolve(), ("ready", "degraded")
            )
            evaluator = mock.Mock()
            evaluator.measure_candidate.return_value = {"status": "measured"}
            with mock.patch.object(
                self.controller, "_load_evaluate_module", return_value=evaluator
            ):
                state = self.controller.start_run(control, run_dir)

            self.assertEqual(state["next_action"], "register_change")
            evaluator.measure_candidate.assert_called_once()
            report = json.loads((run_dir / "readiness" / "report.json").read_text("utf-8"))
            self.assertEqual(report["status"], "degraded")
            self.assertTrue(report["can_start_diagnosis"])


if __name__ == "__main__":
    unittest.main()
