"""Small user-owned workload adapter for the opt-in SM120 acceptance lane."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

WORKLOAD_DEPENDENCIES = ["objective.json"]

_active = None


def _context() -> dict:
    value = globals().get("CUDA_OPTIMIZER_CONTEXT", {})
    return value if isinstance(value, dict) else {}


def _load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location(
        f"sm120_workload_candidate_{path.stat().st_ino}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load workload candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "setup", None)) or not callable(
        getattr(module, "run_kernel", None)
    ):
        raise ValueError("workload candidate requires setup() and run_kernel()")
    return module


def prepare(candidate):
    """Load one candidate and allocate deterministic real CUDA inputs."""
    global _active
    if _active is not None:
        raise RuntimeError("workload candidate is already prepared")
    supplied = Path(candidate).expanduser().absolute()
    if supplied.is_symlink() or not supplied.is_file():
        raise ValueError("workload smoke accepts a regular Python kernel")
    path = supplied.resolve(strict=True)
    if path.suffix != ".py":
        raise ValueError("workload smoke accepts a Python kernel")
    case = _context().get("case", {})
    size = case.get("N", 1_048_576) if isinstance(case, dict) else 1_048_576
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("workload case N must be a positive integer")
    module = _load_candidate(path)
    state = module.setup(N=size, seed=20260717)
    if not isinstance(state, dict) or not isinstance(state.get("inputs"), dict):
        raise ValueError("candidate setup() must return an inputs mapping")
    _active = {"path": path, "module": module, "state": state}


def _require_active(candidate):
    if _active is None:
        raise RuntimeError("workload candidate is not prepared")
    path = Path(candidate).expanduser().resolve(strict=True)
    if path != _active["path"]:
        raise ValueError("prepared workload candidate changed")
    return _active


def validate(candidate):
    """Execute the candidate and compare its real GPU output to the reference."""
    import torch

    active = _require_active(candidate)
    inputs = active["state"]["inputs"]
    active["module"].run_kernel(**inputs)
    torch.cuda.synchronize()
    expected = inputs["x"] * inputs["x"] + 1.0
    maximum_error = float((inputs["out"] - expected).abs().max().item())
    return {"valid": maximum_error <= 1e-5, "max_abs_error": maximum_error}


def benchmark(candidate):
    """Measure repeated real kernel launches with CUDA events."""
    import torch

    active = _require_active(candidate)
    inputs = active["state"]["inputs"]
    run_kernel = active["module"].run_kernel
    for _ in range(5):
        run_kernel(**inputs)
    torch.cuda.synchronize()
    repeats = 20
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        run_kernel(**inputs)
    end.record()
    end.synchronize()
    latency_ms = float(start.elapsed_time(end)) / repeats
    output_checksum = float(inputs["out"].double().sum().item())
    return {
        "latency_ms": latency_ms,
        "output_checksum": output_checksum,
        "launches_per_observation": repeats,
    }


def metrics():
    """Load the frozen objective also checked into this fixture directory."""
    return json.loads(Path(__file__).with_name("objective.json").read_text("utf-8"))


def cleanup():
    """Release this observation's module and CUDA tensor references."""
    global _active
    _active = None
