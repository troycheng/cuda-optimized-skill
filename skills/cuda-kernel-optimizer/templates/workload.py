"""Copy this file and replace every TODO with your real workload behavior.

The optimizer cannot infer, download, or manufacture a representative
end-to-end workload.  This template intentionally raises until the user has
connected the four observation lifecycle functions and metrics() to their own
system.
"""

# Optional local Python files whose exact bytes are part of the frozen workload
# identity. Keep this a literal list/tuple of non-symlink relative paths.
# External packages, services, and environment state remain user-owned runtime
# dependencies and must be held stable by the experiment procedure.
WORKLOAD_DEPENDENCIES = []


def prepare(candidate):
    """TODO: deploy/load ``candidate`` using CUDA_OPTIMIZER_CONTEXT.

    During prepare(), validate(), benchmark(), and cleanup(), the context is
    exactly
    ``{"role": <trimmed role>, "case": <detached finite JSON object>}``.
    """
    raise NotImplementedError("TODO: prepare the real workload target")


def validate(candidate):
    """TODO: return a literal bool or ``{"valid": <literal bool>, ...}``."""
    raise NotImplementedError("TODO: validate the real workload result")


def benchmark(candidate):
    """TODO: return one finite JSON object from the real user-owned workload."""
    raise NotImplementedError("TODO: benchmark the real workload")


def metrics():
    """TODO: return the explicit optimization objective JSON object.

    The optimizer calls this during normalization/preflight.
    It receives no candidate, role, or case context.
    Keep it a lightweight pure function: do not deploy a candidate, run a
    benchmark, or depend on mutable observation state.

    Example shape only (replace metric names and thresholds with real ones):
      {
        "primary_metric": {"name": "p50_latency_ms", "direction": "lower"},
        "min_effect_pct": 1.0,
        "constraints": [
          {"name": "p99_latency_ms", "max_regression_pct": 0.5}
        ]
      }
    """
    raise NotImplementedError("TODO: define the real workload objective")


def cleanup():
    """TODO: undo prepare() safely; this is always called exactly once."""
    raise NotImplementedError("TODO: clean up the real workload target")
