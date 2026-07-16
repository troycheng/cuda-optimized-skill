"""Copy this file and replace every TODO with your real workload behavior.

The optimizer cannot infer, download, or manufacture a representative
end-to-end workload.  This template intentionally raises until the user has
connected all five lifecycle functions to their own system.
"""


def prepare(candidate):
    """TODO: deploy/load ``candidate`` into the user's real test target."""
    raise NotImplementedError("TODO: prepare the real workload target")


def validate(candidate):
    """TODO: validate correctness before collecting performance evidence."""
    raise NotImplementedError("TODO: validate the real workload result")


def benchmark(candidate):
    """TODO: run one observation and return the raw user-owned result."""
    raise NotImplementedError("TODO: benchmark the real workload")


def metrics():
    """TODO: return the explicit optimization objective JSON object.

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
