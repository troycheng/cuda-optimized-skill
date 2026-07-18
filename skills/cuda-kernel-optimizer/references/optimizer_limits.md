# Optimizer capability and limits

The skill assumes the agent can reason about CUDA execution, memory hierarchy,
occupancy, latency hiding, CUTLASS/CuTe, Triton, compiler IR, PTX/SASS, profiling,
and common PyTorch or serving bottlenecks. That knowledge is useful for forming
hypotheses, not for declaring target behavior without measurement.

## Limits that require evidence

- current compiler code generation and undocumented microarchitecture behavior;
- proprietary operator semantics and hidden application constraints;
- the target shape, data, concurrency, and request distribution;
- CPU, GPU, network, scheduler, allocator, and I/O interactions;
- whether a candidate improves the actual objective on the target machine.

## How the skill compensates

- capability detection instead of architecture inference;
- compact profiler and workload summaries instead of raw-log intuition;
- bottleneck-conditioned knowledge retrieval instead of global method priority;
- explicit hypotheses, falsification tests, and bounded search;
- correctness oracles, paired timing, identity binding, and claim ceilings;
- strategy memory for failed methods and stop rules for exhausted directions;
- optional independent challenge when local reasoning plateaus.

The skill cannot compensate for a missing runnable target, representative input,
correctness oracle, or measurement environment. In those cases it must surface
the gap, help prepare the foundation, and stop below the unsupported claim.
