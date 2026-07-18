# Offline optimization knowledge

The bundled knowledge is a compact routing aid for offline work. It is not a
mirror of CUDA, Triton, CUTLASS, PyTorch, vLLM, or profiler manuals.

## Contents

- `method_registry.json`: kernel methods and exact architecture capabilities.
- `workload_methods.json`: framework, CPU/data, transfer, communication, I/O,
  serving, and environment diagnosis cards.
- `knowledge_sources.json`: primary source, version, and freshness manifest.
- `ncu_metrics_guide.md`: profiler metric interpretation and degradation rules.
- `sass_signatures.json`: generated-code evidence patterns.
- `compatibility.md`: observed toolchain versions and architecture boundaries.

Query a bounded subset with `scripts/knowledge_query.py`. Do not load the full
catalog when the architecture, layer, and bottleneck are already known.
Pass a compact profiler metric JSON when available. Matching bad signals are
ranked first; cards without an observed signal are labeled `unverified`, and a
non-triggered observed signal is ranked last. Registry priority alone is not
direction-admission evidence.

## Freshness rules

1. Match the exact SM capability set. Never inherit features by numeric SM
   ordering.
2. Compare the local driver, toolkit, compiler, framework, and profiler versions
   with `knowledge_sources.json` and `compatibility.md`.
3. Treat a version mismatch, undocumented behavior, renamed metric, or missing
   source as unverified.
4. Prefer local capability probes and generated-code evidence over remembered
   syntax.
5. Treat historical speedup ranges as context, never as expected gain or
   promotion evidence.

Offline mode must still work when every external provider is unavailable. A
stale card may suggest a measurement, but it cannot establish that an API or
hardware feature exists on the target.
