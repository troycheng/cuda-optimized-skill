# Case studies

These are historical examples of how the workflow reached a decision. Their
numbers apply only to the recorded code, inputs, environment, and objective.

## Reproducible workload fixture

The fixture's end-to-end latency improved by 60.4616% while its declared
constraints passed. This demonstrates that the complete workflow can preserve a
reference, measure a baseline and candidate, and retain a verified change. It
does not set an expected gain for another project.

## User-provided vLLM workload

An isolated kernel metric improved by 26.3287%, while the real workload changed
by -0.0097%. The original implementation was retained because the product
objective did not improve. This is the intended separation between a kernel
result and an end-to-end result.

## Existing Nsight Compute report

The read-only importer parsed 140 metrics without launching the profiled
program. The result supported diagnosis only: it did not establish current GPU
counter permissions, binary identity, or environment cleanliness.

See [Validation status](validation.md) for project checks rather than workload
outcomes.
