# CUDA Kernel Optimizer

**Evidence-driven CUDA, CUTLASS and Triton optimization for Codex.**

`cuda-kernel-optimizer` is a reusable Codex skill that connects profiling,
bounded code changes, correctness checks, and paired performance evaluation. It
supports a single kernel, a complete GPU workload, a serving experiment, or
read-only analysis of an existing Nsight Compute report.

## Start here

- [Getting Started](getting-started.md) — install the skill and prepare a task.
- [Environment readiness](environment-readiness.md) — find the strongest result
  the current workload and test environment can support.
- [Workflows](workflows.md) — choose the claim that matches your inputs.
- [Long-running optimization](long-running-optimization.md) — see how the
  workload contract, controller, evidence ledger, and recurring audits keep a
  long search aligned.
- [Evidence & Safety](evidence-and-safety.md) — understand what must pass before
  a result can be trusted.
- [Compatibility](compatibility.md) — check toolchain and target requirements.
- [Validation status](validation.md) — see where the project itself has been tested.
- [Case studies](case-studies.md) — review workload-specific historical results.
- [Knowledge and research](knowledge-and-research.md) — understand offline
  knowledge, current-source search, and independent model challenge.
- [Agent Protocol](https://github.com/troycheng/cuda-optimized-skill/blob/main/skills/cuda-kernel-optimizer/SKILL.md)
  — read the canonical Codex execution instructions.

## What makes a result usable

A fast sample is not a result by itself. The skill keeps a change only when the
declared correctness checks, paired performance design, constraints, identity
bindings, and required environment evidence close. Kernel-level evidence and
end-to-end workload evidence are reported separately.

Host drivers, permissions, clocks, power limits, and system configuration are
outside the automatic modification scope.
