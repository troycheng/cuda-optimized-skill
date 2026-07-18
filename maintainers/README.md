# Maintainer material

This directory is not part of the user guide or the runtime skill context.

- `history/` preserves design and implementation records created while earlier
  versions were developed. Git history is authoritative when a record differs
  from current behavior.
- New long-lived design decisions should be short, current, and placed in a
  future `decisions/` directory. Temporary implementation plans should normally
  remain outside the public documentation tree.

User documentation belongs in `docs/`. Agent execution instructions and
on-demand technical knowledge belong in `skills/cuda-kernel-optimizer/`.
