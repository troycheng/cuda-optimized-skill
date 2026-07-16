from __future__ import annotations


def reference(x, out, N: int) -> None:
    out.copy_(x * x + 1.0)
