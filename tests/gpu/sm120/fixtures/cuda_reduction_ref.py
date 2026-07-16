from __future__ import annotations

atol = 2e-1
rtol = 2e-3


def reference(input, output, N: int) -> None:
    output.zero_()
    output[0] = input[:N].sum()
