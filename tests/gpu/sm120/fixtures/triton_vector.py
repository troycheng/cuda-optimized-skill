from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def vector_kernel(x, out, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask)
    tl.store(out + offsets, values * values + 1.0, mask=mask)


def setup(N: int, seed: int | None = None) -> dict:
    torch.manual_seed(seed or 0)
    return {
        "inputs": {
            "x": torch.randn(N, device="cuda", dtype=torch.float32),
            "out": torch.empty(N, device="cuda", dtype=torch.float32),
            "N": N,
        },
        "outputs": ["out"],
    }


def run_kernel(x, out, N: int) -> None:
    block = 256
    vector_kernel[(triton.cdiv(N, block),)](x, out, N, BLOCK=block)
