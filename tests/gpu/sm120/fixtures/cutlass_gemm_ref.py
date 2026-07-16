from __future__ import annotations

atol = 5e-2
rtol = 2e-2


def reference(A, B, C, M: int, N: int, K: int) -> None:
    a = A[: M * K].view(M, K)
    b = B[: K * N].view(K, N)
    C[: M * N].view(M, N).copy_(a @ b)
