#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

using RowMajor = cutlass::layout::RowMajor;
using Gemm = cutlass::gemm::device::Gemm<
    float,
    RowMajor,
    float,
    RowMajor,
    float,
    RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

extern "C" void solve(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K) {
    Gemm gemm;
    typename Gemm::Arguments arguments(
        {M, N, K},
        {A, K},
        {B, N},
        {C, N},
        {C, N},
        {1.0f, 0.0f});
    gemm(arguments);
}
