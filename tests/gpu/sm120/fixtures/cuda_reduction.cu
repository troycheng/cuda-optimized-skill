#include <cuda_runtime.h>

__global__ void reduce_atomic_kernel(const float* input, float* output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        atomicAdd(output, input[index]);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    reduce_atomic_kernel<<<blocks, threads>>>(input, output, N);
}
