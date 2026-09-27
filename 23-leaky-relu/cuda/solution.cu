// LeetGPU Solution
// Challenge: #23 · Leaky ReLU
// Language: CUDA (cuda)

#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        //don't call input[i] multiple times. reads from global mem are slow
        float curr = input[i];
        output[i] = max(0.0f, curr);
        if (output[i] == 0) {
            output[i] = curr * 0.01;
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
