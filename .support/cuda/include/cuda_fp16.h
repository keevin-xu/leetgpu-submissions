#ifndef LEETGPU_CUDA_FP16_H
#define LEETGPU_CUDA_FP16_H
#include <cuda_runtime.h>
struct __half { unsigned short __x; };
struct __half2 { __half x, y; };
typedef __half half;
typedef __half2 half2;
__host__ __device__ __half __float2half(float value);
__host__ __device__ float __half2float(__half value);
__device__ __half __hadd(__half a, __half b);
__device__ __half __hsub(__half a, __half b);
__device__ __half __hmul(__half a, __half b);
__device__ __half __hdiv(__half a, __half b);
#endif
