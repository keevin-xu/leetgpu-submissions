#ifndef LEETGPU_CUDA_RUNTIME_H
#define LEETGPU_CUDA_RUNTIME_H
#include <stddef.h>
#include <stdint.h>
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __shared__
#define __shared__
#endif
#ifndef __constant__
#define __constant__
#endif
#ifndef __managed__
#define __managed__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif
struct uint3 { unsigned int x, y, z; };
struct dim3 {
  unsigned int x, y, z;
  constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
};
extern "C" unsigned __cudaPushCallConfiguration(
  dim3 gridDim,
  dim3 blockDim,
  size_t sharedMem = 0,
  void *stream = 0
);
extern const uint3 threadIdx;
extern const uint3 blockIdx;
extern const dim3 blockDim;
extern const dim3 gridDim;
extern const int warpSize;
typedef enum cudaError { cudaSuccess = 0 } cudaError_t;
typedef void *cudaStream_t;
typedef void *cudaEvent_t;
typedef enum cudaMemcpyKind {
  cudaMemcpyHostToHost,
  cudaMemcpyHostToDevice,
  cudaMemcpyDeviceToHost,
  cudaMemcpyDeviceToDevice,
  cudaMemcpyDefault
} cudaMemcpyKind;
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char *cudaGetErrorString(cudaError_t error);
cudaError_t cudaMalloc(void **ptr, size_t size);
cudaError_t cudaFree(void *ptr);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0);
cudaError_t cudaMemset(void *ptr, int value, size_t count);
cudaError_t cudaMemsetAsync(void *ptr, int value, size_t count, cudaStream_t stream = 0);
cudaError_t cudaStreamCreate(cudaStream_t *stream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float *milliseconds, cudaEvent_t start, cudaEvent_t end);

// CUDA single-precision device intrinsics. These declarations mirror the
// stable scalar intrinsic set exposed without an additional CUDA header.
__device__ float __cosf(float x);
__device__ float __exp10f(float x);
__device__ float __expf(float x);
__device__ float __fadd_rd(float x, float y);
__device__ float __fadd_rn(float x, float y);
__device__ float __fadd_ru(float x, float y);
__device__ float __fadd_rz(float x, float y);
__device__ float __fdiv_rd(float x, float y);
__device__ float __fdiv_rn(float x, float y);
__device__ float __fdiv_ru(float x, float y);
__device__ float __fdiv_rz(float x, float y);
__device__ float __fdividef(float x, float y);
__device__ float __fmaf_rd(float x, float y, float z);
__device__ float __fmaf_rn(float x, float y, float z);
__device__ float __fmaf_ru(float x, float y, float z);
__device__ float __fmaf_rz(float x, float y, float z);
__device__ float __fmul_rd(float x, float y);
__device__ float __fmul_rn(float x, float y);
__device__ float __fmul_ru(float x, float y);
__device__ float __fmul_rz(float x, float y);
__device__ float __frcp_rd(float x);
__device__ float __frcp_rn(float x);
__device__ float __frcp_ru(float x);
__device__ float __frcp_rz(float x);
__device__ float __frsqrt_rn(float x);
__device__ float __fsqrt_rd(float x);
__device__ float __fsqrt_rn(float x);
__device__ float __fsqrt_ru(float x);
__device__ float __fsqrt_rz(float x);
__device__ float __fsub_rd(float x, float y);
__device__ float __fsub_rn(float x, float y);
__device__ float __fsub_ru(float x, float y);
__device__ float __fsub_rz(float x, float y);
__device__ float __log10f(float x);
__device__ float __log2f(float x);
__device__ float __logf(float x);
__device__ float __powf(float x, float y);
__device__ float __saturatef(float x);
__device__ void __sincosf(float x, float *sptr, float *cptr);
__device__ float __sinf(float x);
__device__ float __tanf(float x);

__device__ void __syncthreads(void);
__device__ int __syncthreads_count(int predicate);
__device__ int __syncthreads_and(int predicate);
__device__ int __syncthreads_or(int predicate);
__device__ void __syncwarp(unsigned mask = 0xffffffffu);
__device__ void __threadfence_block(void);
__device__ void __threadfence(void);
__device__ void __threadfence_system(void);
__device__ int __all_sync(unsigned mask, int predicate);
__device__ int __any_sync(unsigned mask, int predicate);
__device__ unsigned __ballot_sync(unsigned mask, int predicate);
__device__ unsigned __activemask(void);
template <typename T> __device__ T atomicAdd(T *address, T value);
template <typename T> __device__ T atomicSub(T *address, T value);
template <typename T> __device__ T atomicMax(T *address, T value);
template <typename T> __device__ T atomicMin(T *address, T value);
template <typename T> __device__ T atomicExch(T *address, T value);
template <typename T> __device__ T atomicCAS(T *address, T compare, T value);
template <typename T> __device__ T atomicAnd(T *address, T value);
template <typename T> __device__ T atomicOr(T *address, T value);
template <typename T> __device__ T atomicXor(T *address, T value);
__device__ unsigned atomicInc(unsigned *address, unsigned limit);
__device__ unsigned atomicDec(unsigned *address, unsigned limit);
template <typename T> __device__ T __shfl_sync(unsigned mask, T value, int srcLane, int width = 32);
template <typename T> __device__ T __shfl_down_sync(unsigned mask, T value, unsigned delta, int width = 32);
template <typename T> __device__ T __shfl_up_sync(unsigned mask, T value, unsigned delta, int width = 32);
template <typename T> __device__ T __shfl_xor_sync(unsigned mask, T value, int laneMask, int width = 32);
template <typename T> __device__ unsigned __match_any_sync(unsigned mask, T value);
template <typename T> __device__ unsigned __match_all_sync(unsigned mask, T value, int *predicate);
#endif
