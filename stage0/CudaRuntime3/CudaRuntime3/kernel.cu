#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

// 1. 正常合并访存的 Kernel
__global__ void addKernelCoalesced(float* c, const float* a, const float* b, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // 相邻的线程访问相邻的地址，硬件完美合并
        c[i] = a[i] + b[i];
    }
}

// 2. 跨步访问（未合并访存）的 Kernel
__global__ void addKernelStrided(float* c, const float* a, const float* b, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // 假设我们人为制造一种乱序/跨步访问：
        // 每个线程去访问间隔很远的地址（此处用一种简单的反转映射来破坏连续性）
        int strideIndex = (i * 32) % size;
        c[i] = a[strideIndex] + b[strideIndex];
    }
}

// 辅助计时函数
template <typename Kernel>
float runTest(
    Kernel kernel,
    float* d_c,
    const float* d_a,
    const float* d_b,
    int size,
    int blockSize)
{
    int blocks = (size + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<blocks, blockSize>>>(d_c, d_a, d_b, size);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main()
{
    const int arraySize = 50000000; // 5千万个元素，数据量够大才能看出带宽差异
    const int bytes = arraySize * sizeof(float);

    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    for (int i = 0; i < arraySize; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // CUDA Kernel 预热
    const int warmupBlockSize = 256;
    const int warmupBlocks =
        (arraySize + warmupBlockSize - 1) / warmupBlockSize;

    addKernelCoalesced << <warmupBlocks, warmupBlockSize >> > (
        d_c,
        d_a,
        d_b,
        arraySize
        );

    // 检查 Kernel 启动错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // 等待预热 Kernel 执行完成
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    printf("--- 测试 1: 不同 Block Size 对合并访存的影响 ---\n");
    int blockSizes[] = { 32, 64, 128, 256, 512, 1024, 100 };
    for (int bs : blockSizes) {
        float time = runTest(addKernelCoalesced, d_c, d_a, d_b, arraySize, bs);
        printf("Block Size: %4d, Time: %f ms\n", bs, time);
    }

    printf("\n--- 测试 2: 合并访存 vs 跨步访问 (Block Size 固定为 256) ---\n");
    float time_coalesced = runTest(addKernelCoalesced, d_c, d_a, d_b, arraySize, 256);
    float time_strided = runTest(addKernelStrided, d_c, d_a, d_b, arraySize, 256);

    printf("合并访存 (Coalesced): %f ms\n", time_coalesced);
    printf("跨步访问 (Strided)  : %f ms\n", time_strided);

    free(h_a); free(h_b);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}