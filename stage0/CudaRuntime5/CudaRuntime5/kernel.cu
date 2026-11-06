#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define N 2048
#define BLOCK_SIZE 32

// 1. 无 Padding 版本 (会产生严重的 Bank Conflict)
__global__ void transposeNoPad(float* out, const float* in, int n)
{
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE]; // 32 x 32

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x < n && y < n) tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    __syncthreads();

    int new_x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    int new_y = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    // 读取 Shared Memory 时，按列读取，发生冲突
    if (new_x < n && new_y < n) out[new_y * n + new_x] = tile[threadIdx.x][threadIdx.y];
}

// 2. 有 Padding 版本 (完美避开冲突)
__global__ void transposeWithPad(float* out, const float* in, int n)
{
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1]; // 32 x 33

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x < n && y < n) tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    __syncthreads();

    int new_x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    int new_y = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    // 读取 Shared Memory 时，按列读取，但完美错开 Bank
    if (new_x < n && new_y < n) out[new_y * n + new_x] = tile[threadIdx.x][threadIdx.y];
}

// 计时函数
float runTest(void (*kernel)(float*, const float*, int), float* d_out, float* d_in, int n, const char* name)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    kernel << <grid, block >> > (d_out, d_in, n);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    int iterations = 20;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel << <grid, block >> > (d_out, d_in, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iterations;
    printf("%s -> Time: %8.4f ms\n", name, avg_ms);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    return avg_ms;
}

int main()
{
    const int bytes = N * N * sizeof(float);
    float* h_in = (float*)malloc(bytes);
    float* d_in, * d_out;
    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    printf("--- 矩阵转置 Bank Conflict 验证 (%d x %d) ---\n", N, N);
    // 务必使用 Release 模式
    runTest(transposeNoPad, d_out, d_in, N, "No Padding (32x32)  ");
    runTest(transposeWithPad, d_out, d_in, N, "With Padding (32x33)");

    free(h_in);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}