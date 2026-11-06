#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define N 2048 // 矩阵维度 (2048 x 2048, 约 16MB)
#define BLOCK_SIZE 32 // 平铺的块大小

// 1. Naive 转置：读取合并，写入跨步（极慢）
__global__ void transposeNaive(float* out, const float* in, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        // 读取 in[row][col] 是连续的（合并）
        // 写入 out[col][row] 时，由于行号突变，是跨步的（非合并）
        out[col * n + row] = in[row * n + col];
    }
}

// 2. 共享内存转置：读取合并，写入合并（极快）
__global__ void transposeShared(float* out, const float* in, int n)
{
    // 申请一块二维共享内存，大小是 32 x 33
    // 为什么是 33 而不是 32？这是为了避开 Bank Conflict（银行冲突），明天细讲，今天先用
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // 1. 读取全局内存到共享内存 (合并读取)
    if (x < n && y < n) {
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    }

    // 必须等待同一个 Block 里的所有线程都完成数据拷贝
    __syncthreads();

    // 2. 从共享内存读出，写到全局内存 (合并写入)
    // 注意这里行列反转了
    int new_x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    int new_y = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    if (new_x < n && new_y < n) {
        out[new_y * n + new_x] = tile[threadIdx.x][threadIdx.y];
    }
}

// 计时辅助函数
float runTest(void (*kernel)(float*, const float*, int), float* d_out, float* d_in, int n, const char* name)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    // 预热
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

    // 计算带宽: 读一次 + 写一次
    float bandwidth = (n * n * sizeof(float) * 2.0f) / (avg_ms * 1000000.0f);
    printf("%s -> Time: %8.4f ms | Bandwidth: %6.2f GB/s\n", name, avg_ms, bandwidth);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    return avg_ms;
}

int main()
{
    const int bytes = N * N * sizeof(float);
    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);

    for (int i = 0; i < N * N; i++) h_in[i] = i;

    float* d_in, * d_out;
    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    printf("--- 矩阵转置性能对比 (%d x %d) ---\n", N, N);
    // 请务必在 VS 顶部切换到 Release 模式再运行！
    runTest(transposeNaive, d_out, d_in, N, "Naive Transpose ");
    runTest(transposeShared, d_out, d_in, N, "Shared Transpose");

    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}