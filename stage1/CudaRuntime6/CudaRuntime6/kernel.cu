#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>

#define N 1024

// Naive 矩阵乘法
__global__ void gemmNaive(
    float* C,
    const float* A,
    const float* B,
    int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;

        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }

        C[row * n + col] = sum;
    }
}

int main()
{
    int bytes = N * N * sizeof(float);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);

    if (h_A == nullptr || h_B == nullptr) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    // 初始化输入矩阵
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&d_A, bytes);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc d_A failed: %s\n",
            cudaGetErrorString(cudaStatus));
        return 1;
    }

    cudaStatus = cudaMalloc((void**)&d_B, bytes);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc d_B failed: %s\n",
            cudaGetErrorString(cudaStatus));
        cudaFree(d_A);
        return 1;
    }

    cudaStatus = cudaMalloc((void**)&d_C, bytes);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc d_C failed: %s\n",
            cudaGetErrorString(cudaStatus));
        cudaFree(d_A);
        cudaFree(d_B);
        return 1;
    }

    cudaMemcpy(
        d_A,
        h_A,
        bytes,
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_B,
        h_B,
        bytes,
        cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y);

    int iterations = 20;

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ==================================================
    // Naive GEMM
    // ==================================================

    gemmNaive << <grid, block >> > (d_C, d_A, d_B, N);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("Naive kernel launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    }

    cudaDeviceSynchronize();

    float ms = 0.0f;

    cudaEventRecord(start);

    for (int i = 0; i < iterations; i++) {
        gemmNaive << <grid, block >> > (d_C, d_A, d_B, N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);

    float avg_ms = ms / iterations;

    double gflops =
        (2.0 * N * N * N) /
        (avg_ms * 1000000.0);

    printf(
        "Naive GEMM  -> Time: %8.4f ms | Performance: %6.2f GFLOPS\n",
        avg_ms,
        gflops);

    // ==================================================
    // cuBLAS SGEMM
    // ==================================================

    cublasHandle_t handle = nullptr;

    // 这里只调用一次 cublasCreate
    cublasStatus_t cublasStatus = cublasCreate(&handle);

    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        printf("cublasCreate failed\n");

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        free(h_A);
        free(h_B);

        return 1;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    /*
       cuBLAS 默认使用列主序。

       原始代码使用行主序计算：

           C = A * B

       将行主序数组交给 cuBLAS 后，相当于：

           C^T = B^T * A^T

       因此这里传入 d_B、d_A，而不是 d_A、d_B。
    */

    // cuBLAS 预热
    cublasStatus = cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        N,
        N,
        &alpha,
        d_B,
        N,
        d_A,
        N,
        &beta,
        d_C,
        N);

    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        printf("Warm-up cublasSgemm failed\n");
        cublasDestroy(handle);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        free(h_A);
        free(h_B);

        return 1;
    }

    cudaDeviceSynchronize();

    // cuBLAS 正式计时
    cudaEventRecord(start);

    for (int i = 0; i < iterations; i++) {
        cublasStatus = cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            N,
            N,
            &alpha,
            d_B,
            N,
            d_A,
            N,
            &beta,
            d_C,
            N);

        if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
            printf("cublasSgemm failed at iteration %d\n", i);
            break;
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);

    avg_ms = ms / iterations;

    gflops =
        (2.0 * N * N * N) /
        (avg_ms * 1000000.0);

    printf(
        "cuBLAS SGEMM -> Time: %8.4f ms | Performance: %6.2f GFLOPS\n",
        avg_ms,
        gflops);

    // 释放 cuBLAS 句柄
    cublasDestroy(handle);

    // 释放 CUDA Event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 释放 GPU 显存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 释放 CPU 内存
    free(h_A);
    free(h_B);

    return 0;
}