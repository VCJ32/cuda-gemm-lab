#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <chrono> // 用于CPU计时

// 这是运行在 GPU 上的 Kernel（核函数）
__global__ void addKernel(float* c, const float* a, const float* b, int size)
{
    // 计算当前线程的全局索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 防止越界
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

// CPU 版本的加法，用于对比
void addCPU(float* c, const float* a, const float* b, int size)
{
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    const int arraySize = 10000000; // 1千万个元素
    const int bytes = arraySize * sizeof(float);

    // 1. 分配 Host (CPU) 内存
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c_cpu = (float*)malloc(bytes);
    float* h_c_gpu = (float*)malloc(bytes);

    // 初始化数据
    for (int i = 0; i < arraySize; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // ================= 测试 CPU 耗时 =================
    auto start_cpu = std::chrono::high_resolution_clock::now();
    addCPU(h_c_cpu, h_a, h_b, arraySize);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    printf("CPU Time: %f ms\n", cpu_time.count());

    // ================= 测试 GPU 耗时 =================
    float* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    // 把数据从 CPU 拷贝到 GPU
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // 配置线程网络：每个 Block 256 个线程，计算需要多少个 Block
    int threadsPerBlock = 256;
    int blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock;

    // 使用 CUDA Event 进行精确计时
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu); // 开始计时
    // 启动 Kernel
    addKernel << <blocksPerGrid, threadsPerBlock >> > (d_c, d_a, d_b, arraySize);
    cudaEventRecord(stop_gpu); // 结束计时
    cudaEventSynchronize(stop_gpu); // 等待 GPU 算完

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    printf("GPU Time: %f ms\n", gpu_time);

    // 把结果拷贝回 CPU 验证
    cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost);

    // 简单验证一下结果对不对（抽查第一个和最后一个元素）
    printf("Check result: h_c_gpu[0] = %f, h_c_gpu[last] = %f\n", h_c_gpu[0], h_c_gpu[arraySize - 1]);

    // 释放内存
    free(h_a); free(h_b); free(h_c_cpu); free(h_c_gpu);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(start_gpu); cudaEventDestroy(stop_gpu);

    return 0;
}