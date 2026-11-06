#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // 获取 0 号 GPU（你的 RTX 3050）的属性

    printf("--- GPU Device Query ---\n");
    printf("Device Name: %s\n", prop.name);
    printf("CUDA Compute Capability: %d.%d\n", prop.major, prop.minor);

    printf("\n--- 执行模型 ---\n");
    printf("Streaming Multiprocessors (SM): %d\n", prop.multiProcessorCount); // 有多少个 SM
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);           // 一个 Block 最多多少线程
    printf("Warp Size: %d\n", prop.warpSize);                                 // 线程束大小（通常是 32）

    printf("\n--- 内存层级 ---\n");
    printf("Global Memory (VRAM): %.2f GB\n", (float)prop.totalGlobalMem / (1024 * 1024 * 1024));
    printf("Shared Memory per Block: %.2f KB\n", (float)prop.sharedMemPerBlock / 1024);
    printf("L2 Cache Size: %.2f MB\n", (float)prop.l2CacheSize / (1024 * 1024));
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);

    // 新的代码（使用 cudaDeviceGetAttribute）：
    int memoryClockRateKHz = 0;
    cudaDeviceGetAttribute(&memoryClockRateKHz, cudaDevAttrMemoryClockRate, 0); // 0 是设备编号

    // 然后使用 memoryClockRateKHz 替代 prop.memoryClockRate
    float bandwidth = (memoryClockRateKHz * 1000.0f * prop.memoryBusWidth * 2) / (8 * 1000000000.0f);
    printf("Theoretical Memory Bandwidth: %.2f GB/s\n", bandwidth);

    return 0;
}