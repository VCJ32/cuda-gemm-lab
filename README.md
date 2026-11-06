# cuda-gemm-lab

## 📌 项目简介
本项目从零开始，逐步记录从基础 CUDA 编程到 SGEMM（单精度通用矩阵乘法）深度优化、性能建模（Roofline）与 Profiling 的全过程，注重**用数据找瓶颈，用架构原理解释性能差异**。

## 💻 实验环境
- **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop (Ampere, Compute Capability 8.6)
- **硬件参数**: 20 SMs, 4GB VRAM, Warp Size: 32, 理论显存带宽: 187.87 GB/s
- **软件环境**: Windows, VS 2022 (Release Mode), CUDA 12.4

## 📊 实验记录与进度

### [Phase 0] Day 01-02: 环境配置与设备摸底
- **Vector Add**: 实现基础向量加法，初步体会 Host 与 Device 之间的数据拷贝开销。
- **Device Query**: 编写硬件属性查询程序，明确 RTX 3050 Ti 的 SM 数量、Warp Size (32) 及显存带宽天花板（后续判定算子是否 Memory Bound 的基准线）。
- **执行模型**: 绘制并理解 Grid -> Block -> Thread -> Warp 的层级关系。

### [Phase 0] Day 03: SIMT 与内存合并访问 (Memory Coalescing)
**实验目的**：验证 Warp 调度机制与合并访存对带宽的巨大影响。
**测试对象**：5000万个元素的 Vector Add。

| Block Size | 耗时 (ms) | 现象与架构分析 |
| :--- | :--- | :--- |
| 32 | 16.87 | 仅1个Warp，并发极低，无法掩盖延迟 |
| 128 | 5.75 | **甜点区**，占用率与并发达到平衡 |
| 256 | 5.76 | **甜点区**，最常用的通用配置 |
| 1024 | 8.74 | Block过大导致 Occupancy（占用率）下降 |
| 100 | 7.34 | 非 32 倍数，产生残缺的 Warp，浪费硬件资源 |

**合并访存 vs 跨步访问**：
- 合并访存耗时：**5.76 ms**
- 跨步访问耗时：**18.90 ms**（慢了 3.3 倍）
- **结论**：连续访问可将 32 个请求合并为 1 个 128 字节的连续事务；跨步访问彻底击穿显存带宽，导致性能暴跌。

### [Phase 0] Day 04: 共享内存 (Shared Memory) 与矩阵转置
**实验目的**：利用 Shared Memory 消除全局内存的“写非合并”瓶颈。
**测试对象**：2048 x 2048 的矩阵转置。

| 算法版本 | 耗时 (ms) | 有效带宽 (GB/s) | 核心逻辑 |
| :--- | :--- | :--- | :--- |
| Naive Transpose | 0.6726 | 49.89 | 读操作连续，写操作跨步，带宽被严重浪费 |
| **Shared Transpose** | **0.2264** | **148.20** | 读入时合并，经 `__syncthreads()` 后，从共享内存中读出并合并写回 |

- **加速比**：**约 3 倍**
- **带宽利用率**：148.20 / 187.87 ≈ **78.8%**，逼近该硬件的理论上限。
- **思考**：通过 Tiling 技术（分块），将全局内存的数据暂存到片上极快的 Shared Memory 中，把“读”和“写”全部转换为合并访存。下一步需解决 Shared Memory 内部的 **Bank Conflict（银行冲突）**（代码中已使用 `tile[32][33]` 进行初步规避）。

## 🎯 后续规划 (Roadmap)
- [ ] **Day 05**: Bank Conflict（银行冲突）原理与 Padding 技巧验证。
- [ ] **Day 06-10**: SGEMM 优化四连：Naive -> Shared Memory Tiling -> 向量化/寄存器分块 -> 挑战 cuBLAS。
- [ ] **Phase 2**: Roofline 模型建立与 Nsight Compute 性能瓶颈定位。
- [ ] **Phase 3**: YOLOv8 推理热点分析与 TensorRT 低精度部署对比。