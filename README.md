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


### [Phase 0] Day 05: 共享内存 Bank Conflict 及其解决
**实验目的**：验证 Bank Conflict 的物理机制以及 Padding 技术的优化效果。
**测试对象**：2048 x 2048 的矩阵转置（使用 Shared Memory）。

| 算法版本 | 耗时 (ms) | 相对性能 |
| :--- | :--- | :--- |
| 无 Padding (`tile[32][32]`) | 0.3455 | 基准（存在严重 Bank Conflict）|
| **有 Padding (`tile[32][33]`)** | **0.2255** | **提升约 53%** |

- **架构原理**：Shared Memory 有 32 个 Bank，无 Padding 时按列读取会导致 32 个线程全部命中同一个 Bank，引发严重的串行化等待。增加 1 列填充后，行宽变为 33（与 32 互质），完美错开 Bank 索引，实现无冲突并发访问。

## 🎯 阶段 0 总结与验收 (Day 01 - Day 05)
至此，阶段 0（计算机系统与 GPU 基础认知）圆满完成！
- [x] 掌握 GPU 硬件参数查询与理论带宽计算。
- [x] 深刻理解 Grid-Block-Thread-Warp 执行模型与 SIMT 机制。
- [x] 能通过实验验证 Memory Coalescing 对全局内存带宽的决定性影响。
- [x] 掌握 Shared Memory 的使用，并能利用 Padding 技术消除 Bank Conflict。

### [Phase 1] Day 06: Naive GEMM 与 cuBLAS 性能摸底
**实验目的**：实现最基础的矩阵乘法（C = A * B），并确立性能优化的基准线。
**测试对象**：1024 x 1024 的单精度（FP32）矩阵乘法。

| 算法版本 | 耗时 (ms) | 算力 (GFLOPS) | 硬件利用率 | 核心瓶颈分析 |
| :--- | :--- | :--- | :--- | :--- |
| Naive GEMM | 4.6077 | 466.06 | < 8% | 无数据复用，重复全局内存访存，Memory Bound |
| **cuBLAS SGEMM** | **0.3706** | **5794.04** | **~100%** | 深度调优：Tiling + 寄存器分块 + 向量化 + 流水线 |

- **架构洞察**：RTX 3050 Ti 的 FP32 理论极限约为 6 TFLOPS。cuBLAS 已贴紧物理极限，而 Naive 版本仅发挥了不到 10% 的算力。这 **12.4 倍** 的差距，正是 AI 架构与性能优化岗位的核心价值所在。
- **优化 Roadmap**：
  - [ ] Step 1: 引入 Shared Memory Tiling，消除全局内存重复访问。（预计提升 2-3 倍）
  - [ ] Step 2: 寄存器分块 (Register Blocking)，减少 SMEM 访问频次。
  - [ ] Step 3: 向量化访存与 Double Buffering 流水线。

## 🎯 后续规划 (Roadmap)
... (原有内容保留)