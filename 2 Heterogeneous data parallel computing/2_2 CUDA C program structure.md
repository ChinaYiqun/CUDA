# 2.2 CUDA C程序结构

## 2.2.1 主机代码与设备代码

- **主机代码（Host Code）**&zwnj;：运行在CPU上，可以使用标准C语言编写。
- **设备代码（Device Code）**&zwnj;：运行在GPU上，需要使用CUDA C的特定语法。

## 2.2.2 CUDA程序的执行流程

1. **启动**：程序开始执行时，控制权在CPU（主机）上。
2. **调用内核**：主机代码调用一个内核函数（Kernel），这个调用会触发GPU上的并行计算。
3. **执行内核**：内核函数在GPU上以多个线程（Thread）的形式并行执行。
4. **同步**：所有内核执行完成后，控制权返回给CPU。

## 2.2.3 内核函数

- **内核函数**：是设备代码的核心，可以被看作是GPU上的一个并行任务。
- **线程**：内核函数的执行单元，每个线程执行相同的代码，但可以处理不同的数据。
- **线程块**：一组线程的集合，它们可以共享某些数据（例如，共享内存）。

## 2.2.4 分配内存

- **全局内存**：GPU上的主要存储区域，主机和设备都可以访问。
- **共享内存**：仅由同一个线程块内的线程共享。
- **常量内存和纹理内存**：特殊类型的内存，用于优化数据访问模式。

## 2.2.5 数据传输

- **从主机到设备**：将数据从CPU内存复制到GPU的全局内存中。
- **从设备到主机**：将数据从GPU的全局内存复制回CPU内存。

## 2.2.6 编译CUDA程序

- **nvcc编译器**：NVIDIA提供的编译器，用于将CUDA C代码编译成可以在GPU上执行的机器代码。

## 2.2.7 示例：向量加法

这一节通常会通过一个简单的向量加法示例来展示CUDA C程序的结构。

- **主机代码**：分配内存，初始化数据，调用内核函数。
- **设备代码**：定义一个内核函数，用于执行实际的向量加法。

## 2.2.8 错误检查和调试

- **错误检查**：对CUDA API调用进行错误检查，确保程序的健壮性。
- **调试**：使用CUDA提供的工具来调试设备代码。