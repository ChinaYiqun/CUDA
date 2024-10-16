#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 定义一个内核函数，用于在GPU上执行并行计算
__global__ void sum(float *x) {
    // 获取当前block的ID，即在所有block中的编号
    int block_id = blockIdx.x;
    // 获取当前线程的全局ID，即在所有线程中的编号
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 获取当前线程在其所属block内的局部ID
    int local_tid = threadIdx.x;
    // 打印当前线程的信息
    printf("current block=%d, thread id in current block =%d, global thread id=%d\n", block_id, local_tid, global_tid);
    // 将全局ID作为索引，对数组x的对应元素加1
    x[global_tid] += 1;
}

int main() {
    int N = 32; // 定义数组的大小
    int nbytes = N * sizeof(float); // 计算需要分配的内存大小
    float *dx, *hx; // 定义指向GPU和CPU内存的指针

    // 分配GPU内存
    cudaMalloc((void **)&dx, nbytes); // 使用二级指针是因为cudaMalloc需要一个void类型的指针
    // 分配CPU内存
    hx = (float*) malloc(nbytes); // 分配与GPU同等大小的内存空间

    // 初始化CPU上的数据
    printf("hx original: \n");
    for (int i = 0; i < N; i++) {
        hx[i] = i; // 将数组初始化为0到N-1的序列
        printf("%g\n", hx[i]); // 打印原始数据
    }

    // 将CPU的数据复制到GPU
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice); // 从主机到设备的内存复制

    // 启动GPU内核函数
    sum<<<1, N>>>(dx); // 启动内核，1个block，N个线程

    // 将数据从GPU复制回CPU
    cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost); // 从设备到主机的内存复制

    // 打印修改后的数据
    printf("hx current: \n");
    for (int i = 0; i < N; i++) {
        printf("%g\n", hx[i]); // 打印经过GPU计算后的数据
    }

    // 释放GPU和CPU的内存
    cudaFree(dx);
    free(hx);

    return 0;
}