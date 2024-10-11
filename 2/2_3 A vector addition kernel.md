### 2.3 向量加法内核


#### 2.3.1 内核函数的概念

在CUDA中，内核函数是一类特殊的函数，它的设计目的是在GPU上并行执行。当内核函数被调用时，它会在GPU上启动多个线程来执行相同的代码。这些线程可以并行处理数据，从而实现数据的并行处理。

#### 2.3.2 向量加法的CUDA实现

向量加法的目的是实现两个向量的元素逐个相加。假设有两个浮点数向量`A`和`B`，它们具有相同的长度`N`，我们希望得到它们的和`C`，其中`C[i] = A[i] + B[i]`。

#### 2.3.3 编写内核函数

在CUDA中，编写一个向量加法的内核函数需要遵循以下步骤：

1. **定义内核函数**：使用`__global__`关键字来定义一个内核函数。这个函数会被并行地在多个线程上执行。

```cuda
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}
```

2. **计算线程索引**：内核函数中的每个线程需要计算它所负责处理的数据元素的索引。这通常通过结合线程的ID（`threadIdx.x`）、块的ID（`blockIdx.x`）以及块的尺寸（`blockDim.x`）来实现。

3. **访问和处理数据**：线程使用它的索引来访问全局内存中的输入向量`A`和`B`，并计算结果向量`C`。

4. **边界检查**：为了避免线程访问超出向量长度的元素，内核函数中通常包含一个边界检查。

#### 2.3.4 调用内核函数

内核函数的调用需要指定执行的配置参数，包括网格的维度（即块的数量）和每个块的线程数。例如：

```cuda
int numElements = 1024;
size_t size = numElements * sizeof(float);
float *d_A = NULL;
float *d_B = NULL;
float *d_C = NULL;

// 分配设备内存
cudaMalloc((void **)&d_A, size);
cudaMalloc((void **)&d_B, size);
cudaMalloc((void **)&d_C, size);

// 将向量从主机复制到设备
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

// 启动内核函数
vectorAdd<<<(numElements + 255) / 256, 256>>>(d_A, d_B, d_C, numElements);

// 将结果从设备复制回主机
cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
```

这里的`<<<(numElements + 255) / 256, 256>>>`是CUDA中的执行配置，表示我们希望以256个线程为一个块来执行内核函数，并且根据向量的长度计算需要多少个块。

#### 2.3.5 总结

通过向量加法的例子，我们可以了解到如何在CUDA中编写和执行内核函数。这个过程涉及到定义内核函数、计算线程索引、处理数据以及正确地调用内核函数。向量加法是并行计算中的一个基础案例，理解了这个过程，就可以将其扩展到更复杂的并行计算任务中。
