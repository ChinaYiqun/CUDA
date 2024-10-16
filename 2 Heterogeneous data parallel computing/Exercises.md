## 练习题

#### 1. 如果我们想使用网格中的每个线程来计算向量加法的一个输出元素，那么将线程/块索引映射到数据索引(i)的表达式是什么？

- (A) `i=threadIdx.x + threadIdx.y;`
- (B) `i=blockIdx.x + threadIdx.x;`
- (C) `i=blockIdx.x*blockDim.x + threadIdx.x;`
- (D) `i=blockIdx.x * threadIdx.x;`
- [C]

#### 2. 假设我们想让每个线程计算向量加法中相邻的两个元素。那么将线程/块索引映射到线程要处理的第一个元素的数据索引(i)的表达式是什么？

- (A) `i=blockIdx.x*blockDim.x + threadIdx.x + 2;`
- (B) `i=blockIdx.x*threadIdx.x * 2;`
- (C) `i=(blockIdx.x*blockDim.x + threadIdx.x) * 2;`
- (D) `i=blockIdx.x*blockDim.x * 2 + threadIdx.x;`
  
-  [C] 详细见3_2 章节
#### 3. 我们想让每个线程计算向量加法中的两个元素。每个线程块处理2*blockDim.x个连续的元素，形成两部分。每个块中的所有线程首先处理一部分，每个线程处理一个元素。然后它们都会移动到下一部分，每个线程处理一个元素。假设变量i应该是线程要处理的第一个元素的索引。那么将线程/块索引映射到第一个元素的数据索引的表达式是什么？

- (A) `i=blockIdx.x*blockDim.x + threadIdx.x + 2;`
- (B) `i=blockIdx.x*threadIdx.x * 2;`
- (C) `i=(blockIdx.x*blockDim.x + threadIdx.x) * 2;`
- (D) `i=blockIdx.x*blockDim.x * 2 + threadIdx.x;`

#### 4. 对于一个向量加法，假设向量长度为8000，每个线程计算一个输出元素，线程块大小为1024个线程。程序员配置内核调用，以确保覆盖所有输出元素的最少线程块数量。网格中将有多少个线程？

- (A) 8000
- (B) 8196
- (C) 8192
- (D) 8200

#### 5. 如果我们想在CUDA设备全局内存中分配一个包含v个整数元素的数组，那么cudaMalloc调用的第二个参数的适当表达式是什么？

- (A) n
- (B) v
- (C) n * sizeof(int)
- (D) v * sizeof(int)
D

#### 6. 如果我们想分配一个包含n个浮点元素的数组，并且有一个指向已分配内存的浮点指针变量A_d，那么cudaMalloc()调用的第一个参数的适当表达式是什么？

- (A) n
- (B) `(void *) A_d`
- (C) A_d
- (D) `(void *) &A_d`
- cudaMalloc((void **)&A_d, size)

#### 7. 如果我们想从主机数组A_h（A_h是指向源数组元素0的指针）复制3000字节的数据到设备数组A_d（A_d是指向目标数组元素0的指针），那么在CUDA中进行此数据复制的适当API调用是什么？

- (A) `cudaMemcpy(3000, A_h, A_d, cudaMemcpyHostToDevice);`
- (B) `cudaMemcpy(A_h, A_d, 3000, cudaMemcpyDeviceToHost);`
- (C) `cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);`
- (D) `cudaMemcpy(3000, A_d, A_h, cudaMemcpyHostToDevice);`
- [C]
- 
#### 8. 如何声明一个可以适当接收CUDA API调用返回值的变量err？

- (A) `int err;`
- (B) `cudaError err;`
- (C) `cudaError_t err;`
- (D) `cudaSuccess_t err;`