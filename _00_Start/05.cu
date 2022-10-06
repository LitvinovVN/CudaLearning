#include <iostream>

__device__ int getGlobalIndex()
{
    //Индекс текущего блока в гриде
    int blockIndex = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.y*gridDim.x;
    //Индекс треда внутри текущего блока
    int threadIndex = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;

    //глобальный индекс нити
    int i = blockIndex*blockDim.x*blockDim.y*blockDim.z + threadIndex; 

    return i;
}

__global__ void cuda_threadIdx(float* A, float* B, float* C, int size){
    int i = getGlobalIndex();
    
    C[i] = A[i] + B[i];
}

int main() {
    // Размерность массива
    int N = 10;

    // Размер массива, байт
    int dataSize = N * sizeof(float);
    printf("dataSize = %d bytes\n", dataSize);

    // Выделение памяти в ОЗУ
    float *a = (float*)malloc(dataSize);
    float *b = (float*)malloc(dataSize);
    float *c = (float*)malloc(dataSize);

    float **arrays = (float**)malloc(3*sizeof(float*));
    arrays[0]=a;
    arrays[1]=b;
    arrays[2]=c;
    printf("arrays[0]=%p\n", arrays[0]);
    printf("arrays[1]=%p\n", arrays[1]);
    printf("arrays[2]=%p\n", arrays[2]);

    printf("RAM massives initialization\n");
    printf("a\tb\tc\n");
    for(int i=0;i<N;i++)
    {
        //a[i] = i;
        //b[i] = 0.2*i;
        //c[i] = 0;
        arrays[0][i] = i;
        arrays[1][i] = 0.2*i;
        arrays[2][i] = 0;
        printf("%g\t%g\t%g\n", a[i], b[i], c[i]);
    }

    // Выделение памяти в GPU
    float *dev_a, *dev_b, *dev_c;
    cudaMalloc((void**)&dev_a, dataSize);
    cudaMalloc((void**)&dev_b, dataSize);
    cudaMalloc((void**)&dev_c, dataSize);
   
    
    cudaMemcpy(dev_a, a, dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, dataSize, cudaMemcpyHostToDevice);

    int numBlocks = 2;
    dim3 threadsPerBlock(N/numBlocks);

    cuda_threadIdx<<<numBlocks,threadsPerBlock>>>(dev_a, dev_b, dev_c, N);
        
    cudaError_t err1 = cudaMemcpy(a, dev_a, dataSize, cudaMemcpyDeviceToHost);
    printf(cudaGetErrorString (err1));
    cudaMemcpy(b, dev_b, dataSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(c, dev_c, dataSize, cudaMemcpyDeviceToHost);

    printf("\nRAM massives after CUDA kernel\n");
    printf("----------------------\n");
    printf("a\tb\tc\n");
    printf("----------------------\n");
    for(int i=0;i<N;i++)
    {        
        //printf("%g\t%g\t%g\n", a[i], b[i], c[i]);
        printf("%g\t%g\t%g\n", arrays[0][i], arrays[1][i], arrays[2][i]);
    }
    printf("----------------------\n");

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    free(a);
    free(b);
    free(c);

    return 0;
}