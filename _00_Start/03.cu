#include <iostream>

__global__ void cuda_threadIdx(float* A, float* B, float* C, int size){    
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main() {
    // Размерность массива
    int N = 10;

    // Выделение памяти в ОЗУ
    float *a = (float*)malloc(N*sizeof(float));
    float *b = (float*)malloc(N*sizeof(float));
    float *c = (float*)malloc(N*sizeof(float));

    // Выделение памяти в GPU
    float *dev_a, *dev_b, *dev_c;
    cudaMalloc((void**)&dev_a, N*sizeof(float));
    cudaMalloc((void**)&dev_b, N*sizeof(float));
    cudaMalloc((void**)&dev_c, N*sizeof(float));

    printf("RAM massives initialization\n");
    printf("a\tb\tc\n");
    for(int i=0;i<N;i++)
    {
        a[i] = i;
        b[i] = 0.2*i;
        c[i] = 0;
        printf("%g\t%g\t%g\n", a[i], b[i], c[i]);
    }
    
    cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, N*sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = 1;
    dim3 threadsPerBlock(N);

    cuda_threadIdx<<<numBlocks,threadsPerBlock>>>(dev_a, dev_b, dev_c, N);
        
    cudaError_t err1 = cudaMemcpy(a, dev_a, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf(cudaGetErrorString (err1));
    cudaMemcpy(b, dev_b, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(c, dev_c, N*sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nRAM massives after CUDA kernel\n");
    printf("----------------------\n");
    printf("a\tb\tc\n");
    printf("----------------------\n");
    for(int i=0;i<N;i++)
    {        
        printf("%g\t%g\t%g\n", a[i], b[i], c[i]);
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