#include <iostream>

__global__ void cuda_threadIdx(float* A, float* B, float* C, int size){
    printf("Hello World from GPU!\n");
    printf("A[2]=%g\n", A[2]);
    int i = 2;// threadIdx.x;
    //C[i] = A[i] + B[i]; 
    A[2] = 10;
    printf("A[2]=%g\n", A[2]);
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
    for(int i=0;i<N;i++)
    {
        a[i] = i;
        b[i] = -i;
        c[i] = 0.01*i;
        printf("%g %g %g \n", a[i], b[i], c[i]);
    }
    
    cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaThreadSynchronize();

    int numBlocks = 3;
    dim3 threadsPerBlock(N);
    //cuda_threadIdx<<<numBlocks,threadsPerBlock>>>(dev_a, dev_b, dev_c, N);
    cuda_threadIdx<<<1,1>>>(dev_a, dev_b, dev_c, N);
    
    cudaError_t err1 = cudaMemcpy(a, dev_a, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf(cudaGetErrorString (err1));
    cudaMemcpy(b, dev_b, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(c, dev_c, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    printf("\nRAM massives after CUDA kernel\n");
    for(int i=0;i<N;i++)
    {
        a[i] = i;
        b[i] = -i;
        c[i] = 0.01*i;
        printf("%g %g %g \n", a[i], b[i], c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}