#include <iostream>

__global__ void cuda_threadIdx(float* A, int size){
    printf("\n--- Start CUDA kernel cuda_threadIdx ---\n");
    printf("A[2]=%g\n", A[2]);    
    A[2] = 10;
    printf("A[2]=%g\n", A[2]);
    printf("--- End CUDA kernel cuda_threadIdx ---\n");
}

void printFloatArray(float* A, int size)
{
    for(int i=0; i<size;i++)
        printf("%g ", A[i]);
}

int main() {
    // Количество элементов массива
    int N = 10;
    printf("N = %d elements\n", N);

    // Размер массива, байт
    int dataSize = N * sizeof(float);
    printf("dataSize = %d bytes\n", dataSize);
    
    // Выделение памяти в ОЗУ
    float *a = (float*)malloc(dataSize);    

    // Инициализация массива a
    printf("RAM massives initialization\n");
    for(int i=0;i<N;i++)
    {
        a[i] = i;        
    }
    printFloatArray(a, N);

    // Выделение памяти в GPU
    float *dev_a;
    cudaMalloc((void**)&dev_a, dataSize);
    
    // Копирование 
    cudaMemcpy(dev_a, a, dataSize, cudaMemcpyHostToDevice);    
    
    // Вызов CUDA-ядра cuda_threadIdx
    cuda_threadIdx<<<1,1>>>(dev_a, N);
    
    // Копирование данных из видеопамяти в ОЗУ
    cudaError_t err1 = cudaMemcpy(a, dev_a, dataSize, cudaMemcpyDeviceToHost);
    printf(cudaGetErrorString (err1));

    // Вывод на экран результатов
    printf("\nRAM massive after CUDA kernel\n");
    printFloatArray(a, N);

    // Очистка видеопамяти
    cudaFree(dev_a);
    // Очистка ОЗУ
    free(a);

    return 0;
}