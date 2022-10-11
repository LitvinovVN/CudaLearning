// Задача 004.
// Создать массив из десяти чисел типа float.
// Скопировать значения элементов массива из ОЗУ в GPU.
// Передать в CUDA-ядро массив из десяти чисел типа float.
// Увеличить значения кождого из элементов массива на 1.5.
// Скопировать значения элементов массива из GPU в ОЗУ.
// Вывести значения элементов массива в консоль.
// Запуск:
// nvcc 004.cu
// ./a

#include <iostream>

__global__ void cuda_array_add(float* dev_arrFloat, int size){ 
    printf("----- cuda_array_add START -----\n");
    for(int i = 0; i < size; i++)
    {
        printf("i = %d: %f + 1.5 = ", i, dev_arrFloat[i]);
        dev_arrFloat[i] += 1.5;
        printf("%f\n", dev_arrFloat[i]);
    }        
    printf("----- cuda_array_add END -----\n");
}

int main() {
    int N = 10;

    float* arrFloat = (float*)malloc(N * sizeof(float));
    for(int i = 0; i < N; i++)
    {
        arrFloat[i] = i;
        printf("arrFloat[%i] = %f\n", i, arrFloat[i]);
    }

    float* dev_arrFloat;
    cudaMalloc((void**)&dev_arrFloat, N * sizeof(float));
    cudaMemcpy(dev_arrFloat, arrFloat, N * sizeof(float), cudaMemcpyHostToDevice);

    cuda_array_add<<<1,1>>>(dev_arrFloat, N);

    cudaError_t err1 = cudaMemcpy(arrFloat, dev_arrFloat, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf(cudaGetErrorString (err1));

    printf("\n-----------------------\n");
    for(int i = 0; i < N; i++)
    {        
        printf("arrFloat[%i] = %f\n", i, arrFloat[i]);
    }

    return 0;
}