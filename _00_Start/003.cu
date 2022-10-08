// Задача 003. Передать в CUDA-ядро число типа float. Увеличить его на 1.5.
// Запуск:
// nvcc 002.cu
// ./a

#include <iostream>

__global__ void cuda_hello(float* dev_varFloat){    
    printf("cuda_hello START: varFloat = %f\n", dev_varFloat);
    dev_varFloat += 1.5;
    printf("cuda_hello END: varFloat = %f\n", dev_varFloat);
}

int main() {
    float varFloat = -43.0123456789;
    printf("main START: varFloat = %f\n", varFloat);

    float* dev_varFloat;

    cuda_hello<<<1,1>>>(dev_varFloat);
    cudaDeviceSynchronize();// Ожидание основным потоком выполнения функции cuda_hello
    
    printf("main END: varFloat = %f\n", varFloat);

    return 0;
}