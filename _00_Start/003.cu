// Задача 003. Передать в CUDA-ядро число типа float.
// Увеличить его на 1.5. Скопировать измененное значение из GPU в ОЗУ.
// Вывести измененное значение переменной в консоль.
// Запуск:
// nvcc 003.cu
// ./a

#include <iostream>

__global__ void cuda_hello(float* dev_varFloat){    
    printf("cuda_hello START: varFloat = %f\n", *dev_varFloat);
    *dev_varFloat += 1.5;
    printf("cuda_hello END: varFloat = %f\n", *dev_varFloat);
}

int main() {
    float varFloat = -43.0123456789;
    printf("main START: varFloat = %f\n", varFloat);

    float* dev_varFloat;
    cudaMalloc((void**)&dev_varFloat, sizeof(float));
    cudaMemcpy(dev_varFloat, &varFloat, sizeof(float), cudaMemcpyHostToDevice);

    cuda_hello<<<1,1>>>(dev_varFloat);
    cudaMemcpy(&varFloat, dev_varFloat, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("main END: varFloat = %f\n", varFloat);

    return 0;
}