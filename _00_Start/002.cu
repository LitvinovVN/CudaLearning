// Задача 002. Передать в CUDA-ядро значения различных численных типов и вывести их из CUDA-ядра.
// Обратить внимание на погрешность вывода вещественных чисел!
// Запуск:
// nvcc 002.cu
// ./a

#include <iostream>

__global__ void cuda_hello(bool varBool, int varInt, long varLong, float varFloat, double varDouble){
    printf("varBool = %d\n", varBool);        
    printf("varBool = %s\n", varBool ? "true" : "false");
    printf("varInt = %d\n", varInt); 
    printf("varLong = %ld\n", varLong);
    printf("varFloat = %f\n", varFloat); 
    printf("varFloat = %.4f\n", varFloat);
    printf("varDouble = %f\n", varDouble); 
    printf("varDouble = %g\n", varDouble);
}

int main() {
    bool varBool = true;
    int varInt = 15;
    long varLong = 1234567890;
    float varFloat = -43.0123456789;
    double varDouble = -43.0123456789;

    cuda_hello<<<1,1>>>(varBool, varInt, varLong, varFloat, varDouble); 

    return 0;
}