// Задача 006.
// Создать структуру, содержащую три числа типов bool, int, float.
// Скопировать структуру из ОЗУ в GPU.
// Передать в CUDA-ядро структуру.
// В CUDA-ядре меняем значения полей структуры
// Скопировать структуру из GPU в ОЗУ.
// Вывести значения элементов структуры в консоль.
// Запуск:
// nvcc 006.cu
// ./a

#include <iostream>

typedef struct 
{
    bool varBool;
    int varInt;
    float varFloat;
} my_struct;

__global__ void cuda_struct_print(my_struct* dev_my_struct){ 
    printf("----- cuda_struct_print START -----\n");
    dev_my_struct->varBool = true;
    dev_my_struct->varInt += 10;
    dev_my_struct->varFloat += 0.5;
    printf("dev_my_struct->varInt = %d\n", dev_my_struct->varBool);
    printf("dev_my_struct->varInt = %d\n", dev_my_struct->varInt);
    printf("dev_my_struct->varInt = %f\n", dev_my_struct->varFloat);
    printf("----- cuda_struct_print END -----\n");
}

int main() {    
    my_struct* ram_my_struct = (my_struct*)malloc(sizeof(my_struct));
    ram_my_struct->varBool = false;
    ram_my_struct->varInt = 0;
    ram_my_struct->varFloat = 0;

    my_struct* dev_my_struct;
    cudaMalloc((void**)&dev_my_struct, sizeof(my_struct));
    cudaMemcpy(dev_my_struct, ram_my_struct, sizeof(my_struct), cudaMemcpyHostToDevice);

    cuda_struct_print<<<1,1>>>(dev_my_struct);

    cudaMemcpy(ram_my_struct, dev_my_struct, sizeof(my_struct), cudaMemcpyDeviceToHost);
    printf("ram_my_struct->varInt = %d\n", ram_my_struct->varBool);
    printf("ram_my_struct->varInt = %d\n", ram_my_struct->varInt);
    printf("ram_my_struct->varInt = %f\n", ram_my_struct->varFloat);
    
    return 0;
}