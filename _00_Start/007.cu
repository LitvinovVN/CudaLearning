// Задача 007.
// Создать структуру, содержащую массив типа float и число элементов массива типа int.
// Скопировать структуру из ОЗУ в GPU.
// Передать в CUDA-ядро структуру.
// В CUDA-ядре изменяем значения элементов массива структуры
// Скопировать структуру из GPU в ОЗУ.
// Вывести значения элементов структуры в консоль.
// Запуск:
// nvcc 007.cu
// ./a

#include <iostream>

typedef struct 
{
    float* array;
    int size;    
} my_struct;

__global__ void cuda_struct_print(my_struct* dev_my_struct){ 
    printf("----- cuda_struct_print START -----\n");
    printf("dev_my_struct->size = %d\n", dev_my_struct->size);

    for(int i = 0; i < dev_my_struct->size; i++)
    {
        dev_my_struct->array[i] += 10;
        printf("ram_my_struct->arr[%d] = %f\n", i, dev_my_struct->array[i]);
    }

    printf("----- cuda_struct_print END -----\n");
}

int main() {
    int N = 10;

    my_struct* ram_my_struct = (my_struct*)malloc(sizeof(my_struct));    
    ram_my_struct->size = N;
    float* arr = (float*)malloc(N * sizeof(int));
    ram_my_struct->array = arr;

    for(int i = 0; i < N; i++)
    {
        ram_my_struct->array[i] = i;
        printf("ram_my_struct->arr[%d] = %f\n", i, ram_my_struct->array[i]);
    }

    my_struct* dev_my_struct;
    cudaMalloc((void**)&dev_my_struct, sizeof(my_struct));
    float* dev_my_struct_array;
    cudaMalloc((void**)&dev_my_struct_array, ram_my_struct->size * sizeof(float));
    dev_my_struct->array = dev_my_struct_array;
    cudaMemcpy(dev_my_struct, ram_my_struct, sizeof(my_struct), cudaMemcpyHostToDevice);

    cuda_struct_print<<<1,1>>>(dev_my_struct);

    cudaMemcpy(ram_my_struct, dev_my_struct, sizeof(my_struct), cudaMemcpyDeviceToHost);
    
    
    return 0;
}