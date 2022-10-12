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
        printf("dev_my_struct->arr[%d] = %f\n", i, dev_my_struct->array[i]);
    }

    printf("----- cuda_struct_print END -----\n");
}

int main() {
    int N = 10;

    my_struct* ram_my_struct = (my_struct*)malloc(sizeof(my_struct));    
    ram_my_struct->size = N;
    float* arr = (float*)malloc(N * sizeof(float));
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
    cudaMemcpy(dev_my_struct_array, ram_my_struct->array, ram_my_struct->size * sizeof(float), cudaMemcpyHostToDevice);

    my_struct* my_struct_dto = (my_struct*)malloc(sizeof(my_struct));
    my_struct_dto->size = ram_my_struct->size;
    my_struct_dto->array = dev_my_struct_array;

    cudaMemcpy(dev_my_struct, my_struct_dto, sizeof(my_struct), cudaMemcpyHostToDevice);

    cuda_struct_print<<<1,1>>>(dev_my_struct);

    my_struct* my_struct_dto2 = (my_struct*)malloc(sizeof(my_struct));
    cudaMemcpy(my_struct_dto2, dev_my_struct, sizeof(my_struct), cudaMemcpyDeviceToHost);
    float* dataFromGPU = (float*)malloc(ram_my_struct->size * sizeof(float));;
    
    printf("\tmy_struct_dto2->size = %d\n", my_struct_dto2->size);
    cudaMemcpy(dataFromGPU, my_struct_dto2->array, my_struct_dto2->size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\t70 my_struct_dto2->size = %d\n", my_struct_dto2->size);
    for(int i = 0; i < N; i++)
    {        
        printf("my_struct_dto2->arr[%d] = %f\n", i, dataFromGPU[i]);
    }
    
    return 0;
}