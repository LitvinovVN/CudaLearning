/*
Добавить функцию, создающую в GPU структуру Arrays1DList
для хранения массива экземпляров структур Array1D.
Добавить CUDA-ядро для вывода структуры Arrays1DList в консоль
*/

#include <iostream>

struct Array1D
{
    int* arr;
    int  len;
};


__host__ __device__
void print_1D_array(int* arr, int n)
{
    printf("[");
    for(int i = 0; i < n-1; i++)
        printf("%d, ", arr[i]);
    printf("%d", arr[n-1]);
    printf("]\n");
}

__host__ __device__
void Array1D_print(Array1D* arr1D)
{
    printf("Array1D.len = %d\n", arr1D->len);
    printf("Array1D.arr = ");
    print_1D_array(arr1D->arr, arr1D->len);
}

__host__ __device__
void Array1D_print(Array1D arr1D)
{
    Array1D_print(&arr1D);
}

// Kernel
__global__ void add_M_Kernel(Array1D arr1D, int M)
{
    printf("\n--- START __global__ void add_M_Kernel(Array1D arr1D, int M) START ---\n");
    Array1D_print(&arr1D);
    for(int i=0; i<arr1D.len; i++)
        arr1D.arr[i] += M;
    Array1D_print(&arr1D);
    printf("--- END __global__ void add_M_Kernel(Array1D arr1D, int M) END ---\n\n");
}

Array1D Array1D_create_RAM(int numElements)
{
    int* array = (int*)malloc(numElements * sizeof(int));
    for(int i = 0; i < numElements; i++)
        array[i] = i;    
    
    Array1D array1D;
    array1D.len = numElements;
    array1D.arr = array;
    
    return array1D;
}

Array1D Array1D_create_GPU(int numElements)
{
    int* array = (int*)malloc(numElements * sizeof(int));
    for(int i = 0; i < numElements; i++)
        array[i] = i;    

    Array1D array1D_GPU;
    array1D_GPU.len = numElements;
    cudaMalloc( &(array1D_GPU.arr), array1D_GPU.len * sizeof( array1D_GPU.arr ) );
    cudaMemcpy(array1D_GPU.arr, array, array1D_GPU.len * sizeof(array1D_GPU.arr), cudaMemcpyHostToDevice);

    free(array);

    return array1D_GPU;
}

Array1D Array1D_create_GPU(Array1D array1D_RAM)
{
    printf("START Array1D_create_GPU\n");
    Array1D array1D_GPU;
    array1D_GPU.len = array1D_RAM.len;

    cudaMalloc( &(array1D_GPU.arr), array1D_GPU.len * sizeof( array1D_GPU.arr ) );
    cudaMemcpy(array1D_GPU.arr, array1D_RAM.arr, array1D_GPU.len * sizeof(array1D_GPU.arr), cudaMemcpyHostToDevice);

    printf("END Array1D_create_GPU\n");
    return array1D_GPU;
}

Array1D Array1D_copy_GPU_to_RAM(Array1D array1D_GPU)
{
    Array1D array1D_RAM = Array1D_create_RAM(array1D_GPU.len);
    cudaMemcpy(array1D_RAM.arr, array1D_GPU.arr, array1D_GPU.len * sizeof(array1D_GPU.arr), cudaMemcpyDeviceToHost);
    return array1D_RAM;
}

/// @brief Очищает видеопамять
/// @param array1D_GPU 
void Array1D_free_GPU(Array1D array1D_GPU)
{
    cudaFree( array1D_GPU.arr );
}


///////////////// Структура Arrays1D ///////////////
struct Arrays1DList
{
    Array1D* arrays1D;
    int numElements;
};

// Создаёт экземпляр структуры Arrays1DList, содержащий массив из numArrays1D
// элементов структуры Array1D. Каждый элемент структуры Array1D
// содержит numElements
Arrays1DList Arrays1DList_create_RAM(int numArrays1D, int numElements)
{
    // Создаём в ОЗУ экземпляр структуры Arrays1DList
    Arrays1DList arrays1DList_new;
    arrays1DList_new.numElements = numArrays1D;
    arrays1DList_new.arrays1D = (Array1D*)malloc(arrays1DList_new.numElements * sizeof(arrays1DList_new.arrays1D));
    
    // Создаём в ОЗУ numElements экземпляров структуры Array1D
    for(int i=0; i<arrays1DList_new.numElements; i++)
    {
        arrays1DList_new.arrays1D[i] = Array1D_create_RAM(numElements);
    }    
        
    return arrays1DList_new;
}

// Создаёт в памяти GPU экземпляр структуры Arrays1DList, содержащий массив из numArrays1D
// элементов структуры Array1D. Каждый элемент структуры Array1D
// содержит numElements
Arrays1DList Arrays1DList_copy_to_GPU(Arrays1DList arrays1DListRAM)
{
    printf("\nSTART Arrays1DList_copy_to_GPU(Arrays1DList arrays1DListRAM)\n");
    // Структура в памяти GPU
    Arrays1DList arrays1DList_GPU = Arrays1DList_create_RAM(arrays1DListRAM.numElements, 10);/////////////////////////// 10 !!!!!
    //arrays1DList_GPU.numElements = arrays1DListRAM.numElements;
    
    for(int i=0; i<arrays1DListRAM.numElements; i++)
    {
        printf("\tCreating Array1D number %d\n", i);
        Array1D_print(arrays1DListRAM.arrays1D[i]);
        arrays1DList_GPU.arrays1D[i] = Array1D_create_GPU(arrays1DListRAM.arrays1D[i]);/// ERROR!!!!
        printf("\tArray1D number %d created\n", i);
    }

    cudaMalloc( &(arrays1DList_GPU.arrays1D),
        arrays1DList_GPU.numElements * sizeof( arrays1DList_GPU.arrays1D ) );
    cudaMemcpy(arrays1DList_GPU.arrays1D,
        arrays1DListRAM.arrays1D,
        arrays1DList_GPU.numElements * sizeof(arrays1DList_GPU.arrays1D),
        cudaMemcpyHostToDevice);

    printf("END Arrays1DList_copy_to_GPU(Arrays1DList arrays1DListRAM)\n");
    return arrays1DList_GPU;     
}

__host__ __device__ void printArrays1DList(Arrays1DList arrays1DListGPU)
{
    printf("\n----- __host__ __device__ void printArrays1DList(Arrays1DList arrays1DListGPU) -----\n");
    printf("arrays1DListGPU.numElements = %d\n", arrays1DListGPU.numElements);
    
    for(int i=0; i<arrays1DListGPU.numElements; i++)
        Array1D_print(arrays1DListGPU.arrays1D[i]);
}

__global__ void printArrays1DListKernel(Arrays1DList arrays1DListGPU)
{
    printf("\n----- __global__ printArrays1DListKernel (Arrays1DList arrays1DListGPU) -----\n");
    printArrays1DList(arrays1DListGPU);
}
////////////////////////////////////////////////////

int main()
{
    Arrays1DList arrays1DList = Arrays1DList_create_RAM(2, 10);
    printf("Init Arrays1DList in RAM\n");
    printf("\tprintArrays1DList(arrays1DList)\n");
    printArrays1DList(arrays1DList);
    printf("-------------------------------\n");

    Arrays1DList arrays1DListGPU = Arrays1DList_copy_to_GPU(arrays1DList);
        
    printArrays1DListKernel<<< 1, 1 >>>( arrays1DListGPU );
    
    //Array1D array1D_RAM_result = Array1D_copy_GPU_to_RAM(array1D_GPU);
    //Array1D_print(array1D_RAM_result);

    //Array1D_free_GPU(array1D_GPU);

    return 1;
}