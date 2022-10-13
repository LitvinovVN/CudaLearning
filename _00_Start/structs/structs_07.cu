/*
Добавить функцию, создающую в ОЗУ структуру Arrays1DList для хранения массива экземпляров структур Array1D,
и возвращающую созданный объект
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
    Array1D array1D_GPU;
    array1D_GPU.len = array1D_RAM.len;

    cudaMalloc( &(array1D_GPU.arr), array1D_GPU.len * sizeof( array1D_GPU.arr ) );
    cudaMemcpy(array1D_GPU.arr, array1D_RAM.arr, array1D_GPU.len * sizeof(array1D_GPU.arr), cudaMemcpyHostToDevice);

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
////////////////////////////////////////////////////

int main()
{
    Arrays1DList arrays1DList = Arrays1DList_create_RAM(2, 10);
    Array1D_print(arrays1DList.arrays1D[0]);
    Array1D_print(arrays1DList.arrays1D[1]);

    //Array1D array1D_GPU = Array1D_create_GPU(array1D_RAM);
        
    //add_M_Kernel<<< 1, 1 >>>( array1D_GPU, 5 );
    
    //Array1D array1D_RAM_result = Array1D_copy_GPU_to_RAM(array1D_GPU);
    //Array1D_print(array1D_RAM_result);

    //Array1D_free_GPU(array1D_GPU);

    return 1;
}