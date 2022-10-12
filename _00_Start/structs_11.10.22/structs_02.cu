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
    printf("fooInfo.len = %d\n", arr1D->len);
    printf("fooInfo.arr = ");
    print_1D_array(arr1D->arr, arr1D->len);
}

// Kernel
__global__ void add_M_Kernel(Array1D arr1D, int M)
{
    Array1D_print(&arr1D);
    for(int i=0; i<arr1D.len; i++)
        arr1D.arr[i] += M;
    Array1D_print(&arr1D); 
}

Array1D Array1D_create_RAM(int numElements)
{
    int* array = (int*)malloc(numElements * sizeof(int));// Массив в ОЗУ
    for(int i = 0; i < numElements; i++)
        array[i] = i;

    print_1D_array(array, numElements);
    
    Array1D array1D;
    array1D.len = numElements;
    array1D.arr = array;
    
    return array1D;
}

Array1D Array1D_create_GPU(int numElements)
{
    int* array = (int*)malloc(numElements * sizeof(int));// Массив в ОЗУ
    for(int i = 0; i < numElements; i++)
        array[i] = i;

    print_1D_array(array, numElements);

    Array1D array1D;
    array1D.len = numElements;
    
    cudaMalloc( &(array1D.arr), array1D.len * sizeof( array1D.arr ) );
    cudaMemcpy(array1D.arr, array, array1D.len * sizeof(array1D.arr), cudaMemcpyHostToDevice);

    return array1D;
}

int main()
{
    Array1D array1D_GPU = Array1D_create_GPU(10);
        
    add_M_Kernel<<< 1, 1 >>>( array1D_GPU, 5 );

    Array1D array1D_ram = Array1D_create_RAM(10);
    cudaMemcpy(array1D_ram.arr, array1D_GPU.arr, array1D_GPU.len * sizeof(array1D_GPU.arr), cudaMemcpyDeviceToHost);
    print_1D_array(array1D_ram.arr, array1D_ram.len);
    

    cudaFree( array1D_GPU.arr );

    return 1;
}