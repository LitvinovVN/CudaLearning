// https://codeyarns.com/tech/2011-03-04-cuda-structures-as-kernel-parameters.html#gsc.tab=0
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



int main()
{
    int N = 10;
    int* array = (int*)malloc(N * sizeof(int));// Массив в ОЗУ
    for(int i = 0; i < N; i++)
        array[i] = i;

    print_1D_array(array, N);

    int* dArray = NULL; // Массив в GPU

    Array1D array1D;
    array1D.len = N;
    cudaMalloc( &(array1D.arr), array1D.len * sizeof( array1D.arr ) );

    cudaMemcpy(array1D.arr, array, array1D.len * sizeof(array1D.arr), cudaMemcpyHostToDevice);
    
    add_M_Kernel<<< 1, 1 >>>( array1D, 15 );
    add_M_Kernel<<< 1, 1 >>>( array1D, 5 );

    cudaMemcpy(array, array1D.arr, array1D.len * sizeof(array1D.arr), cudaMemcpyDeviceToHost);
    print_1D_array(array, N);

    cudaFree( array1D.arr );

    return 1;
}