#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include <stdlib.h>
#include <stdio.h>
//#include <math.h>
//#include <time.h>
//#include <cstring>
#include "locale.h"
#include <malloc.h>
using namespace std;

#define GridNx 5 // Размерность расчетной сетки по оси x
#define GridNy 6 // Размерность расчетной сетки по оси y
#define GridNz 10 // Размерность расчетной сетки по оси z
#define GridN GridNx*GridNy*GridNz // Суммарное число узлов расчетной сетки
#define GridXY GridNx * GridNy // Число узлов в плоскости XY, т.е. в одном слое по Z

#define CudaCoresNumber 192 // Количество ядер cuda (https://geforce-gtx.com/710.html - для GT710, для другой видеокарты необходимо уточнить)


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

int Add2Vectors(bool& retflag);
void Print3dArray(int* host_c);
void ConveyorTest();


/// <summary>
/// Отображает параметры видеоадаптера
/// </summary>
void ShowVideoadapterProperties() {
    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount(&count);
    for (size_t i = 0; i < count; i++)
    {
        cudaGetDeviceProperties(&prop, i);
        printf("Наименование устройства:        %s\n", prop.name);
        printf("Вычислительные возможности:     %d.%d\n", prop.major, prop.minor);
        printf("Количество мультипроцессоров:   %d\n", prop.multiProcessorCount);
        printf("Размер warp'а:                  %d\n", prop.warpSize);
    }

    printf("Количество ядер cuda: %d (уточнить по документации к видеоадаптеру)\n", CudaCoresNumber);
}

/// <summary>
/// Отображает параметры сетки
/// </summary>
void ShowGridProperties()
{
    printf("\n--------------Характеристики расчетной сетки----------------\n");
    printf("Размерность расчетной сетки по оси x:                %d\n", GridNx);
    printf("Размерность расчетной сетки по оси y:                %d\n", GridNy);
    printf("Размерность расчетной сетки по оси z:                %d\n", GridNz);
    printf("Суммарное число узлов расчетной сетки:               %d\n", GridN);
    printf("Число узлов в плоскости XY, т.е. в одном слое по Z:  %d\n", GridXY);
    printf("----------------------------------------------------------\n");
}

int main()
{
    // Включение поддержки кириллицы в консоли
    setlocale(LC_CTYPE, "rus");
    // Отображение параметров видеокарты
    ShowVideoadapterProperties();
    // Отображение параметров сетки
    ShowGridProperties();     
    // Тест конвейера вычислений
    ConveyorTest();

    bool retflag;
    int retval = Add2Vectors(retflag);
    if (retflag) return retval;

    return 0;
}

/// <summary>
/// Инициализация вектора на GPU
/// </summary>
/// <param name="c"></param>
/// <param name="size"></param>
/// <returns></returns>
__global__ void initVectorInGpuKernel(int* c, unsigned int size)
{
    // Compute the offset in each dimension
    const size_t offsetX = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t offsetY = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t offsetZ = blockDim.z * blockIdx.z + threadIdx.z;

    // Make sure that you are not actually outs
    if (offsetX >= GridNx || offsetY >= GridNy || offsetZ >= GridNz)
        return;

    // Compute the linear index assuming that X,Y then Z memory ordering
    const size_t idx = offsetZ * GridNx * GridNy + offsetY * GridNx + offsetX;
    printf("blockIdx.x = %d, blockIdx.y = %d, i = %d\n", blockIdx.x, blockIdx.y, idx);
    printf("offsetX = %d, offsetY = %d, offsetZ = %d \n", offsetX, offsetY, offsetZ);
    
    long nodeIndex = idx;
    for (size_t z = 0; z < GridNz; z++)
    {
        nodeIndex += GridNx * GridNy;
        if (idx < size)
        {
            c[nodeIndex] = nodeIndex;
        }
    }        
}

/// <summary>
/// Увеличивает в 2 раза значения в элементах, сумма индексов которых равна s
/// </summary>
/// <param name="c"></param>
/// <param name="size">Кол-во элементов вектора c</param>
/// <param name="s">i+j+k</param>
/// <returns></returns>
__global__ void conveyorKernel(int* c, unsigned int size, int s)
{
    // Compute the offset in each dimension
    const size_t offsetX = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t offsetY = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t offsetZ = blockDim.z * blockIdx.z + threadIdx.z;

    // Make sure that you are not actually outs
    if (offsetX >= GridNx || offsetY >= GridNy || offsetZ >= GridNz)
        return;

    // Compute the linear index assuming that X,Y then Z memory ordering
    const size_t idx = offsetZ * GridNx * GridNy + offsetY * GridNx + offsetX;
    //printf("blockIdx.x = %d, blockIdx.y = %d, i = %d\n", blockIdx.x, blockIdx.y, idx);
    //printf("offsetX = %d, offsetY = %d, offsetZ = %d \n", offsetX, offsetY, offsetZ);

    long nodeIndex = idx;    
    for (size_t z = 0; z < GridNz; z++)
    {
        nodeIndex += GridNx * GridNy;
        if (idx < size && (offsetX + offsetY + offsetZ) == s)
        {
            c[nodeIndex] = c[nodeIndex] * 2;
        }
    }
}

void Print3dArray(int* host_c)
{
    for (size_t k = 0; k < GridNz; k++)
    {
        printf("\n--------------------------------------\n");
        printf("------------ k = %d ------------------\n", k);
        printf("--------------------------------------\n");
        for (size_t j = 0; j < GridNy; j++)
        {
            printf("------------ j = %d ------------------\n", j);
            for (size_t i = 0; i < GridNx; i++)
            {
                printf("%d\t", host_c[i + j * GridNx + k * GridXY]);
            }
            printf("\n");
        }
    }
}

void ConveyorTest()
{
    printf("------------------Тест конвейерного вычисления----------------\n");
    // 1. Вычисляем размер массива данных
    int size = GridN; // Кол-во элементов
    size_t sizeInBytesInt = size * sizeof(int);// Размер массива в байтах

    // 2. Выделяем память под массив в ОЗУ
    int* host_c = 0;
    host_c = (int*)malloc(sizeInBytesInt);

    // 3. Выделяем память под массив на видеокарте        
    int* dev_c = 0;
    cudaError_t cudaStatus = cudaMalloc((void**)&dev_c, sizeInBytesInt);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "!!!!!!!!!!!!cudaMalloc failed in ConveyorTest()!!!!!!!!!!!!");
        return;
    }
    
    // 3. Инициализируем массив на GPU
    initVectorInGpuKernel <<< dim3(GridNx, GridNy), 1 >>> (dev_c, GridN);
    cudaDeviceSynchronize();

    // 4. Старт конвейера
    int s = 1;
    //conveyorKernel <<< dim3(GridNx, GridNy), 1 >>> (dev_c, GridN, s);
    cudaDeviceSynchronize();

    cudaMemcpy(host_c, dev_c, sizeInBytesInt, cudaMemcpyDeviceToHost);
        
    Print3dArray(host_c);
    

    // Удаляем буферы памяти
    free(host_c);
    cudaFree(dev_c);

    printf("--------------Тест конвейерного вычисления (конец)------------\n");
}



__global__ void conveyorTestKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}




/// <summary>
/// Сложение двух векторов
/// </summary>
/// <param name="retflag"></param>
/// <returns></returns>
int Add2Vectors(bool& retflag)
{
    retflag = true;
    const int arraySize = GridN;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    retflag = false;
    return {};
}

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<size, 1>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
