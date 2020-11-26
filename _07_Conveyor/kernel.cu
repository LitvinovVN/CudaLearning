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

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
    {
        printf("\n--------------initVectorInGpuKernel-------------------\n");
        printf("threadIdx.x = %d, threadIdx.y = %d, threadIdx.z = %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
        printf("blockIdx.x = %d,  blockIdx.y = %d,  blockIdx.z = %d\n", blockIdx.x,   blockIdx.y,  blockIdx.y);
        printf("blockDim.x = %d,  blockDim.y = %d , blockDim.z = %d\n", blockDim.x,   blockDim.y,  blockDim.z);
        printf("offsetX = %d,     offsetY = %d,     offsetZ = %d\n",    offsetX,      offsetY,     offsetZ);
        printf("idx = %d\n", idx);
        printf("\n-----------initVectorInGpuKernel (end)--------------\n");
    }

    
    long nodeIndex = idx;
    for (size_t z = 0; z < GridNz; z++)
    {        
        if (idx < size)
        {
            c[nodeIndex] = nodeIndex;
        }
        nodeIndex += GridNx * GridNy;
    }        
}

/// <summary>
/// Увеличивает в 2 раза значения в элементах, сумма индексов которых равна s
/// </summary>
/// <param name="c"></param>
/// <param name="size">Кол-во элементов вектора s</param>
/// <param name="s">i+j+k</param>
/// <returns></returns>
__global__ void conveyorKernel(int* c, unsigned int size, int s)
{
    // Compute the offset in each dimension
    const size_t offsetX = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t offsetY = blockDim.y * blockIdx.y + threadIdx.y;
    //const size_t offsetZ = blockDim.z * blockIdx.z + threadIdx.z;

    // Make sure that you are not actually outs
    if (offsetX >= GridNx || offsetY >= GridNy /*|| offsetZ >= GridNz*/)
        return;

    // Compute the linear index assuming that X,Y then Z memory ordering
    const size_t idx = /*offsetZ * GridNx * GridNy +*/ offsetY * GridNx + offsetX;
    //printf("blockIdx.x = %d, blockIdx.y = %d, i = %d\n", blockIdx.x, blockIdx.y, idx);
    //printf("offsetX = %d, offsetY = %d, offsetZ = %d \n", offsetX, offsetY, offsetZ);

    long nodeIndex = idx;    
    for (size_t z = 0; z < GridNz; z++)
    {        
        if (idx < size && (offsetX + offsetY + z) == s)
        {
            c[nodeIndex] = c[nodeIndex] + 1000;
        }
        nodeIndex += GridNx * GridNy;
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
    
    // 4. Настраиваем параметры блоков CUDA
    dim3 blocks(GridNx, GridNy);

    // 4. Инициализируем массив на GPU
    initVectorInGpuKernel <<< blocks, 1 >>> (dev_c, GridN);
    cudaDeviceSynchronize();

    // 5. Старт конвейера
    int s = 3;
    conveyorKernel <<< blocks, 1 >>> (dev_c, GridN, s);
    cudaDeviceSynchronize();

    // Копируем массив с результатами вычислений из памяти GPU в ОЗУ
    cudaMemcpy(host_c, dev_c, sizeInBytesInt, cudaMemcpyDeviceToHost);
    
    // Выводим на консоль массив с результатами вычислений
    Print3dArray(host_c);
    

    // Удаляем буферы памяти
    free(host_c);
    cudaFree(dev_c);

    // Сбрасываем устройство CUDA
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return;
    }

    printf("--------------Тест конвейерного вычисления (конец)------------\n");
}



__global__ void conveyorTestKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
