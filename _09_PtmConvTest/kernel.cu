#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "locale.h"
#include <malloc.h>
#include <stdlib.h>

#define CudaCoresNumber 192 // Количество ядер cuda (https://geforce-gtx.com/710.html - для GT710, для другой видеокарты необходимо уточнить)
#define ThreadsNumber /*9*/ 384 // от 1 до CudaCoresNumber
// 49152 байт - размер shared-памяти для GT 710
#define SharedMemorySize 49152/sizeof(double) // Размерность массива распределённой памяти для одного слоя XY
#define BlockSizeX ThreadsNumber // Размерность блока по X задаём равной числу нитей в блоке
#define BlockSizeY /*10*/ 25000 /*SharedMemorySize/BlockSizeX*/ // Размерность блока по Y от 1 до CudaCoresNumber

#define GridNx (BlockSizeX + 1) // Размерность расчетной сетки по оси x
#define GridNy BlockSizeY // Размерность расчетной сетки по оси y
#define GridNz 2 // Размерность расчетной сетки по оси z
#define GridN GridNx*GridNy*GridNz // Суммарное число узлов расчетной сетки
#define GridXY GridNx * GridNy // Число узлов в плоскости XY, т.е. в одном слое по Z

#define EPS 0.001

void Print3dArray(int* host_c);
void Print3dArrayDouble(double* host_c);

void PtmTest();


#pragma region Вспомогательные функции

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
        printf("Тактовая частота:               %d МГц\n", prop.clockRate / 1000);
        printf("Перекрытие копирования (уст.): ");
        if (prop.deviceOverlap)
        {
            printf("Разрешено\n");
        }
        else
        {
            printf("Запрещено\n");
        }
        printf("Тайм-аут выполнения ядра: ");
        if (prop.kernelExecTimeoutEnabled)
        {
            printf("Включен\n");
        }
        else
        {
            printf("Выключен\n");
        }
        printf("Количество асинхронных DMA движков: %d (1: копирование данных + ядро, 2: копирование данных up + копирование данных down + ядро)\n", prop.asyncEngineCount);

        printf("------------ Информация о памяти ---------------\n");
        printf("Всего глобальной памяти:        %ld байт\n", prop.totalGlobalMem);
        printf("Всего константной памяти:       %ld байт\n", prop.totalConstMem);

        printf("------------ Информация о мультипроцессорах ---------------\n");
        printf("Количество мультипроцессоров:   %d\n", prop.multiProcessorCount);
        printf("Количество распределяемой памяти на 1 блок:   %d байт\n", prop.sharedMemPerBlock);
        printf("Количество распределяемой памяти на 1 мультипроцессор:   %ld байт\n", prop.sharedMemPerMultiprocessor);
        printf("Количество 32х-битных регистров на 1 блок:   %d байт\n", prop.regsPerBlock);
        printf("Размер warp'а:                  %d\n", prop.warpSize);
        printf("Максимальное количество нитей в блоке: %d\n", prop.maxThreadsPerBlock);
        printf("Максимальное количество нитей в блоке: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Максимальные размеры сетки: (%ld, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
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

__host__ __device__ void Print3dArrayDouble(double* host_c)
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
                printf("%lf\t", host_c[i + j * GridNx + k * GridXY]);
            }
            printf("\n");
        }
    }
}
#pragma endregion

#pragma region Kernels

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
        printf("blockIdx.x = %d,  blockIdx.y = %d,  blockIdx.z = %d\n", blockIdx.x, blockIdx.y, blockIdx.y);
        printf("blockDim.x = %d,  blockDim.y = %d , blockDim.z = %d\n", blockDim.x, blockDim.y, blockDim.z);
        printf("offsetX = %d,     offsetY = %d,     offsetZ = %d\n", offsetX, offsetY, offsetZ);
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
/// ПТМ, проход от 0 до Nx+Ny+Nz
/// </summary>
/// <param name="c"></param>
/// <param name="size">Кол-во элементов вектора s</param>
/// <returns></returns>
__global__ void ptmKernel1(double* r, double* c0, double* c2, double* c4, double* c6, unsigned int size, double omega)
{
    // Compute the offset in each dimension
    const size_t threadX = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Индекс строки, которую обрабатывает текущий поток 
    const size_t idx = threadX + 1;

    size_t currentY = 1; // 0 - граница, берём 1
    for (size_t s = 2; s <= GridNx + GridNy - 2; s++)
    {
        __syncthreads();
        if (idx + currentY == s && s < GridNy + idx)
        {            
            size_t nodeIndex = idx + (BlockSizeX+1) * currentY + GridXY;
                        
            size_t m0 = nodeIndex;

            if (c0[m0] > 0)
            {
                size_t m2 = m0 - 1;
                size_t m4 = m0 - GridNx;
                size_t m6 = m0 - GridXY;                
                r[m0] = (omega * (c2[m0] * r[m2] + c4[m0] * r[m4] + c6[m0] * r[m6]) + r[m0]) / ((0.5 * omega + 1) * c0[m0]);
            }
            
            currentY++;
        }        
    }
        
}


/// <summary>
/// ПТМ, проход от 0 до Nx+Ny+Nz
/// </summary>
/// <param name="c"></param>
/// <param name="size">Кол-во элементов вектора s</param>
/// <returns></returns>
__global__ void ptmKernel2(double* r, double* c0, double* c2, double* c4, double* c6, unsigned int size, double omega)
{
    __shared__ double cache[BlockSizeX];
    
    // Compute the offset in each dimension
    const size_t threadX = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Индекс строки, которую обрабатывает текущий поток 
    const size_t idx = threadX + 1;

    size_t currentY = 1; // 0 - граница, берём 1

    for (size_t s = 2; s <= GridNx + GridNy - 2; s++)
    {
        __syncthreads();
        if (idx + currentY == s && s < GridNy + idx)
        {
            size_t nodeIndex = idx + (BlockSizeX + 1) * currentY + GridXY;

            size_t m0 = nodeIndex;

            double c0m0 = c0[m0];
            if (c0m0 > 0)
            {
                size_t m2 = m0 - 1;
                size_t m4 = m0 - GridNx;
                size_t m6 = m0 - GridXY;

                double rm4 = 0;
                if (s > 2 + threadX)
                {
                    rm4 = cache[threadX];
                }
                else
                {
                    rm4 = r[m4];
                }

                double rm2 = 0;
                if (threadX != 0 && s > 3 + threadX)
                {
                    rm2 = cache[threadX-1];
                }
                else
                {
                    rm2 = r[m2];
                }


                double rm0 = (omega * (c2[m0] * rm2 + c4[m0] * rm4 + c6[m0] * r[m6]) + r[m0]) / ((0.5 * omega + 1) * c0m0);
                cache[threadX] = rm0;
                r[m0] = rm0;
            }
            
            currentY++;
        }
    }

}

#pragma endregion Kernels







int main()
{
    // Включение поддержки кириллицы в консоли
    setlocale(LC_CTYPE, "rus");
    // Отображение параметров видеокарты
    ShowVideoadapterProperties();
    // Отображение параметров сетки
    ShowGridProperties();
    // Тест конвейера вычислений
    PtmTest();   

    return 0;
}




void PtmTest()
{
    printf("------------------Тест конвейерного вычисления----------------\n");
    // 1. Вычисляем размер массива данных
    size_t size = GridN; // Кол-во элементов
    size_t sizeInBytesInt = size * sizeof(int);   // Размер массива int в байтах
    size_t sizeInBytesDouble = size * sizeof(double);// Размер массива double в байтах

    // 2. Выделяем память под массив в ОЗУ    
    double* host_c0 = (double*)malloc(sizeInBytesDouble);
    double* host_c1 = (double*)malloc(sizeInBytesDouble);
    double* host_c2 = (double*)malloc(sizeInBytesDouble);
    double* host_c3 = (double*)malloc(sizeInBytesDouble);
    double* host_c4 = (double*)malloc(sizeInBytesDouble);
    double* host_c5 = (double*)malloc(sizeInBytesDouble);
    double* host_c6 = (double*)malloc(sizeInBytesDouble);
    double* host_u = (double*)malloc(sizeInBytesDouble);
    double* host_f = (double*)malloc(sizeInBytesDouble);
    double* host_r = (double*)malloc(sizeInBytesDouble);
    double* host_Awr = (double*)malloc(sizeInBytesDouble);
    double* host_Rr = (double*)malloc(sizeInBytesDouble);
    double* host_crr = (double*)malloc(sizeInBytesDouble);
    int* host_s = (int*)malloc(sizeInBytesInt);

    double* host_r_cpu = (double*)malloc(sizeInBytesDouble);//массив для проверки вычислений на cpu

    // 2a Инициализация массивов
    for (size_t k = 0; k < GridNz; k++)
    {
        for (size_t j = 0; j < GridNy; j++)
        {
            for (size_t i = 0; i < GridNx; i++)
            {
                size_t m0 = i + j * GridNx + k * GridXY;
                host_c0[m0] = 4;
                host_c1[m0] = -1;
                host_c2[m0] = -1;
                host_c3[m0] = -1;
                host_c4[m0] = -1;
                host_c5[m0] = -1;
                host_c6[m0] = -1;
                host_u[m0] = 0;
                host_f[m0] = 10;
                host_r[m0] = 1;
                host_Awr[m0] = 0;
                host_Rr[m0] = 0;
                host_crr[m0] = 0;
                host_s[m0] = 1;

                host_r_cpu[m0] = host_r[m0];
            }
        }
    }


    // 3. Выделяем память под массивы на видеокарте    
    double* dev_c0 = NULL;
    cudaMalloc((void**)&dev_c0, sizeInBytesDouble);

    double* dev_c1 = NULL;
    cudaMalloc((void**)&dev_c1, sizeInBytesDouble);

    double* dev_c2 = NULL;
    cudaMalloc((void**)&dev_c2, sizeInBytesDouble);

    double* dev_c3 = NULL;
    cudaMalloc((void**)&dev_c3, sizeInBytesDouble);

    double* dev_c4 = NULL;
    cudaMalloc((void**)&dev_c4, sizeInBytesDouble);

    double* dev_c5 = NULL;
    cudaMalloc((void**)&dev_c5, sizeInBytesDouble);

    double* dev_c6 = NULL;
    cudaMalloc((void**)&dev_c6, sizeInBytesDouble);

    double* dev_u = NULL;
    cudaMalloc((void**)&dev_u, sizeInBytesDouble);

    double* dev_f = NULL;
    cudaMalloc((void**)&dev_f, sizeInBytesDouble);

    double* dev_r = NULL;
    cudaMalloc((void**)&dev_r, sizeInBytesDouble);

    double* dev_Awr = NULL;
    cudaMalloc((void**)&dev_Awr, sizeInBytesDouble);

    double* dev_Rr = NULL;
    cudaMalloc((void**)&dev_Rr, sizeInBytesDouble);

    double* dev_crr = NULL;
    cudaMalloc((void**)&dev_crr, sizeInBytesDouble);

    int* dev_s = NULL;
    cudaMalloc((void**)&dev_s, sizeInBytesInt);

    // 4. Копируем массивы из ОЗУ в GPU
    cudaMemcpy(dev_c0, host_c0, sizeInBytesDouble, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c1, host_c1, sizeInBytesDouble, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c2, host_c2, sizeInBytesDouble, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c3, host_c3, sizeInBytesDouble, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c4, host_c4, sizeInBytesDouble, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c5, host_c5, sizeInBytesDouble, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c6, host_c6, sizeInBytesDouble, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_u, host_u, sizeInBytesDouble, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_f, host_f, sizeInBytesDouble, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_r, host_r, sizeInBytesDouble, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Awr, host_Awr, sizeInBytesDouble, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Rr, host_Rr, sizeInBytesDouble, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_crr, host_crr, sizeInBytesDouble, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_s, host_s, sizeInBytesInt, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // 5. Настраиваем параметры блоков CUDA        
    int host_isGreater = 0;
    int* dev_isGreater = NULL;
    cudaMalloc((void**)&dev_isGreater, sizeof(int));

    float gpuTime = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    

    printf("-----------------------------------------------------------\n");
    printf("-------------Start of PTM Iteration------------------------\n");

    // Старт конвейера    
    double omega = 0.05;
    double tay = 2 * omega;

    printf("--- ptmKernel1 Starting... ---\n");

    cudaEventRecord(start, 0);
    //ptmKernel1 << < 1, BlockSizeX >> > (dev_r, dev_c0, dev_c2, dev_c4, dev_c6, GridN, omega);
    ptmKernel2 << < 1, BlockSizeX >> > (dev_r, dev_c0, dev_c2, dev_c4, dev_c6, GridN, omega);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);        
    printf("ptmKernel1 Time: %lf\n", gpuTime);
    cudaDeviceSynchronize();

    printf("-------------End of PTM Iteration------------------------\n");


    // Копируем массив с результатами вычислений из памяти GPU в ОЗУ
    cudaMemcpy(host_r, dev_r, sizeInBytesDouble, cudaMemcpyDeviceToHost);    

    cudaDeviceSynchronize();
    // Выводим на консоль массив с результатами вычислений
    //Print3dArrayDouble(host_r);    


    // Вычисляем на cpu 
    clock_t cpu_start = clock();
            
    for (size_t k = 1; k < GridNz; k++)
    {
        for (size_t j = 1; j < GridNy; j++)
        {
            for (size_t i = 1; i < GridNx; i++)
            {
                size_t m0 = i + j * GridNx + k * GridXY;
                size_t m2 = m0 - 1;
                size_t m4 = m0 - GridNx;
                size_t m6 = m0 - GridXY;
                //host_r_cpu[m0] = host_r_cpu[m2];
                host_r_cpu[m0] = (omega * (host_c2[m0] * host_r_cpu[m2] + host_c4[m0] * host_r_cpu[m4] + host_c6[m0] * host_r_cpu[m6]) + host_r_cpu[m0]) / ((0.5 * omega + 1) * host_c0[m0]);
            }
        }
    }
    clock_t cpu_end = clock();
    double seconds = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("Время расчета на CPU (clock): %f мс\n", seconds * 1000);    

    bool isEquals = true;
    for (size_t k = 1; k < GridNz; k++)
    {
        for (size_t j = 1; j < GridNy; j++)
        {
            for (size_t i = 1; i < GridNx; i++)
            {
                size_t m0 = i + j * GridNx + k * GridXY;
                if (abs(host_r_cpu[m0] - host_r[m0]) > 0.000001)
                {
                    isEquals = false;
                    printf("host_r_cpu[%d] = %lf | host_r[m0]=%lf\n", m0, host_r_cpu[m0], host_r[m0]);
                }
            }
        }
    }
    if (isEquals)
    {
        printf("\nПроверка равенства массивов: УСПЕХ\n");
    }
    else
    {
        printf("\nПроверка равенства массивов: ОШИБКА!\n");
    }

    // Выводим на консоль массив с результатами вычислений
    //printf("\n\nCPU\n");
    //Print3dArrayDouble(host_r_cpu);

    // Удаляем буферы памяти
    free(host_c0);
    free(host_c1);
    free(host_c2);
    free(host_c3);
    free(host_c4);
    free(host_c5);
    free(host_c6);
    free(host_u);
    free(host_f);
    free(host_r);
    free(host_Awr);
    free(host_Rr);
    free(host_crr);
    free(host_s);

    cudaFree(dev_c0);
    cudaFree(dev_c1);
    cudaFree(dev_c2);
    cudaFree(dev_c3);
    cudaFree(dev_c4);
    cudaFree(dev_c5);
    cudaFree(dev_c6);
    cudaFree(dev_u);
    cudaFree(dev_f);
    cudaFree(dev_r);
    cudaFree(dev_Awr);
    cudaFree(dev_Rr);
    cudaFree(dev_crr);
    cudaFree(dev_s);

    // Сбрасываем устройство CUDA
    cudaDeviceReset();

    printf("--------------Тест конвейерного вычисления (конец)------------\n");
}

