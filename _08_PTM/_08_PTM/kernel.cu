#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "locale.h"
#include <malloc.h>
#include <stdlib.h>
using namespace std;

//////////////// Результаты ////////////////
// GT710 2Gb, Core i5-7400 3GHz, 16 Gb ОЗУ
// 100x100x160
// nevyazkaKernelTime = 2019 ms
// nevyazkaGreaterEpsKernel = 0.9 ms
// ptmKernel1. i = 3;   gpuTime = 86.5 ms
// ptmKernel1. i = 182; gpuTime = 92.09 ms
// ptmKernel1. i = 356; gpuTime = 86.9 ms
// ptmKernel2 = 39647 ms
// awrRrKernel + RwRw + Aww + ww = 3029 ms
// uKernel = 281 ms
////////////////////////////////////////////

#define BLOCK_SIZE 256

#define GridNx 100 // Размерность расчетной сетки по оси x
#define GridNy 100 // Размерность расчетной сетки по оси y
#define GridNz 160 // Размерность расчетной сетки по оси z
#define GridN GridNx*GridNy*GridNz // Суммарное число узлов расчетной сетки
#define GridXY GridNx * GridNy // Число узлов в плоскости XY, т.е. в одном слое по Z

#define CudaCoresNumber 192 // Количество ядер cuda (https://geforce-gtx.com/710.html - для GT710, для другой видеокарты необходимо уточнить)

#define EPS 0.001

void Print3dArray(int* host_c);
void Print3dArrayDouble(double* host_c);
double Reduce(double* data, long n);

void PtmTest();
void ReductionTest();

#pragma region Kernels

__global__ void uKernel(double* u, double* r, long size, double tay)
{
    const size_t threadX = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t threadY = blockDim.y * blockIdx.y + threadIdx.y;

    if (threadX >= GridNx - 1 || threadY >= GridNy - 1 || threadX == 0 || threadY == 0)
        return;

    const size_t idx = threadY * GridNx + threadX;
        
    long nodeIndex = idx;
    for (size_t z = 1; z < GridNz - 1; z++)
    {
        long m0 = nodeIndex + z * GridXY;        

        if (nodeIndex < size)
        {
            u[m0] = u[m0] + tay * r[m0];
        }
    }
}

__global__ void sumElFromN1ToN2Kernel(double* sum, double* data, long N1, long N2)
{
    *sum = 0;
    if (threadIdx.x == 0)
    {
        for (size_t i = N1; i <= N2; i++)
        {
            *sum += data[i];
            //printf("sumElFromN1ToN2Kernel, 41: *sum = %lf", *sum);
        }        
    }
}

__global__ void reduceKernel(double* inData, double* outData)
{
    __shared__ double data[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    // Записать сумму первых двух элементов в разделяемую память
    data[tid] = inData[i] + inData[i + blockDim.x];
    
    __syncthreads();  // Дождаться загрузки данных

    for (int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            data[tid] += data[tid + s];
        }

        __syncthreads();
    }

    if (tid < 32)  // Развернуть последние итерации
    {
        data[tid] += data[tid + 32];
        data[tid] += data[tid + 16];
        data[tid] += data[tid + 8];
        data[tid] += data[tid + 4];
        data[tid] += data[tid + 2];
        data[tid] += data[tid + 1];
    }

    if (tid == 0)  // Сохранить сумму элементов блока
    {
        outData[blockIdx.x] = data[0];        
    }
}

#pragma endregion Kernels

/// <summary>
/// Суммирование массива редукцией
/// </summary>
/// <param name="data"></param>
/// <param name="n"></param>
/// <returns></returns>
double Reduce(double* data, long n)
{
    double res = 0;

    double* sums = NULL;
    int numBlocks = n / 512;

    //tex:
    // Суммируем элементы в хвосте массива от $$numBlocks \times 512$$ до $$n - 1$$ 
    double sumRight = 0;
    long N1 = numBlocks * 512;
    long N2 = n - 1;
    //printf("N1 = %d\n", N1);
    //printf("N2 = %d\n", N2);
    
    double* dev_sumRight = NULL;
    cudaMalloc((void**)&dev_sumRight, sizeof(double));
    sumElFromN1ToN2Kernel <<< 1, 1 >> > (dev_sumRight, data, N1, N2);
    cudaMemcpy(&sumRight, dev_sumRight, sizeof(double), cudaMemcpyDeviceToHost);
    //printf("sumRight = %lf\n", sumRight);
    
    res += sumRight;

    // Выделяем память под массив сумм блоков
    cudaMalloc( (void**) &sums, numBlocks * sizeof(double));

    // Проводим поблочную редукцию, записав суммы для каждого блока в массив sums
    reduceKernel << <dim3(numBlocks), dim3(BLOCK_SIZE) >> > (data, sums);

    // Редуцируем массив сумм для блоков
    if (numBlocks > BLOCK_SIZE)
    {
        res = Reduce(sums, numBlocks);
    }
    else
    {
        double* sumsHost = new double[numBlocks];

        cudaMemcpy(sumsHost, sums, numBlocks * sizeof(double), cudaMemcpyDeviceToHost);

        for (int i = 0; i < numBlocks; i++)
        {
            res += sumsHost[i];
        }

        delete[] sumsHost;
    }

    cudaFree(sums);
    return res;
}


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
    PtmTest();
    // Тест редукции массива
    // ReductionTest();

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
/// Расчет невязки
/// </summary>
__global__ void nevyazkaKernel(double* r, double* c0, double* c1, double* c2, double* c3, double* c4, double* c5, double* c6, double* f, double* u, unsigned int size)
{
    const size_t threadX = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t threadY = blockDim.y * blockIdx.y + threadIdx.y;

    if (threadX >= GridNx-1 || threadY >= GridNy-1 || threadX == 0 || threadY == 0)
        return;

    const size_t idx = threadY * GridNx + threadX;       


    long nodeIndex = idx;
    for (size_t z = 1; z < GridNz-1; z++)
    {
        long m0 = nodeIndex + z * GridXY;
        long m1 = m0 + 1;
        long m2 = m0 - 1;
        long m3 = m0 + GridNx;
        long m4 = m0 - GridNx;
        long m5 = m0 + GridXY;
        long m6 = m0 - GridXY;

        if (idx < size)
        {
            r[m0] = f[m0] - c0[m0] * u[m0] + (c1[m0] * u[m1] + c2[m0] * u[m2] + c3[m0] * u[m3] + c4[m0] * u[m4] + c5[m0] * u[m5] + c6[m0] * u[m6]);
            //printf("r[%d] = %lf; \n",m0, r[m0]);
        }        
    }    
}

__global__ void nevyazkaGreaterEpsKernel(int* isGreater, double* r, unsigned int size, double eps)
{
    //printf("----!!!!!!!!!! 170 !!!!!!!! isGreater = %d--------\n", *isGreater);
    if (*isGreater > 0)
    {
        //printf("----!!!!!!!!!! 172 !!!!!!!! isGreater = %d| isGreater > 0 ---> return; --------\n", *isGreater);
        return;
    }
    
    const size_t threadX = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t threadY = blockDim.y * blockIdx.y + threadIdx.y;

    if (threadX >= GridNx - 1 || threadY >= GridNy - 1 || threadX == 0 || threadY == 0)
        return;

    const size_t idx = threadY * GridNx + threadX;
        
    long nodeIndex = idx;
    for (size_t z = 1; z < GridNz - 1; z++)
    {
        long m0 = nodeIndex + z * GridXY;        
        
        if (r[m0] > eps && *isGreater == 0)
        {            
            atomicExch(isGreater, 1);
            //printf("----!!!!!!!!!--- 199 ---!!!!!!!!!r[%d] = %lf; isGreater = %d--------\n", m0, r[m0], *isGreater);
            return;
        }
    }
}

/// <summary>
/// ПТМ, проход от 0 до Nx+Ny+Nz
/// </summary>
/// <param name="c"></param>
/// <param name="size">Кол-во элементов вектора s</param>
/// <param name="s">i+j+k</param>
/// <returns></returns>
__global__ void ptmKernel1(double* r, double* c0, double* c2, double* c4, double* c6, unsigned int size, int s, double omega)
{
    // Compute the offset in each dimension
    const size_t threadX = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t threadY = blockDim.y * blockIdx.y + threadIdx.y;    

    // Make sure that you are not actually outs
    if (threadX >= GridNx - 1 || threadY >= GridNy - 1 || threadX == 0 || threadY == 0)
        return;

    // Compute the linear index assuming that X,Y then Z memory ordering
    const size_t idx = threadY * GridNx + threadX;

    long nodeIndex = idx;
    for (size_t z = 1; z < GridNz-1; z++)
    {        
        if (idx < size && (threadX + threadY + z) == s)
        {            
            long m0 = nodeIndex + z * GridXY;
            
            if (c0[m0] > 0)
            {
                //printf("236: threadX + threadY + z = %d \n", threadX + threadY + z);
                long m2 = m0 - 1;
                long m4 = m0 - GridNx;
                long m6 = m0 - GridXY;

                r[m0] = (omega * (c2[m0] * r[m2] + c4[m0] * r[m4] + c6[m0] * r[m6]) + r[m0]) / ((0.5 * omega + 1) * c0[m0]);                
                //printf("243: r[%d] = %lf\n", m0, r[m0]);
            }
        }        
    }
}


/// <summary>
/// ПТМ, проход от Nx+Ny+Nz до 0
/// </summary>
/// <param name="c"></param>
/// <param name="size">Кол-во элементов вектора s</param>
/// <param name="s">i+j+k</param>
/// <returns></returns>
__global__ void ptmKernel2(double* r, double* c0, double* c1, double* c3, double* c5, unsigned int size, int s, double omega)
{
    // Compute the offset in each dimension
    const size_t threadX = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t threadY = blockDim.y * blockIdx.y + threadIdx.y;
    
    // Make sure that you are not actually outs
    if (threadX >= GridNx - 1 || threadY >= GridNy - 1 || threadX == 0 || threadY == 0)
        return;

    // Compute the linear index assuming that X,Y then Z memory ordering
    const size_t idx = threadY * GridNx + threadX;

    long nodeIndex = idx;
    for (size_t z = GridNz - 2; z >= 1; z--)
    {
        if (idx < size && (threadX + threadY + z) == s)
        {            
            long m0 = nodeIndex + z * GridXY;
            
            if (c0[m0] > 0)
            {                
                long m1 = m0 + 1;                
                long m3 = m0 + GridNx;                
                long m5 = m0 + GridXY;

                r[m0] = (omega * (c1[m0] * r[m1] + c3[m0] * r[m3] + c5[m0] * r[m5]) + r[m0] * c0[m0]) / ((0.5 * omega + 1) * c0[m0]);                
            }
        }
    }
}


__global__ void awrRrKernel(double* Awr, double* Rr, double* crr, double* r, double* c0, double* c1, double* c2, double* c3, double* c4, double* c5, double* c6, unsigned int size)
{
    // Compute the offset in each dimension
    const size_t threadX = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t threadY = blockDim.y * blockIdx.y + threadIdx.y;

    // Make sure that you are not actually outs
    if (threadX >= GridNx - 1 || threadY >= GridNy - 1 || threadX == 0 || threadY == 0)
        return;

    // Compute the linear index assuming that X,Y then Z memory ordering
    const size_t idx = threadY * GridNx + threadX;

    long nodeIndex = idx;
    for (size_t z = 1; z < GridNz - 1; z++)
    {
        if (idx < size)
        {
            long m0 = nodeIndex + z * GridXY;

            if (c0[m0] > 0)
            {
                long m1 = m0 + 1;
                long m2 = m0 - 1;
                long m3 = m0 + GridNx;
                long m4 = m0 - GridNx;
                long m5 = m0 + GridXY;
                long m6 = m0 - GridXY;

                Awr[m0] = (c0[m0] * r[m0] - ( c1[m0] * r[m1] + c2[m0] * r[m2] + c3[m0] * r[m3] + c4[m0] * r[m4] + c5[m0] * r[m5] + c6[m0] * r[m6])) * r[m0];
                double rr = 0.5 * c0[m0] * r[m0] - (c1[m0] * r[m1] + c3[m0] * r[m3] + c5[m0] * r[m5]);
                Rr[m0]  = rr * rr / c0[m0];
                crr[m0] = c0[m0] * r[m0] * r[m0];
            }
        }
    }

    /*__syncthreads();
    if (threadX == 1 && threadY == 1)
    {
        printf("\n\n ------------------- Print3dArrayDouble(Awr) ---------------------\n\n");
        Print3dArrayDouble(Awr);
    }

    __syncthreads();
    if (threadX == 1 && threadY == 1)
    {
        printf("\n\n ------------------- Print3dArrayDouble(Rr) ---------------------\n\n");
        Print3dArrayDouble(Rr);
    }

    __syncthreads();
    if (threadX == 1 && threadY == 1)
    {
        printf("\n\n ------------------- Print3dArrayDouble(crr) ---------------------\n\n");
        Print3dArrayDouble(crr);
    }*/
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

void PtmTest()
{
    printf("------------------Тест конвейерного вычисления----------------\n");
    // 1. Вычисляем размер массива данных
    int size = GridN; // Кол-во элементов
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
    double* host_Rr  = (double*)malloc(sizeInBytesDouble);
    double* host_crr = (double*)malloc(sizeInBytesDouble);
    int* host_s = (int*)malloc(sizeInBytesInt);

    // 2a Инициализация массивов
    for (size_t k = 0; k < GridNz; k++)
    {
        for (size_t j = 0; j < GridNy; j++)
        {
            for (size_t i = 0; i < GridNx; i++)
            {
                int m0 = i + j * GridNx + k * GridXY;
                host_c0[m0] = 4;
                host_c1[m0] = -1;
                host_c2[m0] = -1;
                host_c3[m0] = -1;
                host_c4[m0] = -1;
                host_c5[m0] = -1;
                host_c6[m0] = -1;
                host_u[m0] = 0;
                host_f[m0] = 10;
                host_r[m0] = 0;
                host_Awr[m0] = 0;
                host_Rr[m0] = 0;
                host_crr[m0] = 0;
                host_s[m0] = 1;
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
    cudaMemcpy(dev_Rr,  host_Rr,  sizeInBytesDouble, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_crr, host_crr, sizeInBytesDouble, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_s, host_s, sizeInBytesInt, cudaMemcpyHostToDevice);
    

    // 5. Настраиваем параметры блоков CUDA
    dim3 blocks(GridNx, GridNy);
    
    int it = 1;
    int host_isGreater = 0;
    int* dev_isGreater = NULL;
    cudaMalloc((void**)&dev_isGreater, sizeof(int));

    float gpuTime = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    do
    {
        printf("-----------------------------------------------------------\n");
        printf("-------------Start of PTM Iteration------------------------\n");
        
        cudaEventRecord(start, 0);

        // Вычисление вектора невязки
        nevyazkaKernel << < blocks, 1 >> > (dev_r, dev_c0, dev_c1, dev_c2, dev_c3, dev_c4, dev_c5, dev_c6, dev_f, dev_u, GridN);
        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        printf("nevyazkaKernelTime = %f\n", gpuTime);


        cudaEventRecord(start, 0);

        // Определение, превышает ли хотя бы один элемент вектора невязки максимальное значение ошибки        
        nevyazkaGreaterEpsKernel << <blocks, 1 >> > (dev_isGreater, dev_r, GridN, EPS);
        cudaDeviceSynchronize();
        cudaMemcpy(&host_isGreater, dev_isGreater, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        //printf("host_isGreater = %d----------\n", host_isGreater);
        // ... while()

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        printf("nevyazkaGreaterEpsKernel = %f\n", gpuTime);


        //cudaEventRecord(start, 0);

        // 6. Старт конвейера    
        double omega = 0.05;// Уточнить стартовое значение????????????????????????
        double tay = 2 * omega;

        printf("--- ptmKernel1 Starting... ---\n", gpuTime);
        for (size_t i = 3; i < GridNx + GridNy + GridNz - 3; i++)
        {
            cudaEventRecord(start, 0);
            ptmKernel1 << < blocks, 1 >> > (dev_r, dev_c0, dev_c2, dev_c4, dev_c6, GridN, i, omega);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&gpuTime, start, stop);
            printf("ptmKernel1. i = %d; gpuTime = %f\n", i, gpuTime);
        }
        printf("--- ptmKernel1 End ---\n", gpuTime);
        cudaDeviceSynchronize();

        /*cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        printf("ptmKernel1 = %f\n", gpuTime);*/


        cudaEventRecord(start, 0);

        for (size_t i = GridNx + GridNy + GridNz - 3; i >= 3; i--)
        {
            ptmKernel2 << < blocks, 1 >> > (dev_r, dev_c0, dev_c1, dev_c3, dev_c5, GridN, i, omega);
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        printf("ptmKernel2 = %f\n", gpuTime);


        cudaEventRecord(start, 0);

        // Вычисляем скалярные произведения
        printf("--- awrRrKernel Starting... ---\n");
        awrRrKernel << < blocks, 1 >> > (dev_Awr, dev_Rr, dev_crr, dev_r, dev_c0, dev_c1, dev_c2, dev_c3, dev_c4, dev_c5, dev_c6, GridN);
        cudaDeviceSynchronize();
        printf("--- awrRrKernel Ended ---\n");

        printf("--- RwRw = Reduce(dev_Rr, GridN); Starting... ---\n");
        double RwRw = Reduce(dev_Rr, GridN);
        printf("--- RwRw = Reduce(dev_Rr, GridN); Ended ---\n");
        printf("--- Aww = Reduce(dev_Awr, GridN); Starting... ---\n");
        double Aww = Reduce(dev_Awr, GridN);
        printf("--- Aww = Reduce(dev_Awr, GridN); Ended ---\n");
        printf("--- ww = Reduce(dev_crr, GridN); Starting... ---\n");
        double ww = Reduce(dev_crr, GridN);
        printf("--- ww = Reduce(dev_crr, GridN); Ended ---\n");
        //printf("RwRw = %lf\n", RwRw);
        //printf("Aww = %lf\n", Aww);
        //printf("ww = %lf\n", ww);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        printf("awrRrKernel + RwRw + Aww + ww = %f\n", gpuTime);


        cudaEventRecord(start, 0);
        if (ww > 0)
        {
            tay = 2 * omega + ww / Aww;
            omega = sqrt(ww / RwRw);
        }

        // Перерасчет dev_u
        uKernel << < blocks, 1 >> > (dev_u, dev_r, GridN, tay);                       

        it++;

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        printf("uKernel = %f\n", gpuTime);
        printf("-------------End of PTM Iteration------------------------\n");
    } while (host_isGreater > 0 && it < 2/*200*/);

    // Копируем массив с результатами вычислений из памяти GPU в ОЗУ
    //cudaMemcpy(host_r, dev_r, sizeInBytesDouble, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_u, dev_u, sizeInBytesDouble, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    // Выводим на консоль массив с результатами вычислений
    //Print3dArrayDouble(host_r);
    //Print3dArrayDouble(host_u);

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

void ReductionTest()
{
    // 1. Вычисляем размер массива данных
    int size = 100000; // Кол-во элементов  
    
    size_t sizeInBytesDouble = size * sizeof(double);// Размер массива double в байтах

    // 2. Выделяем память под массив в ОЗУ    
    double* host_a = (double*)malloc(sizeInBytesDouble);
 
    // 2a Инициализация массива, вычисление суммы элементов массива
    double host_a_sum = 0;
    for (size_t k = 0; k < size; k++)
    {
        host_a[k] = k + 0.2;            
        host_a_sum += host_a[k];
    }

    // 3. Выделяем память под массив на видеокарте    
    double* dev_a = NULL;
    cudaMalloc((void**)&dev_a, sizeInBytesDouble);

    // 4. Копируем массив из ОЗУ в GPU
    cudaMemcpy(dev_a, host_a, sizeInBytesDouble, cudaMemcpyHostToDevice);

    double dev_a_sum = Reduce(dev_a, size);

    printf("host_a_sum = %lf\ndev_a_sum = %lf\n", host_a_sum, dev_a_sum);
}
