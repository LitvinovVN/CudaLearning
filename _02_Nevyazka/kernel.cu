#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>

#define BLOCK_SIZE 1000

__global__ 
void nevyazkaKernel(double* r, double* rMax, double* f, double* u)
{
    int tid = threadIdx.x;

    r[tid] = f[tid] - u[tid];

    //printf("treadId = %d: r[i] = %lf - %lf = %lf\n",tid, f[tid], u[tid], r[tid]);
    //printf("\n%lf; ", rMax[0]);
    
    __syncthreads();
      


    __shared__ double data[BLOCK_SIZE];    
    data[tid] = r[tid];
    //printf("\ndata[%d]=%lf; ",tid, data[tid]);
    
    //printf("29: treadId = %d: r[tid] = %lf; data[tid] = %lf\n", tid, r[tid], data[tid]);
    
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s>>=1 )
    {
        if (tid < s)
        {
            if (fabs(data[tid]) < fabs(data[tid + s]))
            {
                //printf("38: treadId = %d: r[tid] = %lf; data[tid] = %lf --- data[tid + s] = %lf \n", tid, r[tid], data[tid], data[tid + s]);
                data[tid] = fabs(data[tid + s]);                
            }
        }            

        __syncthreads();               
    }
    
    //printf("42: treadId = %d: r[tid] = %lf; data[tid] = %lf\n", tid, r[tid], data[tid]);

    if (tid < 32)
    {
        if (fabs(data[tid]) < fabs(data[tid + 32]))
            data[tid] = fabs(data[tid + 32]);

        if (fabs(data[tid]) < fabs(data[tid + 16]))
            data[tid] = fabs(data[tid + 16]);

        if (fabs(data[tid]) < fabs(data[tid + 8]))
            data[tid] = fabs(data[tid + 8]);

        if (fabs(data[tid]) < fabs(data[tid + 4]))
            data[tid] = fabs(data[tid + 4]);

        if (fabs(data[tid]) < fabs(data[tid + 2]))
            data[tid] = fabs(data[tid + 2]);

        if (fabs(data[tid]) < fabs(data[tid + 1]))
            data[tid] = fabs(data[tid + 1]);
    }

    if (tid == 0)
    {
        rMax[0] = data[0];
        printf("\n\nrMax[0] = %lf;\n\n", rMax[0]);
    }//*/
}




void init_f(double* f, int array_size)
{
    for (size_t i = 0; i < array_size; i++)
    {
        f[i] = 1000 + i;
        //printf("%.1lf; ", f[i]);
    }
}

void init_u(double* u, double* f, int array_size)
{
    for (size_t i = 0; i < array_size; i++)
    {
        // Генерирует случайное действительное число от -100 до 100
        double delta = ((double)(rand()) / RAND_MAX * 200 - 100);

        u[i] = f[i] + delta;
        //printf("%.1lf; ", r[i]);
    }
}

void init_r(double* r, int array_size)
{
    for (size_t i = 0; i < array_size; i++)
    {
        r[i] = 0;
        //printf("%.1lf; ", r[i]);
    }
}

/// <summary>
/// Вычисляет вектор невязки и его максимальное значение
/// </summary>
/// <param name="r">Вектор невязки (возврат)</param>
/// <param name="rMax">Максимальное значение вектора невязки (возврат)</param>
/// <param name="f">Массив значений функции</param>
/// <param name="u">Массив предварительно-рассчитанных значений функции</param>
/// <param name="array_size">Размер массивов</param>
/// <returns></returns>
cudaError_t calcNevyazkaWithCuda(double* r, double &rMax, double* f, double* u, int array_size)
{
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);



    double* dev_r = 0;
    double* dev_f = 0;
    double* dev_u = 0;
    double* dev_rMax = 0;
    cudaError_t cudaStatus;

    // Выбор GPU для запуска
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    ///////////////////////////////////////////////////////////////

    // Выделение буферов памяти на GPU (2 входа, 1 выход)
    size_t array_size_in_bytes = array_size * sizeof(double);

    cudaStatus = cudaMalloc((void**)&dev_r, array_size_in_bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_f, array_size_in_bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_u, array_size_in_bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_rMax, sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    //////////////////////////////////////////////////////////

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_f, f, array_size_in_bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_u, u, array_size_in_bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    /////////////////////////////////////////////////////////////    

    

    // Launch a kernel on the GPU with one thread for each element.
    nevyazkaKernel << <1, array_size >> > (dev_r, dev_rMax, dev_f, dev_u);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    printf("Time for the kernel: %f ms\n", time);
    
    printf("\n---------------------\n");
    printf("Время выполнения nevyazkaKernel: %f ms\n", time);
    printf("---------------------\n");

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    //////////////////////////////////////////////////////////////

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(r, dev_r, array_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    double* rMaxArray = new double[1];
    cudaStatus = cudaMemcpy(rMaxArray, dev_rMax, sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    

    /*for (int i = 0; i < array_size; i++)
    {
        printf("r[%d] = %lf - %lf = %lf\n", i, f[i], u[i], r[i]);
    }*/


Error:
    cudaFree(dev_r);
    cudaFree(dev_f);
    cudaFree(dev_u);

    return cudaStatus;
}


int main()
{
    // Размер массива
    int array_size = BLOCK_SIZE;

    // Массив известных значений функции
    double* f = new double[array_size];
    // Массив рассчитанных значений функции
    double* u = new double[array_size];
    // Вектор невязки
    double* r = new double[array_size];
    // Максимальное значение невязки
    double rMax = 0;

    // Инициализация массива f значениями от 1000 до (1000 + array_size - 1)
    init_f(f, array_size);

    // Инициализация массива u cоответствующими значениями массива f со случайным отклонением
    init_u(u, f, array_size);

    // Инициализация массива r нулевыми значениями
    init_r(r, array_size);

    // Вычисление вектора невязки и его максимального значения на Cuda
    cudaError_t cudaStatus = calcNevyazkaWithCuda(r, rMax, f, u, array_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "calcNevyazkaWithCuda failed!");
        return 1;
    }        
}