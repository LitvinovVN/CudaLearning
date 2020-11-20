
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// Конфигурация сетки
const int Nx = 10;
const int Ny = 20;
const int Nz = 30;
const int Nblocks = Nx * Ny * Nz;
// Конфигурация блока в сетке (потоки в 1 блоке)
const int Ndx = 4;
const int Ndy = 3;
const int Ndz = 2;
const int NtreadsInBlock = Ndx * Ndy * Ndz;

const int N = Nblocks * NtreadsInBlock;

cudaError_t addWithCuda(double* c, const double* a, const double* b, unsigned int size);

__global__ void addKernel(double* c, const double* a, const double* b)
{
    int indexXY = threadIdx.x + threadIdx.y * blockDim.x;//Номер потока в текущей плоскости XY текущего блока
    int NumXY = blockDim.x * blockDim.y;//Число потоков в текущей плоскости XY текущего блока
    int indexXYZ = indexXY + NumXY * threadIdx.z;//Номер потока в текущем блоке

    //printf("indexXYZ = %d\n", indexXYZ);

    int indexXYBlock = blockIdx.x + blockIdx.y * gridDim.x;//Номер блока в текущей плоскости XY сетки
    int NumXYBlock = gridDim.x * gridDim.y;//Число блоков в текущей плоскости XY сетки
    int indexXYZBlock = indexXYBlock + NumXYBlock * blockIdx.z;//Номер блока в сетке

    int index = indexXYZBlock * blockDim.x * blockDim.y * blockDim.z + indexXYZ;//глобальный индекс нити

    if (blockIdx.x == 1 && blockIdx.y == 2 && blockIdx.z == 3 && threadIdx.x == 3 && threadIdx.y == 2 && threadIdx.z == 1)
    {
        printf("gridDim.x = %d, gridDim.y = %d, gridDim.z = %d\n",gridDim.x, gridDim.y, gridDim.z);
        printf("blockDim.x = %d, blockDim.y = %d, blockDim.z = %d\n", blockDim.x, blockDim.y, blockDim.z);        
        printf("indexXYZ = %d\n", indexXYZ);                
        printf("indexXYZBlock = %d\n", indexXYZBlock);
        printf("index = %d\n", index);
    }


    //Индекс текущего блока в гриде
    int blockIndex = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;
    //Индекс треда внутри текущего блока
    int ThreadIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;

    //глобальный индекс нити
    int GlobalThreadIndex = blockIndex * blockDim.x * blockDim.y * blockDim.z + ThreadIndex;

    //printf("block: (%d,%d,%d) threads: (%d,%d,%d) blockIndex = %d ThreadIndex = %d GlobalThreadIndex= %d\n ", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, blockIndex, ThreadIndex, GlobalThreadIndex);

    /*if (index != GlobalThreadIndex)
        printf("False\n");*/

    c[GlobalThreadIndex] = a[GlobalThreadIndex] + b[GlobalThreadIndex];
}

int main()
{    
    double* a = new double[N];
    double* b = new double[N];
    double* c = new double[N];
    
    for (size_t z = 0; z < Nz; z++)
    {
        for (size_t y = 0; y < Ny; y++)
        {
            for (size_t x = 0; x < Nx; x++)
            {
                int i = x + y * Nx + z * Nx * Ny;
                a[i] = i;
                b[i] = 2 * i;
            }
        }
    }



    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{%.1lf,%.1lf,%.1lf,%.1lf,%.1lf} + {%.1lf,%.1lf,%.1lf,%.1lf,%.1lf} = {%.1lf,%.1lf,%.1lf,%.1lf,%.1lf}\n",
        a[0], a[1], a[2], a[3], a[4], b[0], b[1], b[2], b[3], b[4], c[0], c[1], c[2], c[3], c[4]);

    for (size_t i = 0; i < N; i++)
    {
        if (a[i] + b[i] == c[i])
            printf("True!!!!!");
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(double *c, const double *a, const double *b, unsigned int size)
{
    double *dev_a = 0;
    double *dev_b = 0;
    double *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.    
    dim3 NxNyNz(Nx, Ny, Nz);
    dim3 NdxNdyNdz(Ndx, Ndy, Ndz);
    addKernel<<<NxNyNz, NdxNdyNdz >>>(dev_c, dev_a, dev_b);

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
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);
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
