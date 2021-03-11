#ifndef HELPER_FILE
#define HELPER_FILE

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include "locale.h"
#include <malloc.h>
#include <stdlib.h>
#include <vector>
#include <array>
#include <thread>

using ::std::thread;
using ::std::array;
using ::std::vector;
using ::std::cout;
using ::std::endl;
using ::std::ref;


/// <summary>
/// ����������� � ��������� ������������
/// </summary>
struct Dim3d {
    size_t x;
    size_t y;
    size_t z;
};

/// <summary>
/// �������� ��������� �����
/// </summary>
struct GridFragment3d {
    Dim3d dimensions;
};

/// <summary>
/// ��������� �����
/// </summary>
struct Grid3d {
    Dim3d dimensions{};

    Grid3d(int x, int y, int z) {
        dimensions.x = x;
        dimensions.y = y;
        dimensions.z = z;
    }

    /// <summary>
    /// ������� � ������� ��������� ��������� �����
    /// </summary>
    __device__ __host__
    void print_dimensions()
    {
        printf("����������� ��������� �����: {%d, %d, %d}\n", dimensions.x, dimensions.y, dimensions.z);
    }
};

/// <summary>
/// ���������� �������� � �������
/// </summary>
inline void ShowSystemProperties() {
    std::cout << std::endl;
    std::cout << "---------------- �������� � ������� -----------------" << std::endl;
    std::cout << "���������� ��������� ������� (���� CPU):" << std::thread::hardware_concurrency() << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;
}

//////////////////////////////////////////////
/// <summary>
/// ���������� ��������� �������������
/// </summary>
inline void ShowVideoadapterProperties() {
    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount(&count);
    for (size_t i = 0; i < count; i++)
    {
        cudaGetDeviceProperties(&prop, i);
        printf("������������ ����������:        %s\n", prop.name);
        printf("�������������� �����������:     %d.%d\n", prop.major, prop.minor);
        printf("�������� �������:               %d ���\n", prop.clockRate / 1000);
        printf("���������� ����������� (���.): ");
        if (prop.deviceOverlap)
        {
            printf("���������\n");
        }
        else
        {
            printf("���������\n");
        }
        printf("����-��� ���������� ����: ");
        if (prop.kernelExecTimeoutEnabled)
        {
            printf("�������\n");
        }
        else
        {
            printf("��������\n");
        }
        printf("���������� ����������� DMA �������: %d (1: ����������� ������ + ����, 2: ����������� ������ up + ����������� ������ down + ����)\n", prop.asyncEngineCount);

        printf("------------ ���������� � ������ ---------------\n");
        printf("����� ���������� ������:        %ld ����\n", prop.totalGlobalMem);
        printf("����� ����������� ������:       %ld ����\n", prop.totalConstMem);

        printf("------------ ���������� � ����������������� ---------------\n");
        printf("���������� �����������������:   %d\n", prop.multiProcessorCount);
        printf("���������� �������������� ������ �� 1 ����:   %d ����\n", prop.sharedMemPerBlock);
        printf("���������� �������������� ������ �� 1 ���������������:   %ld ����\n", prop.sharedMemPerMultiprocessor);
        printf("���������� 32�-������ ��������� �� 1 ����:   %d ����\n", prop.regsPerBlock);
        printf("������ warp'�:                  %d\n", prop.warpSize);
        printf("������������ ���������� ����� � �����: %d\n", prop.maxThreadsPerBlock);
        printf("������������ ���������� ����� � �����: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("������������ ������� �����: (%ld, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
}


inline __device__ int returnsPlus2(int num)
{
    return num + 2;
}



inline __global__ void addKernel(int* c, const int* a, const int* b, Grid3d* g)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];

    //g->print_dimensions();
    int n = returnsPlus2(5);
    printf("\n%d\n", n);
}

// Helper function for using CUDA to add vectors in parallel.
inline cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
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

    Grid3d g(10,20,30);

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, size >> > (dev_c, dev_a, dev_b, &g);

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

inline int addWithCudaStart() {
#pragma region addWithCuda
    const int arraySize = 5;
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
#pragma endregion
}

#endif