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

#define GridNx 5 // ����������� ��������� ����� �� ��� x
#define GridNy 6 // ����������� ��������� ����� �� ��� y
#define GridNz 10 // ����������� ��������� ����� �� ��� z
#define GridN GridNx*GridNy*GridNz // ��������� ����� ����� ��������� �����
#define GridXY GridNx * GridNy // ����� ����� � ��������� XY, �.�. � ����� ���� �� Z

#define CudaCoresNumber 192 // ���������� ���� cuda (https://geforce-gtx.com/710.html - ��� GT710, ��� ������ ���������� ���������� ��������)

void Print3dArray(int* host_c);
void ConveyorTest();

/// <summary>
/// ���������� ��������� �������������
/// </summary>
void ShowVideoadapterProperties() {
    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount(&count);
    for (size_t i = 0; i < count; i++)
    {
        cudaGetDeviceProperties(&prop, i);
        printf("������������ ����������:        %s\n", prop.name);
        printf("�������������� �����������:     %d.%d\n", prop.major, prop.minor);
        printf("���������� �����������������:   %d\n", prop.multiProcessorCount);
        printf("������ warp'�:                  %d\n", prop.warpSize);
    }

    printf("���������� ���� cuda: %d (�������� �� ������������ � �������������)\n", CudaCoresNumber);
}

/// <summary>
/// ���������� ��������� �����
/// </summary>
void ShowGridProperties()
{
    printf("\n--------------�������������� ��������� �����----------------\n");
    printf("����������� ��������� ����� �� ��� x:                %d\n", GridNx);
    printf("����������� ��������� ����� �� ��� y:                %d\n", GridNy);
    printf("����������� ��������� ����� �� ��� z:                %d\n", GridNz);
    printf("��������� ����� ����� ��������� �����:               %d\n", GridN);
    printf("����� ����� � ��������� XY, �.�. � ����� ���� �� Z:  %d\n", GridXY);
    printf("----------------------------------------------------------\n");
}



int main()
{
    // ��������� ��������� ��������� � �������
    setlocale(LC_CTYPE, "rus");
    // ����������� ���������� ����������
    ShowVideoadapterProperties();
    // ����������� ���������� �����
    ShowGridProperties();     
    // ���� ��������� ����������
    ConveyorTest();
    
    return 0;
}

/// <summary>
/// ������������� ������� �� GPU
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
/// ����������� � 2 ���� �������� � ���������, ����� �������� ������� ����� s
/// </summary>
/// <param name="c"></param>
/// <param name="size">���-�� ��������� ������� s</param>
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
    printf("------------------���� ������������ ����������----------------\n");
    // 1. ��������� ������ ������� ������
    int size = GridN; // ���-�� ���������
    size_t sizeInBytesInt = size * sizeof(int);// ������ ������� � ������

    // 2. �������� ������ ��� ������ � ���
    int* host_c = 0;
    host_c = (int*)malloc(sizeInBytesInt);

    // 3. �������� ������ ��� ������ �� ����������        
    int* dev_c = 0;
    cudaError_t cudaStatus = cudaMalloc((void**)&dev_c, sizeInBytesInt);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "!!!!!!!!!!!!cudaMalloc failed in ConveyorTest()!!!!!!!!!!!!");
        return;
    }
    
    // 4. ����������� ��������� ������ CUDA
    dim3 blocks(GridNx, GridNy);

    // 4. �������������� ������ �� GPU
    initVectorInGpuKernel <<< blocks, 1 >>> (dev_c, GridN);
    cudaDeviceSynchronize();

    // 5. ����� ���������
    int s = 3;
    conveyorKernel <<< blocks, 1 >>> (dev_c, GridN, s);
    cudaDeviceSynchronize();

    // �������� ������ � ������������ ���������� �� ������ GPU � ���
    cudaMemcpy(host_c, dev_c, sizeInBytesInt, cudaMemcpyDeviceToHost);
    
    // ������� �� ������� ������ � ������������ ����������
    Print3dArray(host_c);
    

    // ������� ������ ������
    free(host_c);
    cudaFree(dev_c);

    // ���������� ���������� CUDA
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return;
    }

    printf("--------------���� ������������ ���������� (�����)------------\n");
}



__global__ void conveyorTestKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
