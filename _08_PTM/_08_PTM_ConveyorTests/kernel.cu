#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "locale.h"
#include <malloc.h>
#include <stdlib.h>
using namespace std;

//#define GridNx 1000 // ����������� ��������� ����� �� ��� x
//#define GridNy 1000 // ����������� ��������� ����� �� ��� y
//#define GridNz 1000 // ����������� ��������� ����� �� ��� z
//#define GridN GridNx*GridNy*GridNz // ��������� ����� ����� ��������� �����
//#define GridXY GridNx * GridNy // ����� ����� � ��������� XY, �.�. � ����� ���� �� Z

#define CudaCoresNumber 192 // ���������� ���� cuda (https://geforce-gtx.com/710.html - ��� GT710, ��� ������ ���������� ���������� ��������)
#define ThreadsNumber 9 // �� 1 �� CudaCoresNumber
// 49152 ���� - ������ shared-������ ��� GT 710
#define SharedMemorySize 49152/sizeof(double) // ����������� ������� ������������� ������ ��� ������ ���� XY
#define BlockSizeX ThreadsNumber // ����������� ����� �� X ����� ������ ����� ����� � �����
#define BlockSizeY 10 /*SharedMemorySize/BlockSizeX*/ // ����������� ����� �� Y �� 1 �� CudaCoresNumber

#define GridNx (BlockSizeX + 1) // ����������� ��������� ����� �� ��� x
#define GridNy BlockSizeY // ����������� ��������� ����� �� ��� y
#define GridNz 1 // ����������� ��������� ����� �� ��� z
#define GridN GridNx*GridNy*GridNz // ��������� ����� ����� ��������� �����
#define GridXY GridNx * GridNy // ����� ����� � ��������� XY, �.�. � ����� ���� �� Z

#define EPS 0.001

void Print3dArray(int* host_c);
void Print3dArrayDouble(double* host_c);

void PtmTest();


#pragma region ��������������� �������

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
/// ���, ������ �� 0 �� Nx+Ny+Nz
/// </summary>
/// <param name="c"></param>
/// <param name="size">���-�� ��������� ������� s</param>
/// <param name="s">i+j+k</param>
/// <returns></returns>
__global__ void ptmKernel1(double* r, double* c0, double* c2, double* c4, double* c6, unsigned int size, double omega)
{
    // Compute the offset in each dimension
    const size_t threadX = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Make sure that you are not actually outs
    /*if (threadX >= GridNx - 1 || threadX == 0)
        return;*/

    // Compute the linear index assuming that X,Y then Z memory ordering
    const size_t idx = threadX + 1;

    //printf("202: idx = %d\n", idx);
    int currentY = 1; // 0 - �������, ���� 1
    for (size_t s = 2; s < GridNx + GridNy - 2; s++)
    {       
        
        if (idx + currentY == s && s < GridNy + idx)
        {          
            
            long nodeIndex = idx + BlockSizeX * currentY;

            if (threadX == 0)
            {
                printf("215: idx = %d, ", idx);
                printf("currentY = %d, ", currentY);
                printf("nodeIndex = %d, ", nodeIndex);
                printf("s = %d\n", s);
            }

            long m0 = nodeIndex;

            if (c0[m0] > 0)
            {
                long m2 = m0 - 1;
                long m4 = m0 - GridNx;
                long m6 = m0 - GridXY;

                r[m0] = (omega * (c2[m0] * r[m2] + c4[m0] * r[m4] + c6[m0] * r[m6]) + r[m0]) / ((0.5 * omega + 1) * c0[m0]);

                printf("243: r[%d] = %lf\n", m0, r[m0]);
            }
        }

        currentY++;        
    }
        
}


#pragma endregion Kernels







int main()
{
    // ��������� ��������� ��������� � �������
    setlocale(LC_CTYPE, "rus");
    // ����������� ���������� ����������
    ShowVideoadapterProperties();
    // ����������� ���������� �����
    ShowGridProperties();
    // ���� ��������� ����������
    PtmTest();   

    return 0;
}




void PtmTest()
{
    printf("------------------���� ������������ ����������----------------\n");
    // 1. ��������� ������ ������� ������
    int size = GridN; // ���-�� ���������
    size_t sizeInBytesInt = size * sizeof(int);   // ������ ������� int � ������
    size_t sizeInBytesDouble = size * sizeof(double);// ������ ������� double � ������

    // 2. �������� ������ ��� ������ � ���    
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

    // 2a ������������� ��������
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


    // 3. �������� ������ ��� ������� �� ����������    
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

    // 4. �������� ������� �� ��� � GPU
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


    // 5. ����������� ��������� ������ CUDA        
    int host_isGreater = 0;
    int* dev_isGreater = NULL;
    cudaMalloc((void**)&dev_isGreater, sizeof(int));

    float gpuTime = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    

    printf("-----------------------------------------------------------\n");
    printf("-------------Start of PTM Iteration------------------------\n");

    // ����� ���������    
    double omega = 0.05;
    double tay = 2 * omega;

    printf("--- ptmKernel1 Starting... ---\n");

    cudaEventRecord(start, 0);
    ptmKernel1 << < 1, BlockSizeX >> > (dev_r, dev_c0, dev_c2, dev_c4, dev_c6, GridN, omega);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);        
    printf("ptmKernel1 Time: %lf\n", gpuTime);
    cudaDeviceSynchronize();

    printf("-------------End of PTM Iteration------------------------\n");


    // �������� ������ � ������������ ���������� �� ������ GPU � ���
    cudaMemcpy(host_r, dev_r, sizeInBytesDouble, cudaMemcpyDeviceToHost);    

    cudaDeviceSynchronize();
    // ������� �� ������� ������ � ������������ ����������
    Print3dArrayDouble(host_r);    

    // ������� ������ ������
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

    // ���������� ���������� CUDA
    cudaDeviceReset();

    printf("--------------���� ������������ ���������� (�����)------------\n");
}

