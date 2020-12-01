#include "cuda_runtime.h"
//#include "cuda.h"
#include "device_launch_parameters.h"
//#include <stdlib.h>
#include <stdio.h>
#include <math.h>
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

#define EPS 0.001

void Print3dArray(int* host_c);
void Print3dArrayDouble(double* host_c);

void PtmTest();

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
    PtmTest();

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
/// ������ �������
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
            printf("r[%d] = %lf; \n",m0, r[m0]);
        }        
    }    
}

__global__ void nevyazkaGreaterEpsKernel(int* isGreater, double* r, unsigned int size, double eps)
{
    printf("----!!!!!!!!!! 170 !!!!!!!! isGreater = %d--------\n", *isGreater);
    if (*isGreater > 0)
    {
        printf("----!!!!!!!!!! 172 !!!!!!!! isGreater = %d| isGreater > 0 ---> return; --------\n", *isGreater);
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
            if (*isGreater > 0)
            {
                printf("----!!!!!!! 195 !!!!!!!!!!! isGreater = %d| isGreater > 0 ---> return; --------\n", *isGreater);
                return;
            }
            atomicExch(isGreater, 1);
            printf("----!!!!!!!!!--- 199 ---!!!!!!!!!r[%d] = %lf; isGreater = %d--------\n", m0, r[m0], *isGreater);
            return;
        }
    }
}

/// <summary>
/// ���, ������ �� 0 �� Nx+Ny+Nz
/// </summary>
/// <param name="c"></param>
/// <param name="size">���-�� ��������� ������� s</param>
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
/// ���, ������ �� Nx+Ny+Nz �� 0
/// </summary>
/// <param name="c"></param>
/// <param name="size">���-�� ��������� ������� s</param>
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

    __syncthreads();
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
    double* host_Rr  = (double*)malloc(sizeInBytesDouble);
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
    cudaMemcpy(dev_Rr,  host_Rr,  sizeInBytesDouble, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_crr, host_crr, sizeInBytesDouble, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_s, host_s, sizeInBytesInt, cudaMemcpyHostToDevice);
    

    // 5. ����������� ��������� ������ CUDA
    dim3 blocks(GridNx, GridNy);
    
    // do...

    // ���������� ������� �������
    nevyazkaKernel <<< blocks, 1 >>> (dev_r, dev_c0, dev_c1, dev_c2, dev_c3, dev_c4, dev_c5, dev_c6, dev_f, dev_u, GridN);
    cudaDeviceSynchronize();

    // �����������, ��������� �� ���� �� ���� ������� ������� ������� ������������ �������� ������    
    int host_isGreater = 0;
    int* dev_isGreater = NULL;
    cudaMalloc((void**)&dev_isGreater, sizeof(int));    
    nevyazkaGreaterEpsKernel << <blocks, 1 >> > (dev_isGreater, dev_r, GridN, EPS);
    cudaDeviceSynchronize();    
    cudaMemcpy(&host_isGreater, dev_isGreater, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("\n-------host_isGreater = %d----------\n", host_isGreater);
    // ... while()

    // 6. ����� ���������
    double omega = 0.05;// �������� ��������� ��������????????????????????????
    
    for (size_t i = 3; i < GridNx + GridNy + GridNz - 3; i++)
    {
        ptmKernel1 << < blocks, 1 >> > (dev_r, dev_c0, dev_c2, dev_c4, dev_c6, GridN, i, omega);
    }
    cudaDeviceSynchronize();

    for (size_t i = GridNx + GridNy + GridNz - 3; i >= 3 ; i--)
    {
        ptmKernel2 << < blocks, 1 >> > (dev_r, dev_c0, dev_c1, dev_c3, dev_c5, GridN, i, omega);
    }

    // ��������� ��������� ������������
    awrRrKernel << < blocks, 1 >> > (dev_Awr, dev_Rr, dev_crr, dev_r, dev_c0, dev_c1, dev_c2, dev_c3, dev_c4, dev_c5, dev_c6, GridN);

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
