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

#include "grid3d.cu"
#include "Dim3d.cpp"

using ::std::thread;
using ::std::array;
using ::std::vector;
using ::std::cout;
using ::std::endl;
using ::std::ref;



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


#endif