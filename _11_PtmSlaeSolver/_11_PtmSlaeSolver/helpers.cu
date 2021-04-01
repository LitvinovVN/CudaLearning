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
/// Отображает сведения о системе
/// </summary>
inline void ShowSystemProperties() {
    std::cout << std::endl;
    std::cout << "---------------- Сведения о системе -----------------" << std::endl;
    std::cout << "Количество доступных потоков (ядер CPU):" << std::thread::hardware_concurrency() << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;
}

//////////////////////////////////////////////
/// <summary>
/// Отображает параметры видеоадаптера
/// </summary>
inline void ShowVideoadapterProperties() {
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


#endif