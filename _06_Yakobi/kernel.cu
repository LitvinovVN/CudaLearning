#include "cuda_runtime.h"
#include "cuda.h"
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cstring>
using namespace std;

#define Thread 512

// https://poisk-ru.ru/s24402t10.html

//Raschet priblijennogo znachenija iteracii k+1
__global__ void KernelJacobi(float* deviceA, float* deviceF, float* deviceX0, float* deviceX1, int N)
{
	float temp;

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	deviceX1[i] = deviceF[i];

	for (int j = 0; j < N; j++) {
		if (i != j)
			deviceX1[i] -= deviceA[j + i * N] * deviceX0[j];
		else
			temp = deviceA[j + i * N];
	}

	deviceX1[i] /= temp;
}

//Raschetdeltidlyausloviaostanovki
__global__ void EpsJacobi(float* deviceX0, float* deviceX1, float* delta, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	delta[i] += fabs(deviceX0[i] - deviceX1[i]);
	deviceX0[i] = deviceX1[i];
}

int main()
{
	setlocale(LC_CTYPE, "rus");
	srand(time(NULL));

	float* hostA, * hostX, * hostX0, * hostX1, * hostF, * hostDelta;
	float sum, eps;
	float EPS = 1.e-5;
	int N = 2048;
	float size = N * N;
	int count;

	int Block = (int)ceil((float)N / Thread);

	dim3 Blocks(Block);
	dim3 Threads(Thread);

	int Num_diag = 0.5f * (int)N * 0.3f;

	unsigned int mem_sizeA = sizeof(float) * size;
	unsigned int mem_sizeX = sizeof(float) * (N);

	//Videlenie pamyati na hoste
	hostA = (float*)malloc(mem_sizeA);
	hostF = (float*)malloc(mem_sizeX);
	hostX = (float*)malloc(mem_sizeX);
	hostX0 = (float*)malloc(mem_sizeX);
	hostX1 = (float*)malloc(mem_sizeX);
	hostDelta = (float*)malloc(mem_sizeX);

	//Inicializacija massivov

	for (int i = 0; i < size; i++) {
		hostA[i] = 0.0f;
	}

	for (int i = 0; i < N; i++) {
		hostA[i + i * N] = rand() % 50 + 1.0f * N;
	}

	for (int k = 1; k < Num_diag + 1; k++) {
		for (int i = 0; i < N - k; i++) {
			hostA[i + k + i * N] = rand() % 5;
			hostA[i + (i + k) * N] = rand() % 5;
		}
	}

	for (int i = 0; i < N; i++) {
		hostX[i] = rand() % 50;
		hostX0[i] = 1.0f;
		hostDelta[i] = 0.0f;
	}

	for (int i = 0; i < N; i++) {
		sum = 0.0f;

		for (int j = 0; j < N; j++)
			sum += hostA[j + i * N] * hostX[j];

		hostF[i] = sum;
	}

	for (int i = 0; i < N; i++)
		hostX1[i] = 1.0f;


	//Videleniepamyati na GPU

	float* deviceA, * deviceX0, * deviceX1, * deviceF, * delta;

	cudaMalloc((void**)&deviceA, mem_sizeA);
	cudaMalloc((void**)&deviceF, mem_sizeX);
	cudaMalloc((void**)&deviceX0, mem_sizeX);
	cudaMalloc((void**)&deviceX1, mem_sizeX);
	cudaMalloc((void**)&delta, mem_sizeX);


	//Peredachadannichna GPU
	cudaMemcpy(deviceA, hostA, mem_sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceF, hostF, mem_sizeX, cudaMemcpyHostToDevice);

	cudaMemcpy(deviceX0, hostX0, mem_sizeX, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceX1, hostX1, mem_sizeX, cudaMemcpyHostToDevice);

	count = 0;
	eps = 1.0f;

	// Инициализация и создание Event'овдля определение времени расчета
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Инициализация переменной времени расчета
	float gpuTime = 0.0f;

	// Запуск таймера
	cudaEventRecord(start, 0);

	// Запуск метода Якоби
	while (eps > EPS) {
		count++;
		cudaMemcpy(delta, hostDelta, mem_sizeX, cudaMemcpyHostToDevice);
		KernelJacobi << <Blocks, Threads >> > (deviceA, deviceF, deviceX0, deviceX1, N);
		EpsJacobi << <Blocks, Threads >> > (deviceX0, deviceX1, delta, N);
		cudaMemcpy(hostDelta, delta, mem_sizeX, cudaMemcpyDeviceToHost);
		eps = 0.0f;

		for (int j = 0; j < N; j++) {
			eps += hostDelta[j];
			hostDelta[j] = 0;
		}

		eps = eps / N;
	}

	cudaMemcpy(hostX1, deviceX1, mem_sizeX, cudaMemcpyDeviceToHost);

	// Остановка таймера
	cudaEventRecord(stop, 0);

	// Синхронизация Event'ов
	cudaEventSynchronize(stop);

	// Получение времени расчета как интервала между Event'ами
	cudaEventElapsedTime(&gpuTime, start, stop);

	/*

	std::cout<< "Matrix A." <<std::endl;

	for (inti = 0; i< N; i++) {
	for (int j = 0; j < N; j++) {
	std::cout<<hostA[i * N + j] << " ";
	}

	std::cout<<std::endl;
	}



	std::cout<< "Matrix F." <<std::endl;

	for (inti = 0; i< N; i++) {
	std::cout<<hostF[i] << " ";
	}

	std::cout<<std::endl;



	std::cout<< "Matrix X." <<std::endl;

	for (inti = 0; i< N; i++) {
	std::cout<<hostX[i] << " ";
	}

	std::cout<<std::endl;



	std::cout<< "Result matrix." <<std::endl;

	for (inti = 0; i< N; i++) {
	std::cout<< hostX1[i] << " ";
	}

	std::cout<<std::endl;
	*/

	std::cout << "Размер матрицы A: " << mem_sizeA / 1024.0 / 1024.0 << " MB." << std::endl;
	std::cout << "Расчет завершен." << std::endl;
	std::cout << "Время расчета на GPU: " << gpuTime << " миллисекунд." << std::endl;

	// Удаление Event'ов
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Osvobojdenie pamyati

	cudaFree(deviceA);
	cudaFree(deviceF);
	cudaFree(deviceX0);
	cudaFree(deviceX1);

	free(hostA);
	free(hostF);
	free(hostX0);
	free(hostX1);
	free(hostX);
	free(hostDelta);

	return 0;
}