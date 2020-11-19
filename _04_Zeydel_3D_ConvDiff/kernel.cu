#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define Nx 20 // размер поля по x
#define Ny 20 // размер поля по y
#define Nz 10 // размерность по z
#define N (Nx*Ny*Nz)
#define hx 1 // шаг по x
#define hy 1 // шаг по y
#define hz 1 // шаг по z
#define lt 10 // время
#define ht 0.1 // шаг по времени
#define sigma 0.5
#define err 0.000000001

#define blocks 1
#define threads ((Nx*Ny + blocks - 1) / blocks)

double O[N], // степень заполненности ячейки
v[N], // компонент вектора скорости
u[N], // компонент вектора скорости
w[N], // компонент вектора скорости
mu[N], // коэфициент диффузии 
Cn[N], // искомое поле
B1[N],
B2[N],
B3[N],
B4[N],
B5[N],
B6[N],
A[N],
F[N]; // ф-я источник

__host__ __device__ void printMatrix(double arr[N]) {
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            printf("%f ", arr[i * Nx + j]);
        }
        printf("\n");
    }
}

__host__ __device__ void printVector(int arr[Nx]) {
    for (int i = 0; i < Nx; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void writeToFile(double arr[N], const char* fileName) {
    FILE* f = fopen(fileName, "w+t");
    if (f) {
        for (int k = 0; k < Nz; k++) {
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    int m0 = i + j * Nx + k * Nx * Ny;
                    fprintf(f, "%f ", arr[m0]);
                }
                fprintf(f, "\n");
            }
        }
    }
    fclose(f);
}

__device__
void calc(double* O, double* u, double* v, double* w, double* mu, double* C, // arrays
    double* A, double* B1, double* B2, double* B3, // values
    double* B4, double* B5, double* B6, double* F, int m0) {

    double B51, B61, B7, B8, B9;

    int m1 = m0 + 1;
    int m2 = m0 - 1;
    int m3 = m0 + Nx;
    int m4 = m0 - Nx;
    int m5 = m0 + Nx * Ny;
    int m6 = m0 - Nx * Ny;
    int m24 = m4 - 1;
    int m26 = m2 - Nx * Ny;
    int m46 = m4 - Nx * Ny;
    int m246 = m24 - Nx * Ny;

    double q1 = (O[m0] + O[m4] + O[m6] + O[m46]) / 4; //заполненность области D
    double q2 = (O[m2] + O[m24] + O[m26] + O[m246]) / 4;
    double q3 = (O[m0] + O[m2] + O[m6] + O[m26]) / 4;
    double q4 = (O[m4] + O[m24] + O[m46] + O[m246]) / 4;
    double q5 = (O[m0] + O[m2] + O[m4] + O[m24]) / 4;
    double q6 = (O[m6] + O[m26] + O[m46] + O[m246]) / 4;
    double q0 = (q1 + q2) / 2;

    //Разностная схема для диффузии-конвекции в канонической форме.
    // ??????????????????????????????????????????????????????????????
    *B1 = q1 * (-(u[m1] + u[m0]) / (4 * hx) + (mu[m1] + mu[m0]) / (2 * hx * hx));
    *B2 = q2 * ((u[m2] + u[m0]) / (4 * hx) + (mu[m2] + mu[m0]) / (2 * hx * hx));
    *B3 = q3 * (-(v[m3] + v[m0]) / (4 * hy) + (mu[m3] + mu[m0]) / (2 * hy * hy));
    *B4 = q4 * ((v[m4] + v[m0]) / (4 * hy) + (mu[m4] + mu[m0]) / (2 * hy * hy));

    B61 = (1 - sigma) * (*B1);
    B7 = (1 - sigma) * (*B2);
    B8 = (1 - sigma) * (*B3);
    B9 = (1 - sigma) * (*B4);

    *B1 = sigma * (*B1);
    *B2 = sigma * (*B2);
    *B3 = sigma * (*B3);
    *B4 = sigma * (*B4);

    *A = q0 / ht + (*B1) + (*B2) + (*B3) + (*B4);
    B51 = q0 / ht - B61 - B7 - B8 - B9;

    *F = B51 * C[m0] + B61 * C[m1] + B7 * C[m2] + B8 * C[m3] + B9 * C[m4];
}

__device__ int max_found = 0;

__global__
void processCalculating(double* O, double* u, double* v, double* w, double* mu, double* C) {
    double A[Nz], B1[Nz], B2[Nz], B3[Nz], B4[Nz], B5[Nz], B6[Nz], F[Nz];
    double t = 0;

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // +1 т.к мы начинаем с 1-го столбца

    //if (i == 1) printf("\nlockIdx: %d %d %d \n", blockIdx.x, blockIdx.y, blockIdx.z);
    //printf("ThreadIdx: %d %d %d \n", threadIdx.x, threadIdx.y, threadIdx.z);

    if (i >= Nx*Ny - 1) return;

    do {
        // рассчитываем новые значения
        for (int k = 1; k < Nz - 1; k++) {
            calc(O, u, v, w, mu, C, &A[k], &B1[k], &B2[k],
                &B3[k], &B4[k], &B5[k], &B6[k], &F[k], i + k * Nx*Ny);
        }

        // пока дельта не достигнет максимальной ошибки
        do {
            atomicCAS(&max_found, 1, 0);
            for (int j = 1; j < 2 * Ny - 3; j++) {
                // рассчитываем новые значения
                int m0 = i + (j - i + 1) * Nx, m1, m2, m3, m4;
                int k = m0 / Nx;
                double w = C[m0];

                if (i > j || (j - i) >= (Nx - 2)) goto l_break;

                m1 = m0 + 1;
                m2 = m0 - 1;
                m3 = m0 + Nx;
                m4 = m0 - Nx;

                C[m0] = (F[k] + B1[k] * C[m1] + B2[k] * C[m2] + B3[k] * C[m3] + B4[k] * C[m4]) / A[k];

                w = fabs(w - C[m0]);
                if (w >= err) {
                    atomicExch(&max_found, 1);
                }
            l_break:
                __syncthreads();
            }
        } while (max_found != 0);

        t += ht;

    } while (t < lt);
}

int main(int argc, char const* argv[]) {
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    printf("Starting...\n");

    for (int k = 0; k < Nz; k++) {
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                int m0 = i + j * Nx + k * Nx * Ny;
                Cn[m0] = 0;
                O[m0] = 0.1;
                mu[m0] = 0.2;
                u[m0] = 0.3;
                v[m0] = 0.4;
                w[m0] = 0.1;
            }
        }
    }

    printf("Initial values filled\n");

    for (int k = 0; k < Nz / 4; k++) {
        for (int i = 1; i < Nx / 4; i++) {
            for (int j = 1; j < Ny / 4; j++) {
                int m0 = i + j * Nx + k * Nx * Ny;
                Cn[m0] = 1;
            }
        }
    }

    printf("Cn values filled\n");

    writeToFile(Cn, "start_cuda.txt");

    double* c_O, * c_u, * c_v, * c_w, * c_mu, * c_C;

    // alloc all arrays
    cudaMalloc(&c_O, N * sizeof(double));
    cudaMalloc(&c_u, N * sizeof(double));
    cudaMalloc(&c_v, N * sizeof(double));
    cudaMalloc(&c_w, N * sizeof(double));
    cudaMalloc(&c_mu, N * sizeof(double));
    cudaMalloc(&c_C, N * sizeof(double));

    printf("Cuda values allocated\n");

    // copy static values
    cudaMemcpy(c_O, O, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_u, u, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_v, v, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_w, w, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_mu, mu, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_C, Cn, N * sizeof(double), cudaMemcpyHostToDevice);

    printf("Cuda values copied\n");

    printf("Trying to calling kernel...\n");
    processCalculating << <blocks, threads >> > (c_O, c_u, c_v, c_w, c_mu, c_C);
    printf("Called:-)\n");
    cudaDeviceSynchronize();
    printf("Synced:-)\n");

    cudaMemcpy(Cn, c_C, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(c_O);
    cudaFree(c_u);
    cudaFree(c_v);
    cudaFree(c_w);
    cudaFree(c_mu);
    cudaFree(c_C);

    writeToFile(Cn, "result_cuda.txt");

    return 0;
}