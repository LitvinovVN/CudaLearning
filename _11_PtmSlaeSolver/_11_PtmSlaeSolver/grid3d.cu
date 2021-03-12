#ifndef GRID3D_FILE
#define GRID3D_FILE

#include "helpers.cu"

/// <summary>
/// Размерность в трёхмерном пространстве, size_t
/// </summary>
struct Dim3d {
    size_t x;
    size_t y;
    size_t z;
};

/// <summary>
/// Размерность в трёхмерном пространстве, float
/// </summary>
struct Dim3df {
    float x;
    float y;
    float z;
};

/// <summary>
/// Фрагмент трёхмерной сетки
/// </summary>
struct GridFragment3d {
    Dim3d dimensions;
};

/// <summary>
/// Трёхмерная сетка
/// </summary>
struct Grid3d {
    /// <summary>
    /// Размерность расчетной сетки (количество узлов по пространственным координатам)
    /// </summary>
    Dim3d dimensions{ 20, 30, 40 };

    /// <summary>
    /// Шаги расчетной сетки по пространственным координатам
    /// </summary>
    Dim3df h{ 1, 1, 1 };

    Grid3d(int nx, int ny, int nz, float hx, float hy, float hz) {
        dimensions.x = nx;
        dimensions.y = ny;
        dimensions.z = nz;
        h.x = hx;
        h.y = hy;
        h.z = hz;
    }

    /// <summary>
    /// Выводит в консоль параметры трёхмерной сетки
    /// </summary>
    __device__ __host__
    void print_dimensions()
    {
        printf("Размерность расчетной сетки: {%lld, %lld, %lld}\n", dimensions.x, dimensions.y, dimensions.z);
        printf("Шаги расчетной сетки: {%f, %f, %f}\n", h.x, h.y, h.z);
    }
};

#endif