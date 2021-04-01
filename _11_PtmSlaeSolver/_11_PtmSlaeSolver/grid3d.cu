#ifndef GRID3D_FILE
#define GRID3D_FILE

#include "helpers.cu"
#include "Dim3d.cpp"


/// <summary>
/// Фрагмент трёхмерной сетки
/// </summary>
struct GridBlock3d {
    /// <summary>
    /// Индекс первого узла фрагмента сетки,
    /// т.е. узла с координатами {0, 0, 0}
    /// в системе координат фрагмента
    /// </summary>
    size_t idxStart{};

    /// <summary>
    /// Размерность фрагмента сетки
    /// </summary>
    Dim3d<size_t> dimensions;
};

/// <summary>
/// Трёхмерная сетка
/// </summary>
template<typename Tindx, typename Tdata>
struct Grid3d {
    /// <summary>
    /// Размерность расчетной сетки (количество узлов по пространственным координатам)
    /// </summary>
    Dim3d<Tindx> dimensions{};

    /// <summary>
    /// Шаги расчетной сетки по пространственным координатам
    /// </summary>
    Dim3d<float> h{ 1, 1, 1 };

    Grid3d(Tindx nx, Tindx ny, Tindx nz, Tdata hx, Tdata hy, Tdata hz) {
        dimensions = Dim3d<Tindx>(nx,ny,nz);
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