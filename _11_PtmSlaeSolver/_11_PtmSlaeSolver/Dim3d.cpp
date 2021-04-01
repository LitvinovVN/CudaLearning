#ifndef Dim3d_FILE
#define Dim3d_FILE

#include <iostream>

/// <summary>
/// Размерность в трёхмерном пространстве
/// </summary>
template<typename T>
struct Dim3d {
    T x{};
    T y{};
    T z{};

    T N{};
    T Nxy{};

    Dim3d() {
    }

    Dim3d(T X, T Y, T Z) : x(X), y(Y), z(Z) {        
        N = X * Y * Z;
        Nxy = X * Y;
    }

    void Print() {
        std::cout << "Dim3d: ";
        std::cout << "x = " << x << "; ";
        std::cout << "y = " << y << "; ";
        std::cout << "z = " << z << "; ";
        std::cout << "Nxy = " << Nxy << "; ";
        std::cout << "N = " << N << ".\n";
    }
    
};

#endif