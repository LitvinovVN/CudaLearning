#include <iostream>

/// <summary>
/// Перечисление классов размерностей
/// </summary>
enum class DimTypes
{
	Dim,
	Dim1D,
	Dim2D,
	Dim3D
};

template <typename T>
/// <summary>
/// Значения различных системах координат (базовый класс)
/// </summary>
struct Dim {
	/// <summary>
	/// Тип
	/// </summary>
	DimTypes Type = DimTypes::Dim;
		
	virtual void print() {		
		std::cout << "Dim print()" << std::endl;
	}

	virtual void printType() {
		switch (Type)
		{
		case DimTypes::Dim:
			std::cout << "Type = Dim" << std::endl;
			break;
		case DimTypes::Dim1D:
			std::cout << "Type = Dim1D" << std::endl;
			break;
		case DimTypes::Dim2D:
			std::cout << "Type = Dim2D" << std::endl;
			break;
		case DimTypes::Dim3D:
			std::cout << "Type = Dim3D" << std::endl;
			break;
		default:
			break;
		}		
	}
};


template <typename T>
/// <summary>
/// Значения в одномерной системе координат
/// </summary>
struct Dim1D : Dim<T> {
	T i{}; // Значение по координате i

	Dim1D()
	{
		Type = DimTypes::Dim1D;
	}
		
	void print() override {
		std::cout << "Dim1D print(): " << i << std::endl;		
	}		
};


template <typename T>
/// <summary>
/// Значения в двумерной системе координат
/// </summary>
struct Dim2D : Dim<T> {
	T i{}; // Значение по координате i
	T j{}; // Значение по координате j	

	Dim2D()
	{
		Type = DimTypes::Dim2D;
	}

	void print() {
		std::cout << "Dim2D print(): (" << i << ", " << j << ")" << std::endl;
	}
};


template <typename T>
/// <summary>
/// Значения в трёхмерной системе координат
/// </summary>
struct Dim3D : Dim<T> {
	T i; // Значение по координате i
	T j; // Значение по координате j
	T k; // Значение по координате k

	Dim3D()
	{
		Type = DimTypes::Dim3D;
	}

	void print() {
		std::cout << "Dim3D print(): (" << i << ", " << j << ", " << k << ")" << std::endl;
	}
};

// Псевдонимы
using Dim_size_t   = Dim<size_t>;
using Dim1D_size_t = Dim1D<size_t>;
using Dim2D_size_t = Dim2D<size_t>;
using Dim3D_size_t = Dim3D<size_t>;

using Dim_int   = Dim<int>;
using Dim1D_int = Dim1D<int>;
using Dim2D_int = Dim2D<int>;
using Dim3D_int = Dim3D<int>;

using Dim_float = Dim<float>;
using Dim1D_float = Dim1D<float>;
using Dim2D_float = Dim2D<float>;
using Dim3D_float = Dim3D<float>;