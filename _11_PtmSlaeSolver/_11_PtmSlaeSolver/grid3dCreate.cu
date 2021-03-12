#include "grid3d.cu"

inline void grid3dCreate() {
	std::cout << std::endl << "----- Создание объекта 3d сетки ----" << std::endl;

	int nx, ny, nz;
	std::cout << "Введите число узлов сетки по оси X: ";
	std::cin >> nx;
	std::cout << "Введите число узлов сетки по оси Y: ";
	std::cin >> ny;
	std::cout << "Введите число узлов сетки по оси Z: ";
	std::cin >> nz;

	float hx, hy, hz;	
	std::cout << "Введите шаг сетки по оси X: ";
	std::cin >> hx;
	std::cout << "Введите шаг сетки по оси Y: ";
	std::cin >> hy;
	std::cout << "Введите шаг сетки по оси Z: ";
	std::cin >> hz;

	Grid3d grid(nx, ny, nz, hx, hy, hz);
	grid.print_dimensions();
}