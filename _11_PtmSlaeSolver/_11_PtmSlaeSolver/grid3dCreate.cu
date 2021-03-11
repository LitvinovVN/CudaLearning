#include "helpers.cu"

inline void grid3dCreate() {
	std::cout << std::endl << "----- Создание объекта 3d сетки ----" << std::endl;

	int x, y, z;
	std::cout << "Введите число узлов сетки по оси X: ";
	std::cin >> x;
	std::cout << "Введите число узлов сетки по оси Y: ";
	std::cin >> y;
	std::cout << "Введите число узлов сетки по оси Z: ";
	std::cin >> z;

	Grid3d grid(x, y, z);
	grid.print_dimensions();
}