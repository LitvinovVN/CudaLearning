#include "grid3d.cu"

inline void grid3dCreate() {
	std::cout << std::endl << "----- �������� ������� 3d ����� ----" << std::endl;

	int nx, ny, nz;
	std::cout << "������� ����� ����� ����� �� ��� X: ";
	std::cin >> nx;
	std::cout << "������� ����� ����� ����� �� ��� Y: ";
	std::cin >> ny;
	std::cout << "������� ����� ����� ����� �� ��� Z: ";
	std::cin >> nz;

	float hx, hy, hz;	
	std::cout << "������� ��� ����� �� ��� X: ";
	std::cin >> hx;
	std::cout << "������� ��� ����� �� ��� Y: ";
	std::cin >> hy;
	std::cout << "������� ��� ����� �� ��� Z: ";
	std::cin >> hz;

	Grid3d grid(nx, ny, nz, hx, hy, hz);
	grid.print_dimensions();
}