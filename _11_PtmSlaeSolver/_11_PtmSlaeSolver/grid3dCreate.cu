#include "helpers.cu"

inline void grid3dCreate() {
	std::cout << std::endl << "----- �������� ������� 3d ����� ----" << std::endl;

	int x, y, z;
	std::cout << "������� ����� ����� ����� �� ��� X: ";
	std::cin >> x;
	std::cout << "������� ����� ����� ����� �� ��� Y: ";
	std::cin >> y;
	std::cout << "������� ����� ����� ����� �� ��� Z: ";
	std::cin >> z;

	Grid3d grid(x, y, z);
	grid.print_dimensions();
}