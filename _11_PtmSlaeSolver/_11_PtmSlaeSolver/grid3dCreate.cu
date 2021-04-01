#include "grid3d.cu"
#include "DataStore.cpp"

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

	Grid3d<size_t, float> grid(nx, ny, nz, hx, hy, hz);
	grid.print_dimensions();

	DataStore<size_t, float> ds(grid.dimensions);
	ds.data[0] = 4;
	auto d0 = ds.data[0];
	ds.dimensions.Print();
	//ds.Print();
	//ds.InitDataByZeros();
	//ds.Print();
	ds.InitDataByIndexes();
	//ds.Print();
	//ds.SaveToFile("DataStore.txt");

	DataStore<size_t, float>  readedDataStore("DataStore.txt");
	readedDataStore.Print();

}