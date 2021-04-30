// DistributedSlaeSolverCMake.cpp: определяет точку входа для приложения.
//

#include "DistributedSlaeSolverCMake.h"


using namespace std;

int main()
{
	Dim<size_t> dimArray[4];
	std::vector<Dim<size_t>*> vec3;

	Dim1D<size_t> d1;
	d1.print();
	d1.printType();	

	Dim2D<size_t> d2;
	d2.print();
	d2.printType();
		
	dimArray[0] = d1;
	dimArray[1] = d2;

	dimArray[0].print();
	dimArray[0].printType();
	dimArray[1].print();
	dimArray[1].printType();

	vec3.push_back(&d1);
	vec3.push_back(&d2);

	auto d1c = dynamic_cast<Dim1D<size_t>*>(vec3[0]);
	(*d1c).print();
	(*d1c).printType();

	auto d2c = dynamic_cast<Dim2D<size_t>*>(vec3[1]);
	(*d2c).print();
	(*d2c).printType();

	cout << "Hello CMake." << endl;
	return 0;
}
