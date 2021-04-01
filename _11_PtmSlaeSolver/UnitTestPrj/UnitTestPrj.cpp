#include "pch.h"
#include "CppUnitTest.h"
#include "../_11_PtmSlaeSolver/dataStore.cpp"
//#include "../_11_PtmSlaeSolver/grid3d.cu"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTestPrj
{
	TEST_CLASS(UnitTestPrj)
	{
	public:
		
		TEST_METHOD(TestMethod1)
		{
			std::string name = "Bill";
			std::string name2 = "Bill";
			Assert::AreEqual(name, name2);
		}
			

		TEST_METHOD(DataStoreCreation)
		{
			Dim3d<size_t> dimension { 5, 10, 2 };
			DataStore<size_t, float> ds{ dimension };
			
			Assert::AreEqual(ds.dimensions.x, size_t{ 5 });
			Assert::AreEqual(ds.dimensions.y, size_t{ 10 });
			Assert::AreEqual(ds.dimensions.z, size_t{ 2 });
			Assert::AreEqual(ds.dimensions.N, size_t{ 5 * 10 * 2 });
			Assert::AreEqual(ds.dimensions.Nxy, size_t{ 5 * 10 });

			Assert::IsNotNull(ds.data);

			ds.data[0] = 5;
			Assert::AreEqual(ds.data[0], 5.0f);
			
			ds.data[dimension.N-1] = -1.5f;
			Assert::AreEqual(ds.data[dimension.N - 1], -1.5f);

			ds.~DataStore();
			Assert::IsNull(ds.data);
		}
	};
}
