#include "pch.h"
#include "CppUnitTest.h"
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

		TEST_METHOD(TestMethod2)
		{
			std::string name = "Bill";
			std::string name2 = "Bill222";
			Assert::AreEqual(name, name2);
		}
	};
}
