#include <pybind11/pybind11.h>
#include <iostream>
#include <string>

namespace py = pybind11;

void myPrint(std::string message)
{
	std::cout << message << std::endl;
}

PYBIND11_MODULE(myprint, m)
{
	m.def("myPrint", &myPrint, "Prints to the console.");
}




/* ADD FUNCTION IS WRITTEN BELOW

#include <pybind11/pybind11.h>

namespace py = pybind11;

int myAdd(int x, int y)
{
	return x + y;
}

PYBIND11_MODULE(myadd, m)
{
	m.def("myAdd", &myAdd, "Add two numbers.");
}
*/