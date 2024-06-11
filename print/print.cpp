#include <iostream>
#include <string>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void println(std::string message)
{
	std::cout << message << std::endl;
}

PYBIND11_MODULE(print, m)
{
	m.def("println", &println, "Prints a message to the console.");
}

// Follow tutorial and get my python module working here.