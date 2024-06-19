#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include<iostream>
#include <math.h>

namespace py = pybind11;

void min_max_scaler()
{
	std::cout << "Min/Max Scaler" << std::endl;
}

PYBIND11_MODULE(tools, m)
{
	m.def("min_max_scaler", &min_max_scaler, "Explain model workings here.");
}
