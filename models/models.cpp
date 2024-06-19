#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <math.h>

namespace py = pybind11;

void linear_regression()
{
	std::cout << "Linear Regression" << std::endl;
}

void logistic_regression()
{
	std::cout << "Logistic Regression" << std::endl;
}

void k_nearest_neighbors()
{
	std::cout << "K Nearest Neighbors" << std::endl;
}

void random_forest()
{
	std::cout << "Random Forest" << std::endl;
}

PYBIND11_MODULE(models, m)
{
	m.def("linear_regression", &linear_regression, "Explain model workings here.");
	m.def("k_nearest_neighbors", &k_nearest_neighbors, "Explain model workings here.");
	m.def("logistic_regression", &logistic_regression, "Explain model workings here.");
	m.def("random_forest", &random_forest, "Explain model workings here.");
}
