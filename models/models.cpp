#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include <iostream>
#include <math.h>

namespace py = pybind11;

///////////////////////////////////////////////////////////////

// Build linear regression class here.

class LinearRegression
{
private:
	// Members
	Eigen::MatrixXd m_data_X;
	Eigen::MatrixXd m_data_Y;
	double m_alpha;
	int m_epochs;

public:
	// Constructor
	LinearRegression(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y, double alpha, int epochs)
		:m_data_X(data_X), m_data_Y(data_Y), m_alpha(alpha), m_epochs(epochs)
	{
		std::cout << "Linear Regression object created" << std::endl;
		std::cout << "Train X = " << data_X << std::endl;
		std::cout << "Train Y = " << data_Y << std::endl;
		std::cout << "Alpha = " << alpha << std::endl;
		std::cout << "Epochs = " << epochs << std::endl;
	}

	// Train method
	void train()
	{
		std::cout << "Training Linear Regression model" << std::endl;
	}

	// Predict method
	Eigen::MatrixXd predict(const Eigen::MatrixXd& data_X)
	{
		std::cout << "Predicting with Linear Regression model" << std::endl;
		return data_X;
	}

	// Score method
	double score(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y)
	{
		std::cout << "Scoring Linear Regression model" << std::endl;
		return 0.0;
	}
};

///////////////////////////////////////////////////////////////


PYBIND11_MODULE(models, m)
{
	py::class_<LinearRegression>(m, "LinearRegression")
		.def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&, int, int>())
		.def("train", &LinearRegression::train)
		.def("predict", &LinearRegression::predict)
		.def("score", &LinearRegression::score);
}

///////////////////////////////////////////////////////////////


/*
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
*/
/*
m.def("k_nearest_neighbors", &k_nearest_neighbors, "Explain model workings here.");
m.def("logistic_regression", &logistic_regression, "Explain model workings here.");
m.def("random_forest", &random_forest, "Explain model workings here.");
*/
