#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include <iostream>
#include <vector>
#include <math.h>

namespace py = pybind11;
using namespace Eigen;


/////////////////////////////
// Linear Regression Class
class LinearRegression
{
private:
	// Training Members
	Eigen::Vector<double, Dynamic> m_weights;
	double m_bias;

public:
	// Constructor and Destructor
	LinearRegression() { std::cout << "Linear Regression object created" << std::endl; }
	~LinearRegression() { m_weights.setZero(); m_bias = 0.0; }

	// Training method
	void train(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y, float learning_rate, int epochs)
	{	 
		// Declare and init training variables.
		double cost = 0.0;
		Eigen::Index feat_count = data_X.cols();
		Eigen::Index sample_count = data_X.rows();
		Eigen::Vector<double, Dynamic> fwb; fwb.setZero();
		Eigen::Vector<double, Dynamic> dw; dw.setZero();
		double db = 0.0;

		// Set weights and bias to zero.
		m_weights.resize(feat_count);
		m_weights.setZero();
		m_bias = 0.0;
		
		// Perform Gradiant Descent
		for (int epoch = 0; epoch < epochs; ++epoch)
		{
			std::cout << "Calculating linear function across all samples." << std::endl;
			// Calculate linear function across all samples.									// <----------------- DEBUGGING HERE!!!!
			fwb = (data_X * m_weights); std::cout << "Passed 1";
			fwb = fwb.array() + m_bias; std::cout << "Passed 2";

			std::cout << "Calculating cost with current function." << std::endl;
			// Calculate cost with current function.
			cost = (fwb - data_Y).squaredNorm();
			cost /= 2 * sample_count;

			std::cout << "Calculating gradient for current function." << std::endl;
			// Calculate gradient for current function.
			dw = (1.0 / (double)sample_count) * (data_X.transpose() * (fwb - data_Y));				// <----------------- DEBUGGING HERE!!!!
			std::cout << "dw = " << dw << std::endl;
			db = ((1.0 / (double)feat_count) * (fwb - data_Y)).sum();
			std::cout << "db =" << db << std::endl;

			std::cout << "Updating weights and bias." << std::endl;
			// Update weights and bias
			for (int j = 0; j < m_weights.size(); j++)
			{
				m_weights(j) = m_weights(j) - learning_rate * dw(j);
			}
			m_bias = m_bias - learning_rate * db;

			std::cout << "Printing current epoch results." << std::endl;
			// Print weights and bias for debugging..								<------------------------------ DEBUGGING HERE!
			std::cout << "weights" << m_weights << std::endl;
			std::cout << "bias" << m_bias << std::endl;
		}
		
		std::cout << "Model Successfully trained." << std::endl;
	}

	////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////

	// Predict method
	Eigen::MatrixXd predict(const Eigen::MatrixXd& data_X)
	{
		// Write this method.
		return data_X;
	}



	// Score method
	double score(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y)
	{
		// Write this method.
		return 0.0;
	}

};//lin-reg


///////////////////////////////
// Pybind11 Expose Modules
PYBIND11_MODULE(models, m)
{
	// Linear Regression
	py::class_<LinearRegression>(m, "LinearRegression")
		.def(py::init<>())
		.def("train", &LinearRegression::train)
		.def("predict", &LinearRegression::predict)
		.def("score", &LinearRegression::score);

	//TODO Logistic Regression
}
