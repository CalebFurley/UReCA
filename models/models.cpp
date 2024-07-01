#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include <iostream>
#include <vector>
#include <math.h>

namespace py = pybind11;
using namespace Eigen;
using namespace std;


/////////////////////////////
// Linear Regression Class
class LinearRegression
{
private:
	// Training Members
	Vector<double, Dynamic> m_weights;
	double m_bias;

public:
	// Constructor and Destructor
	LinearRegression() {}
	~LinearRegression() { m_weights.setZero(); m_bias = 0.0; }

	// Training method
	void train(const MatrixXd& data_X, const MatrixXd& data_Y, float learning_rate, int epochs)
	{	 
		// Declare and init training variables.
		double cost = 0.0;
		Index feat_count = data_X.cols();
		Index sample_count = data_X.rows();
		Vector<double, Dynamic> fwb; fwb.setZero();
		Vector<double, Dynamic> dw; dw.setZero();
		double db = 0.0;

		// Set weights and bias to zero.
		m_weights.resize(feat_count);
		m_weights.setZero();
		m_bias = 0.0;
		
		// Perform Gradiant Descent
		for (int epoch = 0; epoch < epochs; ++epoch)
		{	
			// Calculate linear function across all samples.
			fwb = (data_X * m_weights);
			fwb = fwb.array() + m_bias;

			// Calculate cost with current function.
			cost = (fwb - data_Y).squaredNorm();
			cost /= 2 * sample_count;

			// Calculate gradient for current function.
			dw = (1.0 / (double)sample_count) * (data_X.transpose() * (fwb - data_Y));
			db = ((1.0 / (double)feat_count) * (fwb - data_Y)).sum();

			// Update weights and bias
			for (int j = 0; j < m_weights.size(); j++)
			{
				m_weights(j) = m_weights(j) - learning_rate * dw(j);
			}
			m_bias = m_bias - learning_rate * db;
		}
	}


	// Predict method
	MatrixXd predict(const MatrixXd& data_X)
	{
		// Multiply weights on all data then apply bias and return.
		MatrixXd predictions = (data_X * m_weights).array() + m_bias;
		return predictions;
	}



	// Score method
	double score(const MatrixXd& data_X, const MatrixXd& data_Y)
	{
		// Use model to get predictions.
		MatrixXd predictions = predict(data_X);

		// Calculate the total sum of squares.
		double total_variance = (data_Y.array() - data_Y.mean()).square().sum();

		// Calculate the residual sum of squares.
		double residual_variance = (data_Y.array() - predictions.array()).square().sum();

		// Calculate R^2 score.
		double r2_score = 1 - (residual_variance / total_variance);

		// Return the R^2 score.
		return r2_score;
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
