#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include <iostream>
#include <vector>
#include <math.h>

namespace py = pybind11;

///////////////////////////////////////////////////////////////

// Build linear regression class here.

class LinearRegression
{
private:
	// Constructor Members
	Eigen::MatrixXd m_data_X;
	Eigen::MatrixXd m_data_Y;
	double m_alpha = 0.0;
	int m_epochs = 0;

	// Training Members
	Eigen::MatrixXd m_weights;
	Eigen::MatrixXd m_bias;
	int m_m = 0;
	Eigen::MatrixXd m_fwb;
	double cost = 0.0;
	Eigen::MatrixXd m_d_weights;
	double m_d_bias;


public:
	// Constructor
	LinearRegression()
	{
		std::cout << "Linear Regression object created" << std::endl;
	}

	// Train method
	void train(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y, double alpha, int epochs)
	{
		std::cout << "Training Linear Regression model" << std::endl;

		// Initialze data and hyperparameters
		m_data_X = data_X;
		m_data_Y = data_Y;
		m_alpha = alpha;
		m_epochs = epochs;

		// Initialize weights and bias
		m_weights = Eigen::VectorXd::Zero(m_data_X.cols());
		m_bias = Eigen::VectorXd::Zero(1);
		m_m = m_data_X.rows();

		m_fwb = Eigen::VectorXd::Zero(m_m);

		// Gradient Descent
		for (int i = 0; i < m_epochs; i++)
		{
			// Forward pass
			m_fwb = (m_data_X * m_weights) + m_bias;
			cost = (m_fwb - m_data_Y).squaredNorm() / (2 * m_m);
			
			// Calculate gradients
			m_d_weights = (1/m_m) * (m_data_X.transpose() * (m_fwb - m_data_Y));
			m_d_bias = ((1/m_m) * (m_fwb - m_data_Y)).sum();

			// Update weights and bias
			m_d_weights = m_d_weights * m_alpha;
			m_d_bias = m_d_bias * m_alpha;

			for (int j = 0; j < m_weights.size(); j++)
			{
				m_weights(j) = m_weights(j) - m_d_weights(j);
			}
			m_bias(0) = m_bias(0) - m_d_bias;
		}
	}

	// Predict method
	Eigen::MatrixXd predict(const Eigen::MatrixXd& data_X)
	{
		std::cout << "Predicting with Linear Regression model" << std::endl;

		// Use trained weights and bias to calculate predictions.
		Eigen::MatrixXd data_Y = (data_X * m_weights).array() + m_bias(0);

		return data_Y;
	}

	// Score method
	double score(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y)
	{
		std::cout << "Scoring Linear Regression model" << std::endl;

		// Calculate R^2 score
		Eigen::MatrixXd data_Y_pred = predict(data_X);
		double ss_res = (data_Y - data_Y_pred).squaredNorm();
		Eigen::MatrixXd meanAdjusted = data_Y.array() - data_Y.mean();
		double ss_tot = meanAdjusted.squaredNorm();
		
		double epsilon = 1e-8; // Small value to prevent division by zero
		double r2 = 1 - (ss_res / (ss_tot + epsilon));


		return r2;
	}
};

///////////////////////////////////////////////////////////////


PYBIND11_MODULE(models, m)
{
	// Expose linear regression class
	py::class_<LinearRegression>(m, "LinearRegression")
		.def(py::init<>())
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
