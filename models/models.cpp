#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <iostream>

using namespace pybind11;
using namespace Eigen;
using namespace std;

/// <summary>
/// Linear regression model. Used for classic regression problems.
/// </summary>
class LinearRegression
{
private:
	//training-members
	Vector<double, Dynamic> m_weights;
	double m_bias;

public:
	LinearRegression() { m_weights.setZero(), m_bias = 0.0; }
	~LinearRegression() { m_weights.setZero(); m_bias = 0.0; }

	/// <summary>
	/// This method is used to predict Y values on a data set. To use
	/// correctly, this model must already be trained.
	/// </summary>
	/// <param name="data_X">is a numpy matrix of the features of your dataset.</param>
	/// <param name="data_Y">is a numpy matrix of the outcomes of your dataset.</param>
	/// <param name="learning_rate">is the learning rate or alpha for training.</param>
	/// <param name="epochs">is the number of runs the model will train on the data.</param>
	/// <returns>a numpy vector of predicted y values for all data samples.</returns>
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

	/// <summary>
	/// This method is used to predict outcomes or 'Y' values on a data set. To use
	/// correctly, this model must already be trained.
	/// </summary>
	/// <param name="data_X">is a numpy matrix of the features of your dataset.</param>
	/// <returns>a numpy vector of predicted y values for all data samples.</returns>
	MatrixXd predict(const MatrixXd& data_X)
	{
		// Multiply weights on all data then apply bias and return.
		MatrixXd predictions = (data_X * m_weights).array() + m_bias;
		return predictions;
	}

	/// <summary>
	/// This method is used to score your model using the R^2 method.
	/// </summary>
	/// <param name="data_X">is the features, or 'X' values of your dataset.</param>
	/// <param name="data_Y">is the real test results or 'Y' values of your dataset.</param>
	/// <returns>a double which is your model's R^2 score.</returns>
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
};

/// <summary>
/// Logistic Regression model. Used for classification problems.
/// </summary>
class LogisticRegression
{
private:
	//training-members
	VectorXd m_weights;
	double m_bias;

public:
	LogisticRegression() { m_weights.setZero(), m_bias = 0.0; }
	~LogisticRegression() { m_weights.setZero(); m_bias = 0.0; }

	/// <summary>
	/// This method is used to predict Y values on a data set. To use
	/// correctly, this model must already be trained.
	/// </summary>
	/// <param name="data_X">is a numpy matrix of the features of your dataset.</param>
	/// <param name="data_Y">is a numpy matrix of the outcomes of your dataset.</param>
	/// <param name="learning_rate">is the learning rate or alpha for training.</param>
	/// <param name="epochs">is the number of runs the model will train on the data.</param>
	/// <returns>a numpy vector of predicted y values for all data samples.</returns>
	void train(const MatrixXd& data_X, const MatrixXd& data_Y, float learning_rate, int epochs)
	{
		//TODO write out the logistic regression model's training method here.
	}

	/// <summary>
	/// This method is used to predict outcomes or 'Y' values on a data set. To use
	/// correctly, this model must already be trained.
	/// </summary>
	/// <param name="data_X">is a numpy matrix of the features of your dataset.</param>
	/// <returns>a numpy vector of predicted y values for all data samples.</returns>
	MatrixXd predict(const MatrixXd& data_X)
	{
		// Multiply weights on all data then apply bias and return.
		MatrixXd predictions = (data_X * m_weights).array() + m_bias;
		return predictions;
	}

	/// <summary>
	/// This method is used to score your model using the R^2 method.
	/// </summary>
	/// <param name="data_X">is the features, or 'X' values of your dataset.</param>
	/// <param name="data_Y">is the real test results or 'Y' values of your dataset.</param>
	/// <returns>a double which is your model's R^2 score.</returns>
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
};

// Expose models to python module here.
PYBIND11_MODULE(models, m)
{
	class_<LinearRegression>(m, "LinearRegression")
		.def(init<>())
		.def("train", &LinearRegression::train)
		.def("predict", &LinearRegression::predict)
		.def("score", &LinearRegression::score);

	class_<LogisticRegression>(m, "LogisticRegression")
		.def(init<>())
		.def("train", &LogisticRegression::train)
		.def("predict", &LogisticRegression::predict)
		.def("score", &LogisticRegression::score);
}
