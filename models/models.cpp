#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

using namespace pybind11;
using namespace Eigen;
using namespace std;

/// <summary>
/// Linear Regression model, used for numerical machine learning tasks.
/// </summary>
class LinearRegression
{
private:
	//cf Weights and bias training members.
	Vector<double, Dynamic> m_weights;
	double m_bias;

public:
	//cf Constructor and destructor.
	LinearRegression() { m_weights.setZero(), m_bias = 0.0; }
	~LinearRegression() { m_weights.setZero(); m_bias = 0.0; }

	/// <summary>
	/// This method is used to train the model.
	/// </summary>
	/// <param name="data_X">: a numpy array of the features('X') of your dataset.</param>
	/// <param name="data_Y">: a numpy array of the results('Y') of your dataset.</param>
	/// <param name="alpha">: the learning rate for training the model.</param>
	/// <param name="epochs">: the number times training loop will run.</param>
	void train(const MatrixXd& data_X, const MatrixXd& data_Y, float alpha, int epochs)
	{	 
		//cf Declare and init training variables.
		double cost = 0.0;
		Index feat_count = data_X.cols();
		Index sample_count = data_X.rows();
		Vector<double, Dynamic> fwb; fwb.setZero();
		Vector<double, Dynamic> dw; dw.setZero();
		double db = 0.0;

		//cf Set weights and bias to zero.
		m_weights.resize(feat_count);
		m_weights.setZero();
		m_bias = 0.0;
		
		//cf Perform Gradiant Descent
		for (int epoch = 0; epoch < epochs; ++epoch)
		{	
			//cf Calculate linear function across all samples.
			fwb = (data_X * m_weights);
			fwb = fwb.array() + m_bias;

			//cf Calculate cost with current function.
			cost = (fwb - data_Y).squaredNorm();
			cost /= 2 * sample_count;

			//cf Calculate gradient for current function.
			dw = (1.0 / (double)sample_count) * (data_X.transpose() * (fwb - data_Y));
			db = ((1.0 / (double)feat_count) * (fwb - data_Y)).sum();

			//cf Update weights and bias
			for (int j = 0; j < m_weights.size(); j++)
			{
				m_weights(j) = m_weights(j) - alpha * dw(j);
			}
			m_bias = m_bias - alpha * db;
		}
	}

	/// <summary>
	/// This method is used to predict data. Train model first to use properly.
	/// </summary>
	/// <param name="data_X">: a numpy array of the features('X') of your dataset.</param>
	/// <returns>np.array: a numpy array of predicted y values for all data samples.</returns>
	MatrixXd predict(const MatrixXd& data_X)
	{
		//cf Multiply weights on all data then apply bias and return.
		MatrixXd predictions = (data_X * m_weights).array() + m_bias;
		return predictions;
	}

	/// <summary>
	/// This method is used to score your model using the R^2 method.
	/// </summary>
	/// <param name="data_X">: a numpy array of the features('X') of your dataset.</param>
	/// <param name="data_Y">:  a numpy array of the results('Y') of your dataset.</param>
	/// <returns>double: the R^2 score for the model.</returns>
	double score(const MatrixXd& data_X, const MatrixXd& data_Y)
	{
		//cf Use model to get predictions.
		MatrixXd predictions = predict(data_X);

		//cf Calculate the total & residual sum of squares.
		double total_variance = (data_Y.array() - data_Y.mean()).square().sum();
		double residual_variance = (data_Y.array() - predictions.array()).square().sum();

		//cf Calculate R^2 score.
		double r2_score = 1 - (residual_variance / total_variance);

		//cf Return the R^2 score.
		return r2_score;
	}
};

/// <summary>
/// Logistic Regression model, used for classification machine learning tasks.
/// </summary>
class LogisticRegression
{
private:
	//cf Weights and bias training members.
	VectorXd m_weights;
	double m_bias;

public:
	//cf Constructor and destructor.
	LogisticRegression() { m_weights.setZero(), m_bias = 0.0; }
	~LogisticRegression() { m_weights.setZero(); m_bias = 0.0; }

	/// <summary>
	/// This method is used to train the model.
	/// </summary>
	/// <param name="data_X">: a numpy array of the features('X') of your dataset.</param>
	/// <param name="data_Y">: a numpy array of the results('Y') of your dataset.</param>
	/// <param name="alpha">: the learning rate for training the model.</param>
	/// <param name="epochs">: the number times training loop will run.</param>
	void train(const MatrixXd& data_X, const MatrixXd& data_Y, float alpha, int epochs)
	{
		//cf Local training variables.
		double cost = 0.0;
		Index feat_count = data_X.cols();
		Index sample_count = data_X.rows();
		Vector<double, Dynamic> fwb; fwb.setZero();
		Vector<double, Dynamic> dw; dw.setZero();
		double db = 0.0;

		//cf Set weights and bias to zero.
		m_weights.resize(feat_count);
		m_weights.setZero();
		m_bias = 0.0;

		//cf Perform gradient descent.
		for (int epoch = 0; epoch < epochs; ++epoch)
		{
			//cf Calculate linear function across all samples.
			fwb = (data_X * m_weights).array() + m_bias;
			VectorXd sigmoid = 1.0 / (1.0 + (-fwb).array().exp());

			//cf Calculate cost with current function.
			cost = -(data_Y.array() * fwb.array().log() + (1 - data_Y.array()) * (1 - fwb.array()).log()).mean();

			//cf Calculate gradient for current function.
			dw = (1.0 / sample_count) * (data_X.transpose() * (sigmoid - data_Y));
			db = (1.0 / sample_count) * (sigmoid - data_Y).sum();

			//cf Update weights and bias
			for (int j = 0; j < m_weights.size(); j++)
			{
				m_weights(j) = m_weights(j) - alpha * dw(j);
			}
			m_bias = m_bias - alpha * db;
		}
	}

	/// <summary>
	/// This method is used to predict data. Train model first to use properly.
	/// </summary>
	/// <param name="data_X">: a numpy array of the features('X') of your dataset.</param>
	/// <returns>np.array: a numpy array of predicted y values for all data samples.</returns>
	MatrixXd predict(const MatrixXd& data_X)
	{
		//cf Multiply weights on all data then apply bias.
		MatrixXd linear_combination = (data_X * m_weights).array() + m_bias;
		//cf Apply the sigmoid function to each element to get probabilities.
		MatrixXd predictions = 1.0 / (1.0 + (-linear_combination).array().exp());
		return predictions;
	}

	/// <summary>
	/// This method is used to score your model using the R^2 method.
	/// </summary>
	/// <param name="data_X">: a numpy array of the features('X') of your dataset.</param>
	/// <param name="data_Y">:  a numpy array of the results('Y') of your dataset.</param>
	/// <returns>double: the R^2 score for the model.</returns>
	double score(const MatrixXd& data_X, const MatrixXd& data_Y)
	{
		MatrixXd predictions = predict(data_X);
		//cf Threshold predictions to get binary outcomes.
		predictions = (predictions.array() >= 0.5).cast<double>();

		//cf Calculate accuracy.
		double correct_predictions = (predictions.array() == data_Y.array()).cast<double>().sum();
		double accuracy = correct_predictions / data_Y.rows();

		return accuracy;
	}
};

// KNN goes here..				Due: Monday, August 2.
class KNearestNeighbors
{
private:
	//cf Member variables go here.

	//cf Used to calculate distance between two points.
	float single_euclidean_distance(float x1, float x2)
	{
		float distance = sqrt( ((x1 - x2) * (x1 - x2)) );
		return distance;
	}

public:
	void train();
	void predict();
	void score();


};

// Decision Tree goes here..	Due: Friday, August 9.

// Random Forest goes here..	Due: Friday, August 9.

// Naive Bayes goes here..		Due: Friday, August 16.

// Write the final report..		Due: Friday, August 30. *** very important

/* Expose models to python module here.
*/
PYBIND11_MODULE(models, m)
{
	// Linear Regression Class.
	class_<LinearRegression>(m, "LinearRegression", "Linear Regression model, used for numerical machine learning tasks.")
		.def(init<>())
		.def("train", &LinearRegression::train,
			"Summary:\n"
			"	This method is used to train the model.\n\n"
			"Parameters:\n"
			"	data_X: a numpy array of the features('X') of your dataset.\n"
			"	data_Y: a numpy array of the results('Y') of your dataset.\n"
			"	alpha:	the learning rate for training the model.\n"
			"	epochs: the number times training loop will run.\n\n"
			"Returns:\n"
			"	void: this method returns nothing.")
		.def("predict", &LinearRegression::predict,
			"Summary:\n"
			"	This method is used to predict data. Train model first to use properly.\n\n"
			"Parameters:\n"
			"	data_X: a numpy array of the features('X') of your dataset.\n\n"
			"Returns:\n"
			"	np.array: a numpy array of predicted y values for all data samples.")
		.def("score", &LinearRegression::score,
			"Summary:\n"
			"	This method is used to score your model using the R^2 method.\n\n"
			"Parameters:\n"
			"	data_X: a numpy array of the features('X') of your dataset.\n"
			"	data_Y: a numpy array of the results('Y') of your dataset.\n\n"
			"Returns:\n"
			"	double: the R^2 score for the model.");

	// Logistic Regression Class.
	class_<LogisticRegression>(m, "LogisticRegression", "Logistic Regression model, used for classification machine learning tasks.")
		.def(init<>())
		.def("train", &LogisticRegression::train,
			"Summary:\n"
			"	This method is used to train the model.\n\n"
			"Parameters:\n"
			"	data_X: a numpy array of the features('X') of your dataset.\n"
			"	data_Y: a numpy array of the results('Y') of your dataset.\n"
			"	alpha:	the learning rate for training the model.\n"
			"	epochs: the number times training loop will run.\n\n"
			"Returns:\n"
			"	void: this method returns nothing.")
		.def("predict", &LogisticRegression::predict,
			"Summary:\n"
			"	This method is used to predict data. Train model first to use properly.\n\n"
			"Parameters:\n"
			"	data_X: a numpy array of the features('X') of your dataset.\n\n"
			"Returns:\n"
			"	np.array: a numpy array of predicted y values for all data samples.")
		.def("score", &LogisticRegression::score,
			"Summary:\n"
			"	This method is used to score your model using the R^2 method.\n\n"
			"Parameters:\n"
			"	data_X: a numpy array of the features('X') of your dataset.\n"
			"	data_Y: a numpy array of the results('Y') of your dataset.\n\n"
			"Returns:\n"
			"	double: the R^2 score for the model.");
}
