#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

/// <summary>
/// Linear Regression model, used for numerical machine learning tasks.
/// </summary>
class LinearRegression
{
private:
	//cf Weights and bias training members.
	Eigen::Vector<double, Eigen::Dynamic> m_weights;
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
	void train(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y, float alpha, int epochs)
	{	 
		//cf Declare and init training variables.
		double cost = 0.0;
		Eigen::Index feat_count = data_X.cols();
		Eigen::Index sample_count = data_X.rows();
		Eigen::Vector<double, Eigen::Dynamic> fwb; fwb.setZero();
		Eigen::Vector<double, Eigen::Dynamic> dw; dw.setZero();
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
	Eigen::MatrixXd predict(const Eigen::MatrixXd& data_X)
	{
		//cf Multiply weights on all data then apply bias and return.
		Eigen::MatrixXd predictions = (data_X * m_weights).array() + m_bias;
		return predictions;
	}

	/// <summary>
	/// This method is used to score your model using the R^2 method.
	/// </summary>
	/// <param name="data_X">: a numpy array of the features('X') of your dataset.</param>
	/// <param name="data_Y">:  a numpy array of the results('Y') of your dataset.</param>
	/// <returns>double: the R^2 score for the model.</returns>
	double score(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y)
	{
		//cf Use model to get predictions.
		Eigen::MatrixXd predictions = predict(data_X);

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
	Eigen::VectorXd m_weights;
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
	void train(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y, float alpha, int epochs)
	{
		//cf Local training variables.
		double cost = 0.0;
		Eigen::Index feat_count = data_X.cols();
		Eigen::Index sample_count = data_X.rows();
		Eigen::Vector<double, Eigen::Dynamic> fwb; fwb.setZero();
		Eigen::Vector<double, Eigen::Dynamic> dw; dw.setZero();
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
			Eigen::VectorXd sigmoid = 1.0 / (1.0 + (-fwb).array().exp());

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
	Eigen::MatrixXd predict(const Eigen::MatrixXd& data_X)
	{
		//cf Multiply weights on all data then apply bias.
		Eigen::MatrixXd linear_combination = (data_X * m_weights).array() + m_bias;
		//cf Apply the sigmoid function to each element to get probabilities.
		Eigen::MatrixXd predictions = 1.0 / (1.0 + (-linear_combination).array().exp());
		return predictions;
	}

	/// <summary>
	/// This method is used to score your model using the R^2 method.
	/// </summary>
	/// <param name="data_X">: a numpy array of the features('X') of your dataset.</param>
	/// <param name="data_Y">:  a numpy array of the results('Y') of your dataset.</param>
	/// <returns>double: the R^2 score for the model.</returns>
	double score(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y)
	{
		Eigen::MatrixXd predictions = predict(data_X);
		//cf Threshold predictions to get binary outcomes.
		predictions = (predictions.array() >= 0.5).cast<double>();

		//cf Calculate accuracy.
		double correct_predictions = (predictions.array() == data_Y.array()).cast<double>().sum();
		double accuracy = correct_predictions / data_Y.rows();

		return accuracy;
	}
};

// KNN goes here..				Due: Monday, August 6.
class KNearestNeighbors
{
private:
	//cf Member Variables for data, note no weights or bias for knn.
	Eigen::MatrixXd m_data_X;
	Eigen::VectorXd m_data_Y;
	int m_k = 0; //cf number of neighbors.
	int m_number_of_classes = 0;

	//cf Used to calculate distance between two points.
	double euclidean_distance(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2)
	{
		return sqrt( (x1 - x2).array().square().sum());
	}

public:
	//cf Constructor and destructor.
	KNearestNeighbors(int k, int number_of_classes) : m_k(k), m_number_of_classes(number_of_classes) {}
	~KNearestNeighbors() {}

	void train(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y)
	{
		m_data_X = data_X;
		m_data_Y = data_Y;
	}

	Eigen::MatrixXd predict(const Eigen::MatrixXd& data_X)
	{
		// Stores all predictions.
		Eigen::MatrixXd predictions(data_X.rows(), 1);

		// For all inputs, run through full KNN algorithm.
		for (int i = 0; i < data_X.rows(); ++i)
		{
			// Vector pair for index and calculated distance(cost)
			std::vector<std::pair<double, int>> distances;

			// Compute distance between input(i) and points(j)
			for (int j = 0; j < m_data_X.rows(); ++j)
			{
				double dist = euclidean_distance(data_X.row(i), m_data_X.row(j));
				distances.push_back(std::make_pair(dist, j));
			}

			// Sort the distances.
			std::sort(distances.begin(), distances.end());

			// Get the closest k neighbors.
			std::vector<int> neighbors;
			for (int k = 0; k < m_k; ++k)
			{
				neighbors.push_back(m_data_Y(distances[k].second));
			}

			// Create vector to store counts for voting.
			std::vector<int> class_count(m_number_of_classes, 0);
			for (int k = 0; k < neighbors.size(); ++k)
			{
				class_count[static_cast<int>(neighbors[k])]++;
			}

			// Majority Vote for prediction.
			int predicted_class = -1;
			int max_count = -1;
			for (int c = 0; c < class_count.size(); ++c)
			{
				if (class_count[c] > max_count)
				{
					max_count = class_count[c];
					predicted_class = c;
				}
			}

			// Set predictions[i] with the prediction for that point.
			predictions(i, 0) = predicted_class;
		}

		return predictions;
	}

	double score(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y)
	{
		// Predict the labels for the input data
		Eigen::MatrixXd predictions = predict(data_X);

		// Count the number of correct predictions
		int correct_predictions = 0;
		for (int i = 0; i < data_Y.rows(); ++i)
		{
			if (predictions(i, 0) == data_Y(i, 0))
			{
				correct_predictions++;
			}
		}

		// Calculate accuracy
		double accuracy = static_cast<double>(correct_predictions) / data_Y.rows();
		return accuracy;
	}
};

//////////////////////////////////////////////////////////////////////////////////////////////

// Decision Tree goes here..	Due: Monday, August 12.
class DecisionTree
{
private:
	// members go here.

public:
	DecisionTree();
	~DecisionTree();

	void train();
	void predict();
	double score();
};

//////////////////////////////////////////////////////////////////////////////////////////////

// Random Forest goes here..	Due: Friday, August 16.

// Naive Bayes goes here..		Due: Friday, August ??.

// Refactor project.			Due: August 23.
// * cmake files
// * code structure
// * comment structure
// * class implementation
// * function naming & implementation

// Write the final report..		Due: Friday, August 30. *** very important

/* Expose models to python module here.
*/
PYBIND11_MODULE(models, m)
{
	// Linear Regression Class.
	pybind11::class_<LinearRegression>(m, "LinearRegression", "Linear Regression model, used for numerical machine learning tasks.")
		.def(pybind11::init<>())
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
	pybind11::class_<LogisticRegression>(m, "LogisticRegression", "Logistic Regression model, used for classification machine learning tasks.")
		.def(pybind11::init<>())
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

	// K Nearest Neighbors Class.
	pybind11::class_<KNearestNeighbors>(m, "KNearestNeighbors", "K Nearest Model, used for classification machine learning tasks.")
		.def(pybind11::init<int, int>())
		.def("train", &KNearestNeighbors::train,
			"stubby."
		)
		.def("predict", &KNearestNeighbors::predict,
			"stubby."
		)
		.def("score", &KNearestNeighbors::score,
			"stubby."
		);
}
