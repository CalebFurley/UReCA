// Classification Module Implementation.
#include "classification.h"

// Logistic Regression Implementation.
LogisticRegression::LogisticRegression()
{
	m_weights.setZero(), m_bias = 0.0;
}
LogisticRegression::~LogisticRegression()
{
	m_weights.setZero(); m_bias = 0.0;
}
void LogisticRegression::train(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y, float alpha, int epochs)
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
Eigen::MatrixXd LogisticRegression::predict(const Eigen::MatrixXd& data_X)
{
	//cf Multiply weights on all data then apply bias.
	Eigen::MatrixXd linear_combination = (data_X * m_weights).array() + m_bias;
	//cf Apply the sigmoid function to each element to get probabilities.
	Eigen::MatrixXd predictions = 1.0 / (1.0 + (-linear_combination).array().exp());
	return predictions;
}
double LogisticRegression::score(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y)
{
	Eigen::MatrixXd predictions = predict(data_X);
	//cf Threshold predictions to get binary outcomes.
	predictions = (predictions.array() >= 0.5).cast<double>();

	//cf Calculate accuracy.
	double correct_predictions = (predictions.array() == data_Y.array()).cast<double>().sum();
	double accuracy = correct_predictions / data_Y.rows();

	return accuracy;
}

// KNearest Neighbors Implementation.
KNearestNeighbors::KNearestNeighbors(int k, int number_of_classes) : m_k(k), m_number_of_classes(number_of_classes)
{

}
KNearestNeighbors::~KNearestNeighbors() 
{

}
double KNearestNeighbors::euclidean_distance(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2)
{
	return sqrt((x1 - x2).array().square().sum());
}
void KNearestNeighbors::train(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y)
{
	m_data_X = data_X;
	m_data_Y = data_Y;
}
Eigen::MatrixXd KNearestNeighbors::predict(const Eigen::MatrixXd& data_X)
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
double KNearestNeighbors::score(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y)
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

// TreeNode Implementation. (for decision tree)
TreeNode::TreeNode()
{

}
TreeNode::~TreeNode()
{

}

// Decision Tree Implementation.
DecisionTree::DecisionTree()
{

}
DecisionTree::~DecisionTree()
{

}
void DecisionTree::train()
{

}
void DecisionTree::predict()
{

}
double DecisionTree::score()
{
	return 0.0;
}

// Random Forest Implementation.
RandomForest::RandomForest()
{

}
RandomForest::~RandomForest()
{

}
void RandomForest::train()
{

}
void RandomForest::predict()
{

}
double RandomForest::score()
{
	return 0.0;
}

// Naive Bayes Implementation.
NaiveBayes::NaiveBayes()
{

}
NaiveBayes::~NaiveBayes()
{

}
void NaiveBayes::train()
{

}
void NaiveBayes::predict()
{

}
double NaiveBayes::score()
{
	return 0.0;
}

// Pybind11 Module Generation.
PYBIND11_MODULE(classification, m)
{
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
