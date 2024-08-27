#include "classification.h"

LogisticRegression::LogisticRegression(float learning_rate, int iterations)
	: m_learning_rate(learning_rate), m_iterations(iterations)
{
	m_weights.setZero();
	m_bias = 0.0;
}

LogisticRegression::~LogisticRegression()
{
	m_weights.setZero();
	m_bias = 0.0;
}

void LogisticRegression::train(const Eigen::MatrixXd& data_x, const Eigen::MatrixXd& data_y)
{
	double cost;
	Eigen::Index feat_count = data_x.cols();
	Eigen::Index sample_count = data_x.rows();
	Eigen::Vector<double, Eigen::Dynamic> features_with_bias(feat_count);
	Eigen::Vector<double, Eigen::Dynamic> delta_weights(feat_count);
	double delta_bias = 0.0;

	m_weights.resize(feat_count);
	m_weights.setZero();
	m_bias = 0.0;

	for (int iteration = 0; iteration < m_iterations; ++iteration)
	{
		features_with_bias = (data_x * m_weights).array() + m_bias;
		Eigen::VectorXd sigmoid = 1.0 / (1.0 + (-features_with_bias).array().exp());

		cost = -(data_y.array() * features_with_bias.array().log() + (1 - data_y.array()) * (1 - features_with_bias.array()).log()).mean();

		delta_weights = (1.0 / sample_count) * (data_x.transpose() * (sigmoid - data_y));
		delta_bias = (1.0 / sample_count) * (sigmoid - data_y).sum();

		for (int j = 0; j < m_weights.size(); j++)
		{
			m_weights(j) = m_weights(j) - m_learning_rate * delta_weights(j);
		}
		m_bias = m_bias - m_learning_rate * delta_bias;
	}
}

Eigen::MatrixXd LogisticRegression::predict(const Eigen::MatrixXd& data_X)
{
	Eigen::MatrixXd linear_combination = (data_X * m_weights).array() + m_bias;
	Eigen::MatrixXd predictions = 1.0 / (1.0 + (-linear_combination).array().exp());
	return predictions;
}

double LogisticRegression::score(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y)
{
	Eigen::MatrixXd predictions = predict(data_X);
	predictions = (predictions.array() >= 0.5).cast<double>();

	double correct_predictions = (predictions.array() == data_Y.array()).cast<double>().sum();
	double accuracy = correct_predictions / data_Y.rows();
	return accuracy;
}

KNearestNeighbors::KNearestNeighbors(int number_neighbors, int number_classes) 
	: m_number_neighbors(number_neighbors), m_number_classes(number_classes)
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
	Eigen::MatrixXd predictions(data_X.rows(), 1);

	for (int i = 0; i < data_X.rows(); ++i)
	{
		std::vector<std::pair<double, int>> distances;

		for (int j = 0; j < m_data_X.rows(); ++j)
		{
			double dist = euclidean_distance(data_X.row(i), m_data_X.row(j));
			distances.push_back(std::make_pair(dist, j));
		}

		std::sort(distances.begin(), distances.end());

		std::vector<int> neighbors;
		for (int k = 0; k < m_number_neighbors; ++k)
		{
			neighbors.push_back(m_data_Y(distances[k].second));
		}

		std::vector<int> class_count(m_number_classes, 0);
		for (int k = 0; k < neighbors.size(); ++k)
		{
			class_count[static_cast<int>(neighbors[k])]++;
		}

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
		predictions(i, 0) = predicted_class;
	}
	return predictions;
}

double KNearestNeighbors::score(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y)
{
	Eigen::MatrixXd predictions = predict(data_X);

	int correct_predictions = 0;
	for (int i = 0; i < data_Y.rows(); ++i)
	{
		if (predictions(i, 0) == data_Y(i, 0))
		{
			correct_predictions++;
		}
	}
	double accuracy = static_cast<double>(correct_predictions) / data_Y.rows();
	return accuracy;
}

TreeNode::TreeNode()
{

}

TreeNode::~TreeNode()
{

}

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

PYBIND11_MODULE(classification, m)
{
	pybind11::class_<LogisticRegression>(m, "LogisticRegression", "Logistic Regression model, used for classification machine learning tasks.")
		.def(pybind11::init<float, int>())
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
