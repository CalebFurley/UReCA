//classification module implementation.
#include "classification.h"

//logistic Regression Implementation.
LogisticRegression::LogisticRegression(float learning_rate, int iterations)
	: m_learning_rate(learning_rate), m_iterations(iterations)
{
	m_weights.setZero();
	m_bias = 0.0;
}

//logistic regression destructor.
LogisticRegression::~LogisticRegression()
{
	m_weights.setZero();
	m_bias = 0.0;
}

//logistic regression training method implementation.
void LogisticRegression::train(const Eigen::MatrixXd& data_x, const Eigen::MatrixXd& data_y)
{
//declare local training variables.
	double cost; 																								//cost of the model (error)
	Eigen::Index feat_count;																 	  //number of features in the dataset (columns)
	Eigen::Index sample_count; 																	//number of samples in the dataset (rows)
	Eigen::Vector<double, Eigen::Dynamic> features_with_bias;   //the linear equation including weights and bias
	Eigen::Vector<double, Eigen::Dynamic> delta_weights; 				//the weights when being updated through gradient descent
	double delta_bias;			 																		//the bias when being updated through gradient descent

//initialize member variables.
	cost = 0.0;
	feat_count = data_x.cols();
	sample_count = data_x.rows();
	features_with_bias.setZero();
	delta_weights.setZero();
	delta_bias = 0.0;

//initialize member variables.
	m_weights.resize(feat_count);
	m_weights.setZero();
	m_bias = 0.0;

//perform gradient descent.
	for (int iteration = 0; iteration < m_iterations; ++iteration)
	{
	//calculate linear function across all samples.
		features_with_bias = (data_x * m_weights).array() + m_bias;
		Eigen::VectorXd sigmoid = 1.0 / (1.0 + (-features_with_bias).array().exp());

	//calculate cost with current function.
		cost = -(data_y.array() * features_with_bias.array().log() + (1 - data_y.array()) * (1 - features_with_bias.array()).log()).mean();

	//clculate gradient for current function.
		delta_weights = (1.0 / sample_count) * (data_x.transpose() * (sigmoid - data_y));
		delta_bias = (1.0 / sample_count) * (sigmoid - data_y).sum();

	//update weights and bias.
		for (int j = 0; j < m_weights.size(); j++)
		{
			m_weights(j) = m_weights(j) - m_learning_rate * delta_weights(j);
		}
		m_bias = m_bias - m_learning_rate * delta_bias;
	}
}

//logistic regression prediction method implementation.
Eigen::MatrixXd LogisticRegression::predict(const Eigen::MatrixXd& data_X)
{
//multiply weights on all data then apply bias.
	Eigen::MatrixXd linear_combination = (data_X * m_weights).array() + m_bias;
//apply the sigmoid function to each element to get probabilities.
	Eigen::MatrixXd predictions = 1.0 / (1.0 + (-linear_combination).array().exp());
	return predictions;
}

//logistic regression score method implementation.
double LogisticRegression::score(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y)
{
	Eigen::MatrixXd predictions = predict(data_X);
//threshold predictions to get binary outcomes.
	predictions = (predictions.array() >= 0.5).cast<double>();

//calculate and return accuracy.
	double correct_predictions = (predictions.array() == data_Y.array()).cast<double>().sum();
	double accuracy = correct_predictions / data_Y.rows();
	return accuracy;
}

//knearst neighbors contstructor implementation.
KNearestNeighbors::KNearestNeighbors(int number_neighbors, int number_classes) 
	: m_number_neighbors(number_neighbors), m_number_classes(number_classes)
{
}

//knearest neighbors destructor implementation.
KNearestNeighbors::~KNearestNeighbors() 
{
}

//euclidean distance helper method implementation.
double KNearestNeighbors::euclidean_distance(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2)
{
	return sqrt((x1 - x2).array().square().sum());
}

//knearest neighbors training method implementation.
void KNearestNeighbors::train(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y)
{
	//lazy training, just store the data.
	m_data_X = data_X;
	m_data_Y = data_Y;
}

//knearest neighbors prediction method implementation.
Eigen::MatrixXd KNearestNeighbors::predict(const Eigen::MatrixXd& data_X)
{
//stores all predictions.
	Eigen::MatrixXd predictions(data_X.rows(), 1);

//for all inputs, run through full KNN algorithm.
	for (int i = 0; i < data_X.rows(); ++i)
	{
	//vector pair for index and calculated distance(cost)
		std::vector<std::pair<double, int>> distances;

	//compute distance between input(i) and points(j)
		for (int j = 0; j < m_data_X.rows(); ++j)
		{
			double dist = euclidean_distance(data_X.row(i), m_data_X.row(j));
			distances.push_back(std::make_pair(dist, j));
		}

	//sort the distances.
		std::sort(distances.begin(), distances.end());

	//get the closest k neighbors.
		std::vector<int> neighbors;
		for (int k = 0; k < m_number_neighbors; ++k)
		{
			neighbors.push_back(m_data_Y(distances[k].second));
		}

	//create vector to store counts for voting.
		std::vector<int> class_count(m_number_classes, 0);
		for (int k = 0; k < neighbors.size(); ++k)
		{
			class_count[static_cast<int>(neighbors[k])]++;
		}

	//majority Vote for prediction.
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

	//set predictions[i] with the prediction for that point.
		predictions(i, 0) = predicted_class;
	}
//return predictions.
	return predictions;
}

//knearest neighbors score method implementation.
double KNearestNeighbors::score(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y)
{
//predict the labels for the input data
	Eigen::MatrixXd predictions = predict(data_X);

//count the number of correct predictions
	int correct_predictions = 0;
	for (int i = 0; i < data_Y.rows(); ++i)
	{
		if (predictions(i, 0) == data_Y(i, 0))
		{
			correct_predictions++;
		}
	}
//calculate and return accuracy
	double accuracy = static_cast<double>(correct_predictions) / data_Y.rows();
	return accuracy;
}

//treeNode Implementation. (for decision tree)
TreeNode::TreeNode()
{

}
TreeNode::~TreeNode()
{

}

//decision tree implementation.
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

//random forest implementation.
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

//naive bayes implementation.
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

//pybind11 module generation.
PYBIND11_MODULE(classification, m)
{
//logistic regression class.
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

//knearest neighbors class.
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
