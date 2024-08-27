//regression module implementaion.
#include "regression.h"

//linear regression constructor.
LinearRegression::LinearRegression(float learning_rate, int iterations)
  : m_learning_rate(learning_rate), m_iterations(iterations)
{
	m_weights.setZero(); 
	m_bias = 0.0;
}

//linear regression destructor.
LinearRegression::~LinearRegression() 
{
	m_weights.setZero(); 
	m_bias = 0.0;
}

//linear regression training method implementation.
void LinearRegression::train(const Eigen::MatrixXd& data_x, const Eigen::MatrixXd& data_y)
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

//perform Gradiant Descent.
	for (int iteration = 0; iteration < m_iterations; ++iteration)
	{
	//calculate linear function across all samples.
		features_with_bias = (data_x * m_weights);
		features_with_bias = features_with_bias.array() + m_bias;

	//calculate cost with current function.
		cost = (features_with_bias - data_y).squaredNorm();
		cost /= 2 * sample_count;

	//calculate gradient for current function.
		delta_weights = (1.0 / (double)sample_count) * (data_x.transpose() * (features_with_bias - data_y));
		delta_bias = ((1.0 / (double)feat_count) * (features_with_bias - data_y)).sum();

	//update weights and bias.
		for (int j = 0; j < m_weights.size(); j++)
		{
			m_weights(j) = m_weights(j) - m_learning_rate * delta_weights(j);
		}
		m_bias = m_bias - m_learning_rate * delta_bias;
	}
}

//linear regression prediction method.
Eigen::MatrixXd LinearRegression::predict(const Eigen::MatrixXd& data_X)
{
//multiply weights on all data then apply bias and return.
	Eigen::MatrixXd predictions;
	predictions = (data_X * m_weights).array() + m_bias;
	return predictions;
}

//linear regression scoring method. uses r2.
double LinearRegression::score(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y)
{
//use model to get predictions.
	Eigen::MatrixXd predictions = predict(data_X);

//calculate and return r2 score.
	double total_variance = (data_Y.array() - data_Y.mean()).square().sum();
	double residual_variance = (data_Y.array() - predictions.array()).square().sum();
	double r2_score = 1 - (residual_variance / total_variance);
	return r2_score;
}

//other regression models go here..

//pybind11 module generation.
PYBIND11_MODULE(regression, m)
{
//linear regression class.
	pybind11::class_<LinearRegression>(m, "LinearRegression", "Linear Regression model, used for numerical machine learning tasks.")
		.def(pybind11::init<float, int>())
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
}
