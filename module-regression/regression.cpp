// Regression Module Implementation.
#include "regression.h"

// Linear Regression Implementation.
LinearRegression::LinearRegression() 
{
	m_weights.setZero(), m_bias = 0.0;
}
LinearRegression::~LinearRegression() 
{
	m_weights.setZero(); m_bias = 0.0;
}
void LinearRegression::train(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y, float alpha, int epochs)
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
Eigen::MatrixXd LinearRegression::predict(const Eigen::MatrixXd& data_X)
{
	//cf Multiply weights on all data then apply bias and return.
	Eigen::MatrixXd predictions = (data_X * m_weights).array() + m_bias;
	return predictions;
}
double LinearRegression::score(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y)
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

// Other regression models go here...

// Pybind11 Module Generation.
PYBIND11_MODULE(regression, m)
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
}
