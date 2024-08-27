#include "regression.h"

LinearRegression::LinearRegression(float learning_rate, int iterations)
  : m_learning_rate(learning_rate), m_iterations(iterations)
{
    m_weights.setZero(); 
    m_bias = 0.0;
}

LinearRegression::~LinearRegression() 
{
    m_weights.setZero(); 
    m_bias = 0.0;
}

void LinearRegression::train(const Eigen::MatrixXd& data_x, const Eigen::MatrixXd& data_y)
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
        cost = (features_with_bias - data_y).squaredNorm() / (2 * sample_count);
        delta_weights = (1.0 / sample_count) * (data_x.transpose() * (features_with_bias - data_y));
        delta_bias = ((1.0 / feat_count) * (features_with_bias - data_y)).sum();

        m_weights -= m_learning_rate * delta_weights;
        m_bias -= m_learning_rate * delta_bias;
    }
}

Eigen::MatrixXd LinearRegression::predict(const Eigen::MatrixXd& data_X)
{
    return (data_X * m_weights).array() + m_bias;
}

double LinearRegression::score(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y)
{
    Eigen::MatrixXd predictions = predict(data_X);
    double total_variance = (data_Y.array() - data_Y.mean()).square().sum();
    double residual_variance = (data_Y.array() - predictions.array()).square().sum();
    return 1 - (residual_variance / total_variance);
}

PYBIND11_MODULE(regression, m)
{
    pybind11::class_<LinearRegression>(m, "LinearRegression", "Linear Regression model, used for numerical machine learning tasks.")
        .def(pybind11::init<float, int>())
        .def("train", &LinearRegression::train,
            "Summary:\n"
            "    This method is used to train the model.\n\n"
            "Parameters:\n"
            "    data_X: a numpy array of the features('X') of your dataset.\n"
            "    data_Y: a numpy array of the results('Y') of your dataset.\n"
            "    alpha: the learning rate for training the model.\n"
            "    epochs: the number times training loop will run.\n\n"
            "Returns:\n"
            "    void: this method returns nothing.")
        .def("predict", &LinearRegression::predict,
            "Summary:\n"
            "    This method is used to predict data. Train model first to use properly.\n\n"
            "Parameters:\n"
            "    data_X: a numpy array of the features('X') of your dataset.\n\n"
            "Returns:\n"
            "    np.array: a numpy array of predicted y values for all data samples.")
        .def("score", &LinearRegression::score,
            "Summary:\n"
            "    This method is used to score your model using the R^2 method.\n\n"
            "Parameters:\n"
            "    data_X: a numpy array of the features('X') of your dataset.\n"
            "    data_Y: a numpy array of the results('Y') of your dataset.\n\n"
            "Returns:\n"
            "    double: the R^2 score for the model.");
}