//cf Regression Header File

#ifndef REGRESSION_H
#define REGRESSION_H

#include <Eigen/Dense>

class LinearRegression
{
private:
	Eigen::Vector<double, Eigen::Dynamic> m_weights;
	double m_bias;
public:
    LinearRegression();
    ~LinearRegression();
    void train(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y, float alpha, int epochs);
    Eigen::MatrixXd predict(const Eigen::MatrixXd& data_X);
    double score(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y);
};

#endif