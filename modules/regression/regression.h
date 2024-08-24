// Regression Module Headers

#ifndef REGRESSION_H
#define REGRESSION_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

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