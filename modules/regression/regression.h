//regression module headers.

#ifndef REGRESSION_H
#define REGRESSION_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

/////////////////////////////////////////////////////////////////////////////

//linear regression prototype.
class LinearRegression
{
private:
//weights and bias
  Eigen::Vector<double, Eigen::Dynamic> m_weights;
  double m_bias;
//hyper parameters
  float m_learning_rate;
  int m_iterations;;
public:
//constructors
  LinearRegression(float learning_rate, int iterations); // add hyperparameters here.
  ~LinearRegression();
//core triple methods
  void train(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y);
  Eigen::MatrixXd predict(const Eigen::MatrixXd& data_X);
  double score(const Eigen::MatrixXd& data_X, const Eigen::MatrixXd& data_Y);
};

/////////////////////////////////////////////////////////////////////////////

#endif