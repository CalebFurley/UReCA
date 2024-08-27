#ifndef TOOLSET_H
#define TOOLSET_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

class Scaler
{
private:
  Eigen::VectorXd m_mean;
  Eigen::VectorXd m_std;
public:
  Scaler();
  ~Scaler();
  void scale(const Eigen::MatrixXd& data_X);
};

#endif