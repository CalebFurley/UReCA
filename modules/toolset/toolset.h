//toolset module headers.

#ifndef TOOLSET_H
#define TOOLSET_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

/////////////////////////////////////////////////////////////////////////////

class Scaler
{
private:
//mean and diviation members.
  Eigen::VectorXd m_mean;
  Eigen::VectorXd m_std;
public:
//constructors.
  Scaler();
  ~Scaler();
//scale method.
  void scale(const Eigen::MatrixXd& data_X);
};

/////////////////////////////////////////////////////////////////////////////
/* Possible future implementation.
class Scorer
{
  public:
  template <class Model>
  double score(const Model& model, const Eigen::MatrixXd& data_X, const Eigen::VectorXd& data_y);
};
*/
/////////////////////////////////////////////////////////////////////////////

#endif