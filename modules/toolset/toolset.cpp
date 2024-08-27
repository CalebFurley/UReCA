#include "toolset.h"

Scaler::Scaler() 
{
  m_mean.setZero();
  m_std.setZero();
}

Scaler::~Scaler() 
{
  m_mean.setZero();
  m_std.setZero();
}

void Scaler::scale(const Eigen::MatrixXd& data_X)
{
  std::cout << "Scaling data..." << std::endl;
}

PYBIND11_MODULE(toolset, m)
{
  pybind11::class_<Scaler>(m, "Scaler", "Scaler class used to scale data.")
    .def(pybind11::init<>())
    .def("scale", &Scaler::scale, "");
}
