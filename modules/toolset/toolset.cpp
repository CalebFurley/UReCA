#include "toolset.h"

// Scaler Implementation.
Scaler::Scaler() 
{
  m_mean.setZero(), m_std.setZero();
}
Scaler::~Scaler() 
{
  m_mean.setZero(), m_std.setZero();
}
void Scaler::scale(const Eigen::MatrixXd& data_X)
{
  // Implement scaling here. in place.
  // diabetes_train_x = (diabetes_train_x - np.mean(diabetes_train_x)) / np.std(diabetes_train_x)
  std::cout << "Scaling data..." << std::endl;
}
 
 

// Pybind11 Module Generation.
PYBIND11_MODULE(toolset, m)
{
	// Scaler Class
  pybind11::class_<Scaler>(m, "Scaler", "Scaler class used to scale data.")
    .def(pybind11::init<>())
    .def("scale", &Scaler::scale,"");
}
