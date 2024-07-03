#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <iostream>

using namespace pybind11;
using namespace Eigen;
using namespace std;

/// <summary>
/// Uses zero mean and standard deviation to scale data.
/// </summary>
class StandardScaler
{
public:
	/// <summary>
	/// Scales the data using zero mean and standard deviation.
	/// </summary>
	/// <param name="dataset">: the data to be scaled.</param>
	void scale(MatrixXd& dataset)
	{
		// diabetes_train_x = (diabetes_train_x - np.mean(diabetes_train_x)) / np.std(diabetes_train_x)  #  < -------------- - Use this math to build
		// diabetes_test_x = (diabetes_test_x - np.mean(diabetes_test_x)) / np.std(diabetes_test_x)      #  < -------------- - scaler for tools module.
		//
		//TODO // write this method and test the scaler in python against the sci-kit scaler.
	}
};

// Expose models to python module here.
PYBIND11_MODULE(tools, m)
{
	class_<StandardScaler>(m, "StandardScaler", "Uses zero mean and standard deviation to scale data.")
		.def(init<>())
		.def("scale", &StandardScaler::scale,
			"Summary:\n"
			"	Scales the data using zero mean and standard deviation.\n\n"
			"Parameters:\n"
			"	numpy-array: the data to be scaled.\n\n"
			"Returns:\n"
			"	void: this method returns nothing.");
}
