# UReCA Summer Project 2024

To generate the Python modules, build the project using CMake and move the generated Python modules into your Python project directory.

Currently, the repository is going through a refactoring phase. The entire project is being reorganized and rewritten for easier expansion and cross-platform targeting (CMake for Linux/Windows/Mac for module generation).

## Project Summary

The UReCA Project is an extension of a research fellowship I completed at the University of Oklahoma that utilized sklearn machine learning models to predict melanoma diagnoses in patients given images of the affected region. The extension allowed me to begin development on a sklearn-type library to gain a better understanding of C++, cross-language development, cross-platform development, and of course, machine learning as a whole.

The library is written in C++ and, as mentioned before, utilizes CMake to generate Python modules that can be used with any Python script. The library supports Python 3.12 and Numpy.

## External Libraries

The external libraries used in the project include pybind11 for Python module generation and Eigen for matrices and general linear algebra functionality. Both libraries integrate perfectly together and support Numpy from Python, leading to a seamless integration with the machine learning workflow.

## Change Log

### Completed Tasks
- Must first get CMake working, then get the correct lib, link it, then write the function and test.
  - Also see if can make a sub-project in Python in this project for testing code all in a single editor.
- CMake is working, and Boost is installed. Now to install pybind11 and test with print project.
  - Once pybind11 is installed, write out the print function, package up and test in the corresponding Python file.
- Get pybind11 working with C++ and Python. Then finish the print function.
- Lay out the structure of the library
  - A models and a tools folder, the library will be split into two parts.
- Build Linear Regression Model as a class
  - A class allows for training and predicting with a single object.
- Build Logistic Regression Model
  - Utilize linear regression as a template.
  - Import a classification dataset, diabetes?
- Build K Nearest Neighbors Model
  - Utilize other models as templates.
  - Build for classification.
- Rework Project Structure
  - Split project into classification and regression.
  - Add headers to help clean up and modularize the current code.
- Full Project Refactor
  - Rename and reorganize all files and modules.
  - Rewrite all prototypes.
  - Add the toolset module.
  - Rewrite README.
  - Implement Scaler in toolset.
  - Rewrite models to match new prototypes.
  - Refactor test environment for Python.
  - Check every file in the repository and clean code/comments.
- Build Scaler
  - Scales data using std and mean.

### Future Tasks
- Build Decision Tree Model
  - Watch videos to learn about the model.
  - Build out in models.cpp.
- Build Random Forest Model.
- Build Naive Bayes Model.