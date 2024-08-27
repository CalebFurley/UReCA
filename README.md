
# UReCA Summer Project 2024

To generate the python modules. Build the project using cmake and move the generated python modules into your
python project directory.

Currently the repository is going through a refactoring phase. he entire project is being reaorganized and re-wri
tten for easier expansion and cross platform targeting. (cmake for linux/windows/mac for module generation.)

# Project Summary

The UReCA Project is an extension to a research fellowship I completed at the University of Oklahoma that utilized
sklearn machine learning models to predict melanoma diagnoses in patients given images of the affected region. The
extension allowed me to begin development on a sklearn type library to gain a better understanding of c++, cross
language development, cross platform development, and of course machine learning as a whole.

The Library is written in c++ and as mentioned before utilizes cmake to generate python modules when can be used
with any python script. The library suppot Python 3.12 and Numpy.

# External Libraries

The external libraries used in the project include pybind11 for python modules generation and eigen for matrices
and general linear algebra functionality. Both libraries integrate perfectly together and support numpy from python
leading to a seamless integration with the machine learning workflow.

--------------------------------------------------------------------------------------------------------------------

# Change Log
Must first get CMake working, then get the correct lib, link it, then write the function and test.
- Also see if can make a sub project in Python in this project for testing code all in a single editor.
(completed)
--------------------------------------------------------------------------------------------------------------------
CMake is working, and boost is installed. Now to install pybind11 and test with print project.
Once pybind11 is installed, write out the print function, package up and test in the corresponding python file.
(completed)
--------------------------------------------------------------------------------------------------------------------
Get pybind11 working with C++ and Python. Then finish the print function.
(completed)
--------------------------------------------------------------------------------------------------------------------
Lay out the structure of the library
- a models and a tools folder, the library will be split into two parts
(completed)
--------------------------------------------------------------------------------------------------------------------
Build Linear Regression Model as a class
- a class allows to training and predicting with a single object
(completed)
--------------------------------------------------------------------------------------------------------------------
Build Logistic Regression Model
- utilize linear regression as a template
- import a classification dataset, diabetes?
(completed)
--------------------------------------------------------------------------------------------------------------------
Build K Nearest Neighbors Model
- utilize other models as templates
- build for classification
(completed)
--------------------------------------------------------------------------------------------------------------------
Rework Project Structure
- Split project into classification and regression
- Add headers to help cleanup and modulize the current code.
(completed)
--------------------------------------------------------------------------------------------------------------------
Full Project Refactor
- Rename and reorganize all files and modules.
- Rewrite all prototypes
- Add the toolset module
- Rewrite README
- Implement Scaler in toolset
- Rewrite models to match new prototypes
- Refactor test-enviroment for python
- Check every file in repository and clean code/comments
(complete)
--------------------------------------------------------------------------------------------------------------------
  Build Scaler
- scales data using std and mean
--------------------------------------------------------------------------------------------------------------------
Build Decision Tree Model
- watch videos to learn about model
- build out in models.cpp
(future)
--------------------------------------------------------------------------------------------------------------------
Build Random Forest Model
(future)
--------------------------------------------------------------------------------------------------------------------
Build Naive Bayes Model
(future)
--------------------------------------------------------------------------------------------------------------------
