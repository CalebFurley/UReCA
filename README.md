
# UReCA Summer Project 2024
	This is a Library built in regards to my summer UReCA fellowship. It is a Python module for machine learning split
	into two parts. First being the models themselves which can be imported with "import models" and the second being
	the tools which can be imported with "import tools".

	The library is writen in C++ and uses pybind11 to generate the python modules. As of right now the modules can be
	found inside of the python-testing directory. You will need all the files in that directory besides the .py script
	used for testing.

	The library is developed for Python 3.11, if you are using Python 3.12, you will need to install the previous
	version of Python to use the library.
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
	Build Decision Tree Model
	- watch videos to learn about model
	- build out in models.cpp
	(# CURRENT #)
--------------------------------------------------------------------------------------------------------------------
	Build Random Forest Model
	(future)
--------------------------------------------------------------------------------------------------------------------
	Build Naive Bayes Model
	(future)
--------------------------------------------------------------------------------------------------------------------