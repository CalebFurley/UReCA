//classification module headers.

#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

/////////////////////////////////////////////////////////////////////////////

//logistic regression prototype.
class LogisticRegression
{
private:
//weights and bias
	Eigen::VectorXd m_weights;
	double m_bias;
//hyper parameters
	float m_learning_rate;
	int m_iterations;
public:
//constructors
	LogisticRegression(float learning_rate, int iterations); //add hyperparameters here.
	~LogisticRegression();
//core triple methods.
	void train(const Eigen::MatrixXd& data_x, const Eigen::MatrixXd& data_y);
	Eigen::MatrixXd predict(const Eigen::MatrixXd& data_X);
	double score(const Eigen::MatrixXd& data_x, const Eigen::MatrixXd& data_y);
};

/////////////////////////////////////////////////////////////////////////////

//knearest neighbors prototype.
class KNearestNeighbors
{
private:
//data members (lazy learning)
	Eigen::MatrixXd m_data_X;
	Eigen::VectorXd m_data_Y;
//hyper parameters.
	int m_number_neighbors = 0;
	int m_number_classes = 0;
//helper methods.
	double euclidean_distance(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2);
public:
//constructors.
	KNearestNeighbors(int number_neighbors, int number_classes);
	~KNearestNeighbors();
//core triple methods.
	void train(const Eigen::MatrixXd& data_x, const Eigen::MatrixXd& data_y);
	Eigen::MatrixXd predict(const Eigen::MatrixXd& data_X);
	double score(const Eigen::MatrixXd& data_x, const Eigen::MatrixXd& data_y);
};

/////////////////////////////////////////////////////////////////////////////

//tree node helper class prototype.
class TreeNode
{
private:
//members go here.
public:
	TreeNode();
	~TreeNode();
};

/////////////////////////////////////////////////////////////////////////////

//decision tree prototype.
class DecisionTree
{
private:
//members go here.
public:
	DecisionTree();
	~DecisionTree();
	void train();
	void predict();
	double score();
};

/////////////////////////////////////////////////////////////////////////////

//random forest prototype.
class RandomForest
{
private:
//members go here.
public:
	RandomForest();
	~RandomForest();
	void train();
	void predict();
	double score();
};

/////////////////////////////////////////////////////////////////////////////

//naive bayes prototype.
class NaiveBayes
{
private:
//members go here.
public:
	NaiveBayes();
	~NaiveBayes();
	void train();
	void predict();
	double score();
};

#endif