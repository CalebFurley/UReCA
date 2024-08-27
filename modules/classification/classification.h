#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

class LogisticRegression
{
private:
  Eigen::VectorXd m_weights;
  double m_bias;
  float m_learning_rate;
  int m_iterations;
public:
  LogisticRegression(float learning_rate, int iterations);
  ~LogisticRegression();
  void train(const Eigen::MatrixXd& data_x, const Eigen::MatrixXd& data_y);
  Eigen::MatrixXd predict(const Eigen::MatrixXd& data_X);
  double score(const Eigen::MatrixXd& data_x, const Eigen::MatrixXd& data_y);
};

class KNearestNeighbors
{
private:
  Eigen::MatrixXd m_data_X;
  Eigen::VectorXd m_data_Y;
  int m_number_neighbors;
  int m_number_classes;
  double euclidean_distance(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2);
public:
	KNearestNeighbors(int number_neighbors, int number_classes);
  ~KNearestNeighbors();
  void train(const Eigen::MatrixXd& data_x, const Eigen::MatrixXd& data_y);
  Eigen::MatrixXd predict(const Eigen::MatrixXd& data_X);
  double score(const Eigen::MatrixXd& data_x, const Eigen::MatrixXd& data_y);
};

class TreeNode
{
public:
  TreeNode();
  ~TreeNode();
};

class DecisionTree
{
public:
  DecisionTree();
  ~DecisionTree();
  void train();
  void predict();
  double score();
};

class RandomForest
{
public:
  RandomForest();
  ~RandomForest();
  void train();
  void predict();
  double score();
};

class NaiveBayes
{
public:
  NaiveBayes();
  ~NaiveBayes();
  void train();
  void predict();
  double score();
};

#endif