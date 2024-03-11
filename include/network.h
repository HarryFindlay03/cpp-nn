#include <iostream>
#include <random>
#include <ctime>

#include "Eigen/Dense"

#include "Layer.h"
#include "ML_ANN.h"

#define RANDOM_SEED 1234

/* GLOBAL ACTIVATION FUNCTIONS */
Eigen::MatrixXd f_sigmoid(const Eigen::MatrixXd& mat, bool deriv=false);
Eigen::MatrixXd f_softmax(const Eigen::MatrixXd& mat, bool deriv=false);

Eigen::MatrixXd vector_f_sigmoid_rl(const Eigen::MatrixXd& in, bool deriv);