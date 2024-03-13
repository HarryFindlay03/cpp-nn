#ifndef LAYER_H
#define LAYER_H

double vector_f_sigmoid_rl_output(const Eigen::MatrixXd &inputs);

class Layer
{
    bool is_input, is_output;

    std::function<Eigen::MatrixXd(const Eigen::MatrixXd &, bool)> activation_func;

    

public:
    Eigen::MatrixXd Z; // holds output values;
    Eigen::MatrixXd S; // output values pre activation function - "inputs into the layer"
    Eigen::MatrixXd W; // outgoing weight matrix for layer
    Eigen::MatrixXd Fp; // holds the derivatives for this layer
    Eigen::MatrixXd G; // gradient matrix

    Layer(int curr_size, int next_size, bool is_input, bool is_output, std::function<Eigen::MatrixXd(const Eigen::MatrixXd &, bool)> activation_func);

    ~Layer() {} 

    Eigen::MatrixXd forward_propogate();

    Eigen::MatrixXd forward_propogate_rl();

    void print_layer() {std::cout << S << std::endl;}
};

#endif