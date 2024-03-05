#include <cpp_nn.h>
#include "network.h"

using Eigen::MatrixXd;

Eigen::MatrixXd f_sigmoid(const Eigen::MatrixXd& mat, bool deriv=false);
Eigen::MatrixXd init_random_weight_matrix(size_t rows, size_t cols);

int main()
{
    // random seed
    srand(1234);

    MatrixXd m = init_random_weight_matrix(5, 5);
    std::cout << m << std::endl;

    std::cout << std::endl;

    auto test = f_sigmoid(m, false);
    std::cout << test << std::endl;

    size_t t[4] = {4, 4};
    Layer l(t, 4, false, false, f_sigmoid);

    return 0;
}

/* GLOBAL ACTIVATION FUNCTIONS */
Eigen::MatrixXd f_sigmoid(const Eigen::MatrixXd& mat, bool deriv)
{
    Eigen::MatrixXd res(mat.rows(), mat.cols());
    size_t i;
    
    if(!deriv)
    {
        for(i = 0; i < res.size(); i++)
        {
            auto ptr = res.data() + i;
            *ptr = 1 / (1 + (exp(-(*(mat.data() + i)))));

        }
        return res;
    }
    
    auto temp = f_sigmoid(mat, false);
    for(i = 0; i < temp.size(); i++)
        *(temp.data() + i) *= 1 - *(temp.data() + i);
    
    return temp;
}

Eigen::MatrixXd f_softmax(const Eigen::MatrixXd& mat, bool deriv)
{
    // deriv here to allow contruction with function wrapper - deriv remains unused
    
    // softmax function - https://en.wikipedia.org/wiki/Softmax_function

}

/* LAYER CLASS IMPLEMENTATION */
Layer::Layer(size_t size[2], size_t minibatch_size, bool is_input, bool is_output, std::function<Eigen::MatrixXd(const Eigen::MatrixXd &, bool)> func)
    : is_input(is_input), is_output(is_output), activation_func(func)
{
    Z = Eigen::MatrixXd::Zero(minibatch_size, size[0]);

    // // random generator
    std::default_random_engine g;
    g.seed(RANDOM_SEED);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    auto uni = [&] () {return distribution(g);};

    if(!is_input)
    {
        S = Eigen::MatrixXd::Zero(minibatch_size, size[0]);
        D = Eigen::MatrixXd::Zero(minibatch_size, size[0]);
    }
    if(!is_output)
    {
        W = Eigen::MatrixXd::NullaryExpr(size[0], size[1], uni);
    }
    if((!is_input) && (!is_output)) // hidden layer
    {
        Fp = Eigen::MatrixXd::Zero(size[0], minibatch_size);
    }
}


Eigen::MatrixXd Layer::forward_propogate()
{
    if(is_input)
        return Z * W;

    Z = activation_func(S, false);
    if(is_output)
        return Z;

    // hidden layers

    // resizing Z
    Z.resize(Eigen::NoChange, Z.cols() + 1);
    
    size_t i;
    size_t col = Z.cols();
    for(i = 0; i < Z.rows(); i++)
        Z(i, col) = 1;

    Fp = activation_func(S, true);
    Fp.transposeInPlace();

    return Z * W;
}

/* ML_ANN CLASS */
ML_ANN::ML_ANN(const std::vector<size_t>& layer_config, size_t minibatch_size)
{
    // setting parameters
    this->minibatch_size = minibatch_size;
    num_layers = layer_config.size();

    size_t i;
    for(i = 0; i < (num_layers-1); i++)
    {
        if(i == 0)
        {
            // input layer
            std::cout << "Initialising input layer with size: " << layer_config[i] << std::endl;

            // additional unit at input for bias
            size_t size[2] = {layer_config[i] + 1, layer_config[i+1]};
            Layer l = new Layer(size, minibatch_size, true, false, f_sigmoid);
            layers.push_back(l);
            continue;
        }
        
        // hidden layers
        std::cout << "Initialising hidden layer with size: " << layer_config[i] << std::endl;
        
        size_t size[2] = {layer_config[i]+1, layer_config[i+1]};
        Layer l = new Layer(size, minibatch_size, false, false, f_sigmoid);
        layers.pusb_back(l);
    }

    // output layer
    std::cout << "Initialising outupt layer with size: " << layer_config.back() << std::endl;
    size_t size[2] = {layer_config.back(), 0};
    Layer l = new Layer(size, minibatch_size, false, true, f_softmax);
    layers.pusb_back(l);

    std::cout << "ML-ANN CONSTRUCTION COMPLETE!" << std::endl;
}


/* MISC FUNCTIONS */
Eigen::MatrixXd init_random_weight_matrix(size_t rows, size_t cols)
{
    Eigen::MatrixXd weight_matrix(rows, cols);

    size_t i;
    for(i = 0; i < weight_matrix.size(); i++)
        *(weight_matrix.data() + i) = ((double)rand() / (RAND_MAX));

    return weight_matrix;
}

