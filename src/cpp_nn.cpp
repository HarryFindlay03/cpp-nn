#include <cpp_nn.h>
#include "network.h"

using Eigen::MatrixXd;

Eigen::MatrixXd f_sigmoid(const Eigen::MatrixXd& mat, bool deriv=false);
Eigen::MatrixXd f_softmax(const Eigen::MatrixXd& mat, bool deriv=false);
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
    Layer l(t, 4, false, true, f_sigmoid);

    std::cout << "Testing layer construction!" << std::endl;
    
    std::vector<size_t> t_vec = {100, 100, 100, 100};
    ML_ANN* ann_t = new ML_ANN(t_vec, 100);

    delete ann_t;
    
    Eigen::MatrixXd t_mat(4, 4);
    t_mat << 1, 2, 3, 4,
          5, 6, 7, 8,
          9, 10, 11, 12,
          13, 14, 15, 16;
    std::cout << f_softmax(t_mat) << std::endl;

    Eigen::MatrixXd cpy = t_mat;
    std::cout << std::endl << t_mat << std::endl;

    cpy(0, 3) = 1000;
    std::cout << std::endl << t_mat << std::endl << cpy << std::endl;

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
    
    Eigen::MatrixXd exp_mat(mat.rows(), mat.cols());
    Eigen::MatrixXd softmax_res(mat.rows(), 1);
    
    // summing the rows and placing into a column vector, each row representing the sum of mat_{row i}
    size_t pos = 0;
    for(auto row : mat.rowwise())
    {
        size_t temp = 0;
        for(auto v : row)
        {
            temp += exp(v);
        }
        softmax_res(pos++, 0) = temp;
    }
    
    // exponential matrix
    size_t i, j;
    for(i = 0; i < mat.size(); i++)
        *(exp_mat.data() + i) = exp(*(mat.data() + i));
    
    // for each item in the exponentiation matrix - divide by corresponding row in softmax_res
    for(i = 0; i < exp_mat.rows(); i++)
        for(j = 0; j < exp_mat.cols(); j++)
            exp_mat(i, j) = (exp_mat(i, j) / softmax_res(i));

    return exp_mat;
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
        Z(i, col-1) = 1;

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
            layers.push_back(new Layer(size, minibatch_size, true, false, f_sigmoid));
            continue;
        }
        
        // hidden layers
        std::cout << "Initialising hidden layer with size: " << layer_config[i] << std::endl;
        
        size_t size[2] = {layer_config[i]+1, layer_config[i+1]};
        layers.push_back(new Layer(size, minibatch_size, false, false, f_sigmoid));
    }

    // output layer
    std::cout << "Initialising outupt layer with size: " << layer_config.back() << std::endl;
    size_t size[2] = {layer_config.back(), 0};
    layers.push_back(new Layer(size, minibatch_size, false, true, f_softmax));

    std::cout << "ML-ANN CONSTRUCTION COMPLETE!" << std::endl;
}

ML_ANN::~ML_ANN()
{
    for(auto it = layers.begin(); it != layers.end(); ++it)
        delete *it;
}

Eigen::MatrixXd ML_ANN::forward_propogate(const Eigen::MatrixXd& data)
{
    // for input set Z first to data and add an additional column for bias
    Layer* l_ptr_0 = layers[0];
    size_t i;
    for(i = 0; i < l_ptr_0->Z.size(); i++)
        *(l_ptr_0->Z.data() + i) = *(data.data() + i);
    
    // resize
    l_ptr_0->Z.resize(Eigen::NoChange, l_ptr_0->Z.cols() + 1);

    // add bias column
    for(i = 0; i < l_ptr_0->Z.rows(); i++)
        l_ptr_0->Z(i, l_ptr_0->Z.cols() - 1) = 1;

    // loop through remaining layers and forward propogate input
    for(i = 0; i < num_layers - 1; i++)
        layers[i+1]->S = layers[i]->forward_propogate();

    return (layers.back())->forward_propogate();
}

void ML_ANN::back_propogate(const Eigen::MatrixXd& yhat, const Eigen::MatrixXd& labels)
{
    layers.back()->D = (yhat - labels).transpose();

    // backwards through the layers
    size_t i;
    for(i = num_layers-2; i > 0; i--)
    {
        // deltas for the bias values are not calculated
        Eigen::MatrixXd W_nbias = layers[i]->W; //copying like this works

        std::vector<int> indices(W_nbias.cols()-1);
        std::iota(indices.begin(), indices.end(), 0);
        W_nbias(indices, Eigen::all);

        layers[i]->D = W_nbias * layers[i+1]->D * layers[i]->Fp;
    }

    return;
}

void ML_ANN::update_weights(size_t eta)
{
    size_t i;
    for(i = 0; i < num_layers - 1; i++)
    {
        Eigen::MatrixXd W_grad = -(eta) * ((layers[i+1]->D * layers[i]->Z).transpose());
        layers[i]->W += W_grad;
    }

    return;
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

