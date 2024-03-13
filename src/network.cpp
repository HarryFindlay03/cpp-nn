/***
 * IMPLEMENTATION FILE FOR CPP NEURAL NETWORK - INSPIRATION TAKEN FROM BRIAN DOLHANSKY'S SIMPLE MLP IMPLEMENTATION IN PYTHON
 * SRC - https://github.com/bdol/bdol-ml
***/


#include "network.h"


/******************************/
/* GLOBAL ACTIVATION FUCNTIONS */
/******************************/


Eigen::MatrixXd vector_f_sigmoid_rl(const Eigen::MatrixXd& in, bool deriv)
{
    int i;
    if(!deriv)
    {
        Eigen::MatrixXd res(in.rows(), in.cols());

        for(i = 0; i < in.size(); i++)
            *(res.data() + i) = 1 / (1 + exp(-(*(in.data() + i))));

        return res;
    }

    Eigen::MatrixXd temp = vector_f_sigmoid_rl(in, false);
    for(i = 0; i < temp.size(); i++)
        *(temp.data() + i) *= 1 - *(temp.data() + i);

    return temp;
}


double vector_f_sigmoid_rl_output(const Eigen::MatrixXd& inputs)
{
    double sum = 0;

    int i;
    for(i = 0; i < inputs.size(); i++)
        sum += *(inputs.data() + i);

    return 1 / (1 + (exp(-(sum))));
}


/******************************/
/* LAYER CLASS IMPLEMENTATION */
/******************************/


Layer::Layer(int curr_size, int next_size, bool is_input, bool is_output, std::function<Eigen::MatrixXd(const Eigen::MatrixXd &, bool)> activation_fun) 
    : is_input(is_input), is_output(is_output), activation_func(activation_func)
{
    // all layers have a Z matrix
    Z = Eigen::MatrixXd::Zero(curr_size, 1);

    if(!is_input) // output and hidden layers 
    {
        S = Eigen::MatrixXd::Zero(curr_size, 1);
        G = Eigen::MatrixXd::Zero(curr_size, 1);
    }
    if(!is_output) // input layer and hidden layer
    {
        // random generator
        // todo make this a static function for layer class
        std::default_random_engine g;
        g.seed(RANDOM_SEED);
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        auto uni = [&]() { return distribution(g); };

        W = Eigen::MatrixXd::NullaryExpr(curr_size, next_size, uni);
    }
    if((!is_input) && (!is_output)) // hidden layer only
    {
        Fp = Eigen::MatrixXd::Zero(curr_size, 1);
    }

}


Eigen::MatrixXd Layer::forward_propogate_rl()
{
    if(is_input)
        return Z;

    // output is handled seperately in ML_ANN class

    // hidden layer
    Z = vector_f_sigmoid_rl(S, false); // todo something is wrong with using activation_func wrapper
    Fp = vector_f_sigmoid_rl(S, true);

    return (W.transpose().eval()) * Z;
}


/******************************/
/* ML_ANN CLASS IMPLEMENTATION */
/******************************/


ML_ANN::ML_ANN(const std::vector<size_t>& layer_config)
{
    num_layers = layer_config.size();
    layers.resize(num_layers);

    // input layer
    layers[0] = new Layer(layer_config[0], layer_config[1], true, false, vector_f_sigmoid_rl);

    // hidden layers
    int i;
    for(i = 1; i < (num_layers-1); i++)
        layers[i] = new Layer(layer_config[i], layer_config[i+1], false, false, vector_f_sigmoid_rl);

    // output layers
    layers[num_layers-1] = new Layer(1, 0, false, true, vector_f_sigmoid_rl); // todo this needs to change
}


ML_ANN::~ML_ANN()
{
    for(auto it = layers.begin(); it != layers.end(); ++it)
        delete *it;
}


/* remark: static function */
Eigen::MatrixXd ML_ANN::elem_wise_product(const Eigen::MatrixXd& lhs, const Eigen::MatrixXd& rhs)
{
    // if lhs & rhs do not have same dimensions - todo throw exception
    if(!((lhs.cols() == rhs.cols()) && (lhs.rows() == rhs.rows())))
    {
        std::cout << "ERROR: elem wise multiplication not possible";
        std::cout << " matrix dimensions are not equal!" << std::endl;
        std::exit(-1);
    }

    Eigen::MatrixXd res(lhs.rows(), lhs.cols());
    int i, j;
    for(i = 0; i < lhs.size(); i++)
        *(res.data() + i) = *(lhs.data() + i) * *(rhs.data() + i);

    return res;
}


double ML_ANN::forward_propogate_rl(const std::vector<double>& data)
{
    // todo: bias column

    // for the input set Z to data
    auto l_ptr_0 = layers[0];

    // check input data is of correct size
    if(!(data.size() == l_ptr_0->Z.rows()))
        return -1;

    int i;
    for(i = 0; i < l_ptr_0->Z.size(); i++)
        *(l_ptr_0->Z.data() + i) = data[i];

    // forward propogate through hidden layers
    for(i = 1; i < (num_layers-1); i++)
    {
        layers[i]->S = layers[i-1]->forward_propogate_rl(); 

        // store Fp
        layers[i]->Fp = vector_f_sigmoid_rl(layers[i]->S, true);
    }

    // get output
    layers[num_layers-1]->S = layers[num_layers-2]->forward_propogate_rl();
    return vector_f_sigmoid_rl_output(layers[num_layers-1]->S);
}


void ML_ANN::back_propogate_rl(const double output, const double target)
{
    // output layer
    // output matrix for G
    Eigen::MatrixXd t_out(1, 1);
    t_out << (output - target);

    layers[num_layers-1]->G = t_out;

    // backwards through the remaining layers excluding input
    int i;
    for(i = (num_layers-2); i > 0; i--)
    {
        Eigen::MatrixXd W_nbias = layers[i]->W;
        layers[i]->G = ML_ANN::elem_wise_product(layers[i]->Fp, (W_nbias * layers[i+1]->G));
    }
}


// void ML_ANN::update_weights(size_t eta)
// {
//     size_t i;
//     for(i = 0; i < num_layers - 1; i++)
//     {
//         Eigen::MatrixXd W_grad = -(eta) * ((layers[i+1]->D * layers[i]->Z).transpose());
//         layers[i]->W += W_grad;
//     }

//     return;
// }


// void ML_ANN::evaluate(
//     const Eigen::MatrixXd& train_data,
//     const Eigen::MatrixXd& train_labels,
//     const Eigen::MatrixXd& test_data,
//     const Eigen::MatrixXd& test_labels,
//     size_t num_epochs,
//     double eta,
//     bool eval_train,
//     bool eval_test
//     )
// {
//     size_t train_n = train_data.size();
//     size_t test_n = test_data.size();

//     int i, j;
//     for (i = 0; i < num_epochs; i++)
//     {
//         // for each item in train
//         for(j = 0; j < train_data.rows(); j++)
//         {
//             // get the output from the train data, backprop and update weights
//             Eigen::MatrixXd out = forward_propogate(train_data.row(j));
//             back_propogate(out, train_labels.row(j));
//             update_weights(eta);
//         }

//         // if(eval_train)
//         // {
//         //     // get the training error
//         //     int errs = 0;
//         //     for(j = 0; j < train_data.rows(); j++)
//         //     {
//         //         Eigen::MatrixXd out = forward_propogate(train_data.row(j));
//         //         double yhat = out.maxCoeff();
//         //     }
//         // }

//         // if(eval_test)
//         // {
//         //     // get the testing error
//         // }

//         std::cout << "EPOCH: " << i << std::endl;
//     }
// }


