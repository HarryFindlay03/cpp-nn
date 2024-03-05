#include "Eigen/Dense"

#define RANDOM_SEED 1234

class Layer
{
    bool is_input, is_output;
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd &, bool)> activation_func;

public:
    Eigen::MatrixXd Z; // holds output values;
    Eigen::MatrixXd W; // outgoing weight matrix for layer
    Eigen::MatrixXd S; // inputs to this layer
    Eigen::MatrixXd D; // holds the deltas for this layer
    Eigen::MatrixXd Fp; // holds the derivatives for this layer

    Layer(size_t size[2], size_t minibatch_size, bool is_input, bool is_output, std::function<Eigen::MatrixXd(const Eigen::MatrixXd &, bool)> func);

    ~Layer() {} 

    Eigen::MatrixXd forward_propogate();

    void print_layer() {std::cout << S << std::endl;}
};


class ML_ANN
{
    std::vector<Layer*> layers;
    size_t num_layers;
    size_t minibatch_size;

public:
    ML_ANN(const std::vector<size_t>& layer_config, size_t minibatch_size);

    ~ML_ANN();

    Eigen::MatrixXd forward_propogate(const Eigen::MatrixXd& data);

    void back_propogate(const Eigen::MatrixXd& yhat, const Eigen::MatrixXd& labels);

    void update_weights(size_t learning_rate);

    // TODO evaluate function
};
