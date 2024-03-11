#ifndef ML_ANN_H
#define ML_ANN_H

class ML_ANN
{
    std::vector<Layer*> layers;
    size_t num_layers;
    size_t minibatch_size;

public:
    ML_ANN(const std::vector<size_t>& layer_config, size_t minibatch_size);

    ML_ANN(const std::vector<size_t>& layer_config);

    ~ML_ANN();

    Eigen::MatrixXd forward_propogate(const Eigen::MatrixXd& data);
    double forward_propogate_rl(const std::vector<double>& data);

    void back_propogate(const Eigen::MatrixXd& yhat, const Eigen::MatrixXd& labels);

    void update_weights(size_t learning_rate);

    void evaluate(
        const Eigen::MatrixXd& train_data, 
        const Eigen::MatrixXd& train_label, 
        const Eigen::MatrixXd& test_data, 
        const Eigen::MatrixXd& test_labels,
        size_t num_epochs,
        double eta,
        bool eval_train,
        bool eval_test);
};

#endif