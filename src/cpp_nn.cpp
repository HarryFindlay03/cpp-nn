#include "cpp_nn.h"


int main()
{
    std::vector<size_t> net_layout = {2, 2, 2, 1}; // the 1 here is redundant but still add it

    std::cout << "Hello, world!" << std::endl;

    ML_ANN* q_net = new ML_ANN(net_layout);

    std::vector<double> test_data = {0.5, 0.1};
    std::cout << "TEST: " << q_net->forward_propogate_rl(test_data) << std::endl;

    auto output = q_net->forward_propogate_rl(test_data);
    q_net->back_propogate_rl(output, 0.4);

    delete q_net;
    return 0;
}