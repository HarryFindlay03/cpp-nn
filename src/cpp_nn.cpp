#include "cpp_nn.h"


int main()
{
    std::vector<size_t> net_layout = {3, 3, 6, 7, 9, 8, 4, 2, 3, 1}; // the 1 here is redundant but still add it

    ML_ANN* q_net = new ML_ANN(net_layout);

    std::vector<double> test_data = {0.5, 0.1, 0.7};

    int i;
    int epochs = 100;
    for(i = 0; i < epochs; i++)
    {
        auto output = q_net->forward_propogate_rl(test_data);
        std::cout << "(" << i << ") OUTPUT: " << output << std::endl;

        q_net->back_propogate_rl(output, 0.4);
        q_net->update_weights_rl(0.3);
    }

    delete q_net;
    return 0;
}