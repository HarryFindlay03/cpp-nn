#include "cpp_nn.h"


// int main()
// {
//     // random seed
//     srand(1234);

//     size_t t[4] = {4, 4};
//     Layer l(t, 4, false, true, f_sigmoid);

//     std::cout << "Testing layer construction!" << std::endl;
    
//     std::vector<size_t> t_vec = {100, 100, 100, 100};
//     ML_ANN* ann_t = new ML_ANN(t_vec, 100);

//     delete ann_t;
    
//     Eigen::MatrixXd t_mat(4, 4);
//     t_mat << 1, 2, 3, 4,
//           5, 6, 7, 8,
//           9, 10, 11, 12,
//           13, 14, 15, 16;
//     std::cout << f_softmax(t_mat) << std::endl;

//     Eigen::MatrixXd cpy = t_mat;
//     std::cout << std::endl << t_mat << std::endl;

//     cpy(0, 3) = 1000;
//     std::cout << std::endl << t_mat << std::endl << cpy << std::endl;

//     return 0;
// }

int main()
{
    std::vector<size_t> net_layout = {3, 3, 3, 1}; // the 1 here is redundant but still add it

    std::cout << "Hello, world!" << std::endl;

    ML_ANN* q_net = new ML_ANN(net_layout);

    std::vector<double> test_data = {0.5, 0.1, 0.3};
    // std::cout << "TEST: " << q_net->forward_propogate_rl(test_data) << std::endl;


    delete q_net;
    return 0;
}