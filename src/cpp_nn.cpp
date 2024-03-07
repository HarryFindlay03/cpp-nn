#include "cpp_nn.h"


int main()
{
    // random seed
    srand(1234);

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