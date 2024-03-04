#include <cpp_nn.h>
#include <Eigen/Core>

using Eigen::MatrixXd;

void f_sigmoid(Eigen::MatrixXd& mat, int deriv=0);
Eigen::MatrixXd init_random_weight_matrix(size_t rows, size_t cols);


/* GLOBAL ACTIVATION FUNCTIONS */
void f_sigmoid(Eigen::MatrixXd& mat, int deriv)
{
    size_t i;
    
    if(!deriv)
    {
        for(i = 0; i < mat.size(); i++)
        {
            auto ptr = mat.data() + i;
            *ptr = 1 / (1 + (exp(-(*ptr))));

        }
        return;
    }
    
    for(i = 0; i < mat.size(); i++)
    {
        auto ptr = mat.data() + i;
        auto temp = 1 / (1 + (exp(-(*ptr))));
        *ptr = temp * (1 - temp);
    }

    return;
}

/* LAYER CLASS */
class Layer
{
    bool is_input, is_output;
    std::function<void(Eigen::MatrixXd&, int)> activation_func;

public:
    Eigen::MatrixXd Z; // holds output values
    Eigen::MatrixXd W; // outgoing weight matrix for layer
    Eigen::MatrixXd S; // inputs to this layer
    Eigen::MatrixXd D; // holds the deltas for this layer
    Eigen::MatrixXd Fp; // holds the derivatives for this layer
                        //
    Layer(size_t size, size_t minibatch_size, bool is_input, bool is_output, std::function<void(Eigen::MatrixXd&, int)> func)
        : is_input(is_input)
        , is_output(is_output)
        , activation_func(func)
    {
        Z = Eigen::MatrixXd::Zero(minibatch_size, size);
    }

    void foo()
    {
        std::cout << Z << std::endl;
    }
};


Eigen::MatrixXd init_random_weight_matrix(size_t rows, size_t cols)
{
    Eigen::MatrixXd weight_matrix(rows, cols);

    size_t i;
    for(i = 0; i < weight_matrix.size(); i++)
        *(weight_matrix.data() + i) = ((double)rand() / (RAND_MAX));

    return weight_matrix;
}

int main()
{
    // random seed
    srand(1234);

    MatrixXd m = init_random_weight_matrix(5, 5);
    std::cout << m << std::endl;

    std::cout << std::endl;

    f_sigmoid(m, 1);
    std::cout << m << std::endl;

    double a = 4;
    exp(a);
    std::cout << a << std::endl;


    Layer l(4, 4, false, false, f_sigmoid);

    l.foo();


    return 0;
}
