# Simple MLP Neural Network Implementation in C++

Inspiration taken from a similar implementation in python authored by Brian Dolhansky (https://github.com/bdol/bdol-ml)

### INSTALLATION PREREQUISITES
Please ensure you have downloaded the latest version of Eigen (3.4.0) found here https://eigen.tuxfamily.org/index.php?title=Main_Page and copy the downloaded folder into /include.

### Example Network Usage
For a network with 1 input node and 3 hidden layers of size 30 respectively and 1 output node.

```
std::vector<int> net_layout = {1, 30, 30, 30, 1};
MLP* mlp = new ML_ANN(net_layout, std::make_pair(hidden_activation_func, output_activation_func), double learning_rate, rand_helper* rnd);
```

**Available functions:**
- forward_propogate(input) - takes data as input and feeds the input through the network
- back_propogate(output, target) - propogates the computed error back through the network
- update_weights() - updates the weights using the error calculated by back propogation