# Simple MLP Neural Network Implementation in C++

Inspiration taken from a similar implementation in python authored by Brian Dolhansky (https://github.com/bdol/bdol-ml)

### INSTALLATION PREREQUISITES
Please ensure you have downloaded the latest version of Eigen (3.4.0) found here https://eigen.tuxfamily.org/index.php?title=Main_Page and copy the downloaded folder into /include.

### Example Network Usage
For a network with 3 input nodes and the hidden layers of size 3, 4, 5 respectively the below code initialises the network.

```
std::vector<size_t> net_layout = {3, 3, 4, 5, 1};
ML_ANN* q_net = new ML_ANN(net_layout);
```

The 1 at the end represents the output as currently we only need one output as this network is going to be used as the brain of a deep RL agent.

**Available functions:**
- forward_propogate_rl(input) - takes data as input and feeds the input through the network
- back_propogate_rl(output, target) - propogates the computed error back through the network
- update_weights_rl() - updates the weights using the error calculated by back propogation