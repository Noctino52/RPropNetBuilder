import numpy as np
import functions as fun

class MultilayerNet:
    def __init__(self, n_hidden_layers, n_hidden_nodes_per_layer, act_fun_codes, error_fun_code):
        self.n_input_nodes = 784 # dipende dal dataset: mnist_in = 784
        self.n_output_nodes = 10 # dipende dal dataset: mnist_out = 10
        self.n_layers = n_hidden_layers + 1

        self.error_fun_code = error_fun_code
        self.act_fun_code_per_layer = act_fun_codes.copy()

        self.nodes_per_layer = n_hidden_nodes_per_layer.copy()
        self.nodes_per_layer.append(self.n_output_nodes)

        self.weights = list()
        self.bias = list()
        self.deltaW = list()
        self.deltaB = list()

        self.__initialize_weights_and_bias()

    def __initialize_weights_and_bias(self):
        mu, sigma, lr = 0, 0.1,0.01

        for i in range(self.n_layers):
            if i == 0:
                self.weights.append(np.random.normal(mu, sigma, size=(self.nodes_per_layer[i], self.n_input_nodes)))
                self.deltaW.append(np.full((self.nodes_per_layer[i], self.n_input_nodes), lr))
            else:
                self.weights.append(np.random.normal(mu, sigma, size=(self.nodes_per_layer[i], self.nodes_per_layer[i-1])))
                self.deltaW.append(np.full((self.nodes_per_layer[i], self.nodes_per_layer[i-1]), lr))

            self.bias.append(np.random.normal(mu, sigma, size=(self.nodes_per_layer[i], 1)))
            self.deltaB.append(np.full((self.nodes_per_layer[i], 1), lr))

    def forward_step(self, x): 
        layers_input = list()
        layers_output = list()

        for i in range(self.n_layers):
            if i == 0:
                input = np.dot(self.weights[i], x) + self.bias[i]
                layers_input.append(input)

            else:
                input = np.dot(self.weights[i], layers_output[i-1]) + self.bias[i]
                layers_input.append(input)

            act_fun = fun.activation_functions[self.act_fun_code_per_layer[i]]
            output = act_fun(input)
            layers_output.append(output)

        return layers_input, layers_output

    def sim(self, x):
        for i in range(self.n_layers):
            if i == 0:
                input = np.dot(self.weights[i], x) + self.bias[i]
            else:
                input = np.dot(self.weights[i], output) + self.bias[i]

            act_fun = fun.activation_functions[self.act_fun_code_per_layer[i]]
            output = act_fun(input)

        return output