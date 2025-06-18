import numpy as np

class MultiLayerPerceptron:

    def __init__(self, input_dim: int, output_dim: int, hidden_layers_config: list[int],
                 hidden_activation: callable, output_activation: callable, weight_init_strategy: callable):

        self.input_dim = input_dim # number of features in input data
        self.output_dim = output_dim # number of neurons in output layer
        self.hidden_layers_config = hidden_layers_config # number of neurons per layer
        self.hidden_activation = hidden_activation # hidden layer activation functions
        self.output_activation = output_activation # output layer activation function

        # store number of dimensions per layer and number of layers
        self.layer_dims = [self.input_dim] + hidden_layers_config + [output_dim]
        self.num_layers = len(self.layer_dims) - 1

        self.parameters = {} # dictionary to store weights and biases per layer

        self._initialise_params(weight_init_strategy)

    def forward(self, X):
        pass

    def _initialise_params(self, strategy: callable):
        """
        Initialise weight and bias vectors for each layer.
        :param strategy: weight initialisation function.
        """
        init_func = strategy
        for i in range(self.num_layers):
            layer_idx = i + 1
            input_dim = self.layer_dims[i]
            output_dim = self.layer_dims[i + 1]

            W = init_func(input_dim, output_dim)
            b = np.zeros(output_dim)

            self.parameters[f'W{layer_idx}'] = W
            self.parameters[f'b{layer_idx}'] = b

