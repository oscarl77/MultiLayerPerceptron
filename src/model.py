import numpy as np

class MultiLayerPerceptron:

    def __init__(self, input_dim: int, output_dim: int, hidden_layers_config: list[int],
                 hidden_activation_fn: callable, output_activation_fn: callable, weight_init_strategy: callable):

        self.input_dim = input_dim # number of features in input data
        self.output_dim = output_dim # number of neurons in output layer
        self.hidden_layers_config = hidden_layers_config # number of neurons per layer
        self.hidden_activation_fn = hidden_activation_fn # hidden layer activation functions
        self.output_activation_fn = output_activation_fn # output layer activation function

        # store number of dimensions per layer and number of layers
        self.layer_dims = [self.input_dim] + hidden_layers_config + [output_dim]
        self.num_layers = len(self.layer_dims) - 1

        self.parameters = {} # dictionary to store weights and biases per layer

        self._initialise_params(weight_init_strategy)

    def forward(self, X):
        A_prev = X # Initial input
        cache = [] # Store (Z, A_prev) for each layer

        # For all hidden layers
        for i in range(self.num_layers - 1):
            layer_idx = i + 1

            # Retrieve current layer weights and biases
            W_current = self.parameters[f'W{layer_idx}']
            b_current = self.parameters[f'b{layer_idx}']

            # Compute weighted sum (pre-activation)
            Z_current = A_prev @ W_current.T + b_current

            # Apply hidden layer activation function
            A_current = self.hidden_activation_fn(Z_current)

            # Store current pre-activation and previous output for backward pass
            cache.append((Z_current, A_prev))

            # Store current output as previous for next layer
            A_prev = A_current

        output_layer_idx = self.num_layers
        W_output = self.parameters[f'W{output_layer_idx}']
        b_output = self.parameters[f'b{output_layer_idx}']

        Z_output = A_prev @ W_output.T + b_output
        AL = self.output_activation_fn(Z_output)

        cache.append((Z_output, A_prev))

        return AL, cache

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

