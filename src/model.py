import numpy as np

class MultiLayerPerceptron:

    def __init__(self, input_dim: int, output_dim: int, hidden_layers_config: list[int],
                 hidden_activation_fn: callable, output_activation_fn: callable, weight_init_strategy: callable):

        self.input_dim = input_dim # number of features in input data
        self.output_dim = output_dim # number of neurons in output layer
        self.hidden_layers_config = hidden_layers_config # number of neurons per layer
        self.hidden_activation_fn = hidden_activation_fn[0] # hidden layer activation function
        self.d_hidden_activation_fn = hidden_activation_fn[1] # hidden layer activation function derivative
        self.output_activation_fn = output_activation_fn # output layer activation function

        # store number of dimensions per layer and number of layers
        self.layer_dims = [self.input_dim] + hidden_layers_config + [output_dim]
        self.num_layers = len(self.layer_dims) - 1
        self.parameters = {} # dictionary to store weights and biases per layer
        self.mode = None

        self._initialise_params(weight_init_strategy)

    def train(self):
        self.mode = 'train'

    def eval(self):
        self.mode = 'eval'

    def get_parameters(self):
        return self.parameters

    def set_parameters(self, params_dict):
        self.parameters = params_dict

    def forward(self, X):
        """
        Forward pass algorithm of the model.
        :param X: Initial input array
        :return: Output array of output layer i.e. the predictions.
        """
        A_prev = X # Initial input
        cache = [] # Store (Z_current, A_prev) for each layer

        # For all hidden layers
        for i in range(self.num_layers - 1):
            layer_idx = i + 1

            # Retrieve current layer weights and biases
            W_current = self.parameters[f'W{layer_idx}']
            b_current = self.parameters[f'b{layer_idx}']

            # Compute weighted sum (pre-activation)
            Z_current = A_prev @ W_current + b_current

            # Apply hidden layer activation function
            A_current = self.hidden_activation_fn(Z_current)

            # Store current pre-activation and previous output for backward pass
            cache.append((Z_current, A_prev))

            # Store current output as previous for next layer
            A_prev = A_current

        output_layer_idx = self.num_layers
        W_output = self.parameters[f'W{output_layer_idx}']
        b_output = self.parameters[f'b{output_layer_idx}']

        Z_output = A_prev @ W_output + b_output

        AL = self.output_activation_fn(Z_output)

        cache.append((Z_output, A_prev))

        if self.mode == 'train':
            return AL, cache
        elif self.mode == 'eval':
            return AL

    def backward(self, AL, y_batch, cache):
        """
        Algorithm for backward pass of model where all gradients of the loss
        w.r.t the weights and biases in the network are computed for the current_batch.
        :param AL: predicted labels array from current batch
        :param y_batch: true labels from current batch
        :param cache: List of tuples (Z_current, A_prev) of pre activations
        and outputs for each layer.
        :return: List of gradients for all weights and biases.
        """
        gradients = {}
        # Gradient of loss w.r.t, output layer's pre-activation.
        # As we are using categorical-cross entropy after a softmax activation,
        # the gradient calculation is simplified.
        dZ_current = AL - y_batch

        # Iterate through layers backwards from last to first hidden layer
        for layer_idx in reversed(range(1, self.num_layers + 1)):
            cache_idx = layer_idx - 1

            Z_current, A_prev = cache[cache_idx]
            W_current = self.parameters[f'W{layer_idx}']

            dA_prev, dW, db = self._compute_linear_gradients(dZ_current, A_prev, W_current)

            gradients[f'W{layer_idx}'] = dW
            gradients[f'b{layer_idx}'] = db

            # if not the first layer, calculate the current dZ
            if layer_idx > 1:
                Z_prev = cache[cache_idx - 1][0]
                dZ_current = self.d_hidden_activation_fn(dA_prev, Z_prev)

        return gradients

    @staticmethod
    def _compute_linear_gradients(dZ, A_prev, W):
        """Compute gradients for a linear layer during backpropagation.
        :param dZ: gradient of the loss w.r.t. the pre-activation of the current layer.
        :param A_prev: output from the previous layer.
        :param W: weights of the current layer.
        :return: tuple (dA_prev, dW, db).
        """
        batch_size = dZ.shape[0]

        # compute gradient of loss w.r.t. the weights
        dW = (A_prev.T @ dZ) / batch_size

        # compute gradient of loss w.r.t the biases
        db = np.sum(dZ, axis=0, keepdims=True) / batch_size

        # compute gradient of loss w.r.t. output of previous layer
        dA_prev = dZ @ W.T

        return dA_prev, dW, db


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