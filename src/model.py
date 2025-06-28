from src.mlp_utils.layers import DenseLayer, DropoutLayer
from src.utils.config_loader import load_config
from src.mlp_utils.activations import ACTIVATIONS

class MultiLayerPerceptron:
    """
    Multi Layer Perceptron (MLP) classifier implemented with numpy.

    This class builds a feed-forward neural network trained using Cross-entropy Loss
    and Stochastic Gradient Descent.

    The network can be configured with a variable number of hyperparameters, e.g.
    hidden layers, neurons, learning rate via the config file.

    The model uses ReLU activations for hidden layers and a Softmax activation
    in the output layer.
    """

    def __init__(self):
        self.config = load_config()["NETWORK_CONFIG"]
        self.layers = self._build_layers()
        self.mode = None
        self.parameters = {}
        self._set_initialised_params()

    def get_parameters(self):
        return self.parameters

    def set_parameters(self, parameters):
        self.parameters = parameters
        self._set_parameters_in_layers()

    def train(self):
        """Set the model to training mode."""
        self.mode = "TRAIN"
        for layer in self.layers:
            if isinstance(layer, DropoutLayer):
                layer.enabled = True

    def eval(self):
        """set the model to evaluation mode."""
        self.mode = "EVAL"
        for layer in self.layers:
            if isinstance(layer, DropoutLayer):
                layer.enabled = False

    def forward(self, X):
        """
        Forward pass through the model's layers.
        :param X: Input data.
        :return: Model output.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dL_dAL):
        """
        Backward pass through the model's layers.
        :param dL_dAL: Derivative of Loss w.r.t. model output
        :return: Dictionary representing the gradients of all n layers' parameters, e.g:
            {
                'W1': [gradients], 'b1': [gradients],
                ...,
                'Wn: [gradients], 'bn': [gradients]',
            }
        """
        gradients = {}
        dL_dA_prev = dL_dAL
        layer_idx = len(self.layers)
        for layer in reversed(self.layers):
            dL_dA_prev, dL_dW, dL_db = layer.backward(dL_dA_prev)
            gradients[f'W{layer_idx}'] = dL_dW
            gradients[f'b{layer_idx}'] = dL_db
            layer_idx -= 1
        return gradients

    def _set_initialised_params(self):
        """Store each layer's initialised parameters as a dict in the model."""
        for i, layer in enumerate(self.layers):
            layer_idx = i + 1
            W, b = layer.get_params()
            self.parameters[f'W{layer_idx}'] = W
            self.parameters[f'b{layer_idx}'] = b

    def _set_parameters_in_layers(self):
        """Set each layer's parameters using model's parameter dict."""
        for i, layer in enumerate(self.layers):
            layer_idx = i + 1
            layer.set_params(self.parameters[f'W{layer_idx}'], self.parameters[f'b{layer_idx}'])

    def _build_layers(self):
        """Builds neural network layers defined in the config file."""
        layers = []
        for layer_config in self.config:
            layer_type = layer_config["TYPE"]

            if layer_type == "DENSE":
                self._add_dense_layer(layer_config, layers)

            if layer_type == "DROPOUT":
                self._add_dropout_layer(layer_config, layers)

        return layers

    @staticmethod
    def _add_dense_layer(layer_config, layers):
        """
        Add a dense layer to the model's layers list.
        :param layer_config: Configuration of the layer.
        :param layers: List of model's layers.
        """
        input_dim = layer_config["INPUT_DIM"]
        output_dim = layer_config["OUTPUT_DIM"]
        activation = ACTIVATIONS[layer_config["ACTIVATION"]]()
        dense_layer = DenseLayer(input_dim, output_dim, activation)
        layers.append(dense_layer)

    @staticmethod
    def _add_dropout_layer(layer_config, layers):
        """
        Add a dropout layer to the model's layers list.
        :param layer_config: Configuration of the layer.
        :param layers: List of model's layers.
        """
        dropout_rate = layer_config["RATE"]
        dropout_layer = DropoutLayer(dropout_rate)
        layers.append(dropout_layer)
