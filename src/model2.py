from src.mlp_utils.layers import DenseLayer
from src.utils.config_loader import load_config
from src.mlp_utils.activations import ACTIVATIONS

class MultiLayerPerceptron2:

    def __init__(self):
        self.config = load_config()["NETWORK_CONFIG"]
        self.layers = self.build_layers()
        self.mode = None
        self.parameters = {}
        self.set_params_in_layers()

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.set_params_in_layers()

    def get_parameters(self):
        return self.parameters

    def set_params_in_layers(self):
        for i, layer in enumerate(self.layers):
            layer_idx = i + 1
            W, b = layer.get_params()
            self.parameters[f'W{layer_idx}'] = W
            self.parameters[f'b{layer_idx}'] = b


    def forward(self, X):
        for i, layer in enumerate(self.layers):
            layer_idx = i + 1
            layer.set_params(self.parameters[f'W{layer_idx}'], self.parameters[f'b{layer_idx}'])
            X = layer.forward(X)
        return X


    def backward(self, dL_dAL):
        gradients = {}
        dL_dA_prev = dL_dAL
        layer_idx = len(self.layers)
        for layer in reversed(self.layers):
            dL_dA_prev, dL_dW, dL_db = layer.backward(dL_dA_prev, layer_idx)
            gradients[f'W{layer_idx}'] = dL_dW
            gradients[f'b{layer_idx}'] = dL_db
            layer_idx -= 1
        return gradients


    def train(self):
        self.mode = "TRAIN"

    def eval(self):
        self.mode = "EVAL"


    def build_layers(self):
        """Builds neural network layers defined in the config file."""
        layers = []
        for i, layer_config in enumerate(self.config):
            layer_type = layer_config["TYPE"]

            if layer_type == "DENSE":
                input_dim = layer_config["INPUT_DIM"]
                output_dim = layer_config["OUTPUT_DIM"]
                activation = ACTIVATIONS[layer_config["ACTIVATION"]]()
                dense_layer = DenseLayer(input_dim, output_dim, activation)
                layers.append(dense_layer)
        return layers