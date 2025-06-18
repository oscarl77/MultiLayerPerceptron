from src.model import MultiLayerPerceptron
from src.data.process_dataset import get_preprocessed_datasets
from src.utils.parameter_loader import load_model_hyperparams

def train():
    x_train, y_train, x_val, y_val = get_preprocessed_datasets(mode="TRAIN")
    (n, d) = x_train.shape
    (n, c) = y_train.shape
    hidden_layers, hidden_activation, output_activation, init_func = load_model_hyperparams()

    model = MultiLayerPerceptron(input_dim=d, output_dim=c, hidden_layers_config=hidden_layers,
                                 hidden_activation=hidden_activation, output_activation=output_activation,
                                 weight_init_strategy=init_func)
    pass

if __name__ == '__main__':
    train()