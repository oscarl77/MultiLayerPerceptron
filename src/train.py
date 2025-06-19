from src.model import MultiLayerPerceptron
from src.data_tools.process_dataset import get_preprocessed_datasets
from src.utils.parameter_loader import load_model_hyperparams, load_training_hyperparams
from src.scripts.train_one_epoch import train_one_epoch

def train():
    x_train, y_train, x_val, y_val = get_preprocessed_datasets(mode="TRAIN")
    (n, d) = x_train.shape
    (n, c) = y_train.shape
    hidden_layers, hidden_activation, output_activation, init_func = load_model_hyperparams()
    batch_size, epochs = load_training_hyperparams()

    model = MultiLayerPerceptron(input_dim=d, output_dim=c, hidden_layers_config=hidden_layers,
                                 hidden_activation_fn=hidden_activation, output_activation_fn=output_activation,
                                 weight_init_strategy=init_func)

    for epoch in range(epochs):
        train_one_epoch(model, x_train, y_train, batch_size)


if __name__ == '__main__':
    train()