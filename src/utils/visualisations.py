import matplotlib.pyplot as plt

from src.utils.config_loader import load_config

def plot_losses(train_losses, val_losses):
    """
    PLot training and validation losses
    :param train_losses: List of training losses
    :param val_losses: List of validation losses
    :return: The plt figure
    """
    config = load_config()
    experiment_id = config["LOGGING_SETTINGS"]["EXPERIMENT_ID"]
    epochs = range(len(train_losses))
    fig = plt.figure(figsize=(12, 6))
    plt.suptitle(experiment_id, fontsize=16, fontweight='bold')
    plt.plot(epochs, train_losses, label='Training loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation loss', color='red')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    return fig