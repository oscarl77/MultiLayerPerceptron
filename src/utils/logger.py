import os
import pickle
import shutil

from src.utils.config_loader import load_config

config = load_config()
NAME = config["LOGGING_SETTINGS"]["EXPERIMENT_ID"]
BASE_DIR = config["LOGGING_SETTINGS"]["BASE_DIR"]
DESTINATION_DIR = os.path.join(BASE_DIR, NAME)

def log_experiment(model, fig):
    """
    Log experiment details if logging is enabled
    :param model: Trained model
    :param fig: Plot figure
    """
    _create_experiment_folder()
    _save_config()
    _save_model(model)
    _save_plot(fig)
    print(f"Experiment saved to: {DESTINATION_DIR}")

def _create_experiment_folder():
    """Create folder for current experiment in experiments directory"""
    experiment_path = os.path.join(BASE_DIR, NAME)
    os.makedirs(experiment_path, exist_ok=True)
    return experiment_path

def _save_config():
    """Save config file to experiment folder"""
    try:
        source_config_path = './config.json'
        destination_config_path = os.path.join(str(DESTINATION_DIR), "config.json")
        shutil.copyfile(source_config_path, destination_config_path)
    except FileNotFoundError:
        print("Error saving config file to experiment directory")

def _save_model(model):
    """
    Save trained model parameters as a pickle file.
    :param model: Trained model
    """
    path = os.path.join(str(DESTINATION_DIR), "model.pkl")
    try:
        with open(path, 'wb') as f:
            trained_parameters = model.get_parameters()
            pickle.dump(trained_parameters, f)
    except Exception as e:
        print(f"Error saving parameters: {e}")

def _save_plot(fig):
    """
    Save plot to current experiment folder.
    :param fig: matplotlib figure
    """
    plot_filepath = os.path.join(str(DESTINATION_DIR), "loss_plot.png")
    try:
        fig.savefig(plot_filepath)
    except FileNotFoundError as e:
        print(f"Error saving plot: {e}")