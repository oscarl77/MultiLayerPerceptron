import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.utils.config_loader import load_config

config = load_config()

def generate_batches(X, y, batch_size, shuffle=True):
    """
    Generate mini-batches from given dataset
    :param X: Feature array
    :param y: Label array
    :param batch_size: Size of each mini-batch
    :param shuffle: Whether to shuffle the data before generating batches
    :yield: tuple (X_batch, y_batch) of each mini-batch
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        yield X_batch, y_batch


def get_preprocessed_datasets(mode):
    """
    Get flattened and split datasets for training, validation and testing.
    :param mode: Workflow of choice, either train or test.
    :return: Train and validation sets for training workflow, otherwise
             return the test set for testing workflow.
    """
    if mode == 'TRAIN':
        return _get_preprocessed_training_set(mode)
    elif mode == 'TEST':
        return get_preprocessed_test_set(mode)
    else:
        raise ValueError(f'Invalid mode: {mode}')

def _get_preprocessed_training_set(mode):
    VAL_SPLIT = config["DATA_CONFIG"]["VALIDATION_SPLIT"]
    RANDOM_SEED = config["TRAINING_CONFIG"]["RANDOM_SEED"]
    train_set = load_data(mode)
    x_train, y_train = convert_dataset_to_array(train_set)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=VAL_SPLIT, random_state=RANDOM_SEED, stratify=y_train)
    x_train, y_train = flatten_dataset(x_train), one_hot_encode(y_train)
    x_val, y_val = flatten_dataset(x_val), one_hot_encode(y_val)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train, x_val = scaler.transform(x_train), scaler.transform(x_val)

    scaler_filename = 'scaler.gz'
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler saved to {scaler_filename}")

    return x_train, y_train, x_val, y_val

def get_preprocessed_test_set(mode):
    scaler_filename = 'scaler.gz'
    loaded_scaler = joblib.load(scaler_filename)
    print(f"Scaler loaded from {scaler_filename}")
    test_set = load_data(mode)
    x_test, y_test = convert_dataset_to_array(test_set)
    x_test, y_test = flatten_dataset(x_test), one_hot_encode(y_test)
    x_test = loaded_scaler.transform(x_test)
    return x_test, y_test

def load_data(set_type):
    """
    Load train/validation/test MNIST dataset.
    :param set_type: The type of dataset to load.
    :return: Training and validation set if type is training, otherwise we return the test set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if set_type == 'TRAIN':
        full_train_dataset = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
        return full_train_dataset
    elif set_type == 'TEST':
        full_test_dataset = datasets.MNIST(root="../data", train=False, download=True, transform=transform)
        return full_test_dataset
    else:
        raise ValueError(f'Invalid set type: {set_type}')

def convert_dataset_to_array(dataset):
    """
    Convert PyTorch dataset to numpy arrays.
    :param dataset: PyTorch dataset.
    :return: Numpy arrays of the images and labels
    """
    dataset_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    x_set, y_set = next(iter(dataset_loader))
    x_set_array = x_set.squeeze(1).numpy()
    y_set_array = y_set.numpy()
    return x_set_array, y_set_array

def flatten_dataset(dataset):
    """Convert dataset into 1D array"""
    return dataset.reshape(dataset.shape[0], -1)

def one_hot_encode(labels):
    """One hot encode labels by class"""
    labels_one_hot = np.eye(10)[labels]
    return labels_one_hot
