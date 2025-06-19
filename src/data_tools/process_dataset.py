import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
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
    validation_split = config["DATA"]["VALIDATION_SPLIT"]
    train_set, val_set = load_data(mode)
    train_set = filter_numbers_in_data(train_set)
    x_train, y_train = convert_dataset_to_array(train_set)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_split, random_state=42, stratify=y_train)
    x_train, y_train = flatten_dataset(x_train), one_hot_encode(y_train)
    x_val, y_val = flatten_dataset(x_val), one_hot_encode(y_val)
    return x_train, y_train, x_val, y_val

def get_preprocessed_training_set(mode):
    train_set, val_set = load_data(mode)
    train_set, val_set = filter_numbers_in_data(train_set), filter_numbers_in_data(val_set)
    train_set, val_set = split_train_and_validation(train_set, val_set)
    x_train, y_train = convert_dataset_to_array(train_set)
    x_train, y_train = flatten_dataset(x_train), one_hot_encode(y_train)
    x_val, y_val = convert_dataset_to_array(val_set)
    x_val, y_val = flatten_dataset(x_val), one_hot_encode(y_val)
    return x_train, y_train, x_val, y_val

def get_preprocessed_test_set(mode):
    test_set = load_data(mode)
    test_set = filter_numbers_in_data(test_set)
    x_test, y_test = convert_dataset_to_array(test_set)
    x_test, y_test = flatten_dataset(x_test), one_hot_encode(y_test)
    return x_test, y_test

def load_data(set_type):
    """
    Load train/validation/test MNIST dataset.
    :param set_type: The type of dataset to load.
    :return: Training and validation set if type is training, otherwise we return the test set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    if set_type == 'TRAIN':
        full_train_dataset = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
        full_validation_dataset = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
        return full_train_dataset, full_validation_dataset
    elif set_type == 'TEST':
        full_test_dataset = datasets.MNIST(root="../data", train=False, download=True, transform=transform)
        return full_test_dataset
    else:
        raise ValueError(f'Invalid set type: {set_type}')

def filter_numbers_in_data(dataset):
    """
    Remove certain numbers from the MNIST dataset.
    :return: dataset
    """
    numbers_to_keep = config["DATA"]["NUMBERS_TO_KEEP"]
    MAX_NUMBER_OF_CLASSES = 10

    if type(numbers_to_keep) != list:
        raise TypeError('numbers_to_keep should be a list')

    if len(numbers_to_keep) == MAX_NUMBER_OF_CLASSES:
        return

    dataset_indices = [i for i, num in enumerate(dataset.targets) if num in numbers_to_keep]
    filtered_dataset = Subset(dataset, dataset_indices)

    return filtered_dataset

def split_train_and_validation(train_set, validation_set):
    """
    Split train and validation sets according to the validation split ratio.
    :param train_set: Full train dataset.
    :param validation_set: Full validation dataset.
    :return: The split train and validation sets.
    """
    validation_split = config["DATA"]["VALIDATION_SPLIT"]
    train_size = len(train_set)
    indices = list(range(train_size))
    np.random.shuffle(indices)
    validation_size = int(validation_split * train_size)
    train_size = train_size - validation_size
    train_indices, validation_indices = indices[:train_size], indices[train_size:]
    train_set = Subset(train_set, train_indices)
    validation_set = Subset(validation_set, validation_indices)
    return train_set, validation_set

def convert_dataset_to_array(dataset):
    """
    Convert PyTorch dataset to numpy arrays.
    :param dataset: PyTorch dataset.
    :return: Numpy arrays of the images and labels
    """
    dataset_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    x_set, y_set = next(iter(dataset_loader))
    x_set_array = x_set.squeeze(1).numpy()
    y_set_array = y_set.numpy()
    return x_set_array, y_set_array

def flatten_dataset(dataset):
    """Convert dataset into 1D array"""
    return dataset.reshape(dataset.shape[0], -1)

def one_hot_encode(labels):
    """One hot encode labels by class"""
    numbers_to_keep = config["DATA"]["NUMBERS_TO_KEEP"]
    total_classes = len(numbers_to_keep)
    labels_one_hot = np.eye(total_classes)[labels]
    return labels_one_hot
