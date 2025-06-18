from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

from src.utils.config_loader import load_config

config = load_config()

def get_data_loaders(mode):
    """
    Get split datasets for training, validation and testing.
    :param mode: Workflow of choice, either train or test.
    :return: Train and validation sets for training workflow, otherwise
             return the test set for testing workflow.
    """
    num_workers = config["DATA"]["NUM_WORKERS"]
    batch_size = config["DATA"]["BATCH_SIZE"]
    train_set_filtered, test_set_filtered = filter_numbers_in_data()
    train_set, validation_set = split_train_and_validation(train_set_filtered, test_set_filtered)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set_filtered, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if mode == "TRAIN":
        return train_loader, validation_loader
    if mode == "TEST":
        return test_loader

def load_data():
    """Load train, validation and test MNIST datasets"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    full_validation_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    full_test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return full_train_dataset, full_validation_dataset, full_test_dataset

def filter_numbers_in_data():
    """
    Remove certain numbers from the MNIST dataset.
    :return: Train, validation and test sets.
    """
    numbers_to_keep = config["DATA"]["NUMBERS_TO_KEEP"]

    if type(numbers_to_keep) != list:
        raise TypeError('numbers_to_keep should be a list')

    full_train, full_validation, full_test = load_data()
    if numbers_to_keep is not None:
        return full_train, full_validation, full_test

    train_indices = [i for i, label in enumerate(full_train.targets) if label in numbers_to_keep]
    test_indices = [i for i, label in enumerate(full_test.targets) if label in numbers_to_keep]

    train_dataset_filtered = Subset(full_train, train_indices)
    test_dataset_filtered = Subset(full_test, test_indices)

    return train_dataset_filtered, test_dataset_filtered

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