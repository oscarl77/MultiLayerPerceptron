from src.data_tools.process_dataset import generate_batches
from src.mlp_utils.loss_fns import cross_entropy_loss

def train_one_epoch(model, optimiser, x_train, y_train, batch_size):
    """
    Script to train the model per epoch.
    :param model: The model to be trained.
    :param optimiser: The optimiser of choice.
    :param x_train: Array of training data.
    :param y_train: Array of training labels.
    :param batch_size: Size of the batches to be fed into model.
    :return: The average training loss across all batches.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in generate_batches(x_train, y_train, batch_size=batch_size, shuffle=False):
        n = x_batch.shape[0]

        # Forward pass
        predictions, cache = model.forward(x_batch)

        # Compute cross-entropy loss
        batch_loss = cross_entropy_loss(predictions, y_batch)

        running_loss += batch_loss * n

        # Backward pass
        gradients = model.backward(predictions, y_batch, cache)

        # Update model parameters with SGD
        parameters = model.get_parameters()

        optimiser.step(parameters, gradients)

        total_samples += n

    avg_train_loss = running_loss / total_samples

    return avg_train_loss


def validate_one_epoch(model, x_val, y_val, batch_size):
    """
    Script to validate the model per epoch.
    :param model: The model to be validated.
    :param x_val: Array of validation data.
    :param y_val: Array of validation labels.
    :param batch_size: Size of the batches to be fed into model.
    :return: The average validation loss across all batches.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in generate_batches(x_val, y_val, batch_size=batch_size, shuffle=True):
        n = x_batch.shape[0]

        # Forward pass
        predictions = model.forward(x_batch)

        # Compute cross-entropy loss
        batch_loss = cross_entropy_loss(predictions, y_batch)
        running_loss += batch_loss * n

        total_samples += n

    avg_val_loss = running_loss / total_samples

    return avg_val_loss

def train_one_epoch2(model, optimiser, loss_fn, x_train, y_train, batch_size):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in generate_batches(x_train, y_train, batch_size=batch_size, shuffle=False):
        n = x_batch.shape[0]

        predictions = model.forward(x_batch)

        batch_loss = loss_fn.forward(predictions, y_batch)
        dL_dAL = loss_fn.backward()
        running_loss += batch_loss * n

        gradients = model.backward(dL_dAL)

        parameters = model.get_parameters()

        optimiser.step(parameters, gradients)
        model.set_parameters(parameters)

        total_samples += n

    avg_train_loss = running_loss / total_samples

    return avg_train_loss

def validate_one_epoch2(model, loss_fn, x_val, y_val, batch_size):
    model.eval()
    running_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in generate_batches(x_val, y_val, batch_size=batch_size, shuffle=True):
        n = x_batch.shape[0]

        # Forward pass
        predictions = model.forward(x_batch)

        batch_loss = loss_fn.forward(predictions, y_batch)
        running_loss += batch_loss * n

        total_samples += n

    avg_val_loss = running_loss / total_samples

    return avg_val_loss