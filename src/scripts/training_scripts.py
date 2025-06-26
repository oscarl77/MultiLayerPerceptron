from src.data_tools.process_dataset import generate_batches

def train_one_epoch(model, optimiser, loss_fn, x_train, y_train, batch_size):
    """
    Script for training the model for one epoch.
    :param model: The model to be trained.
    :param optimiser: The optimiser being used.
    :param loss_fn: The loss function being used.
    :param x_train: The training examples.
    :param y_train: The training labels.
    :param batch_size: The number of examples fed into model at one time.
    :return: The average training loss across all batches in one epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in generate_batches(x_train, y_train, batch_size=batch_size, shuffle=True):
        n = x_batch.shape[0]

        # Forward pass
        predictions = model.forward(x_batch)

        batch_loss = loss_fn.forward(predictions, y_batch)
        dL_dAL = loss_fn.backward()
        running_loss += batch_loss * n

        # Backward pass
        gradients = model.backward(dL_dAL)

        parameters = model.get_parameters()

        optimiser.step(parameters, gradients)
        model.set_parameters(parameters)

        total_samples += n

    avg_train_loss = running_loss / total_samples

    return avg_train_loss

def validate_one_epoch(model, loss_fn, x_val, y_val, batch_size):
    """
    Script for validating the model for one epoch.
    :param model: The model to be validated.
    :param loss_fn: The loss function being used.
    :param x_val: The validation examples.
    :param y_val: The validation labels.
    :param batch_size: Number of examples fed into model at a time.
    :return: The average validation loss across all batches in one epoch.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in generate_batches(x_val, y_val, batch_size=batch_size, shuffle=True):
        n = x_batch.shape[0] # number of examples in the current batch

        # Forward pass
        predictions = model.forward(x_batch)

        batch_loss = loss_fn.forward(predictions, y_batch)
        running_loss += batch_loss * n

        total_samples += n

    avg_val_loss = running_loss / total_samples

    return avg_val_loss