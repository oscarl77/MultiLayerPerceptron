import numpy as np

from src.data_tools.process_dataset import generate_batches
from src.mlp_utils.loss_fns import cross_entropy_loss

def train_one_epoch(model, optimiser, x_train, y_train, batch_size, epoch):
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for x_batch, y_batch in generate_batches(x_train, y_train, batch_size=batch_size, shuffle=True):
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

        correct_predictions += compute_training_accuracy(predictions, y_batch)
        total_samples += n

    train_accuracy = correct_predictions / total_samples * 100
    avg_train_loss = running_loss / total_samples
    print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")

def compute_training_accuracy(predictions, y_batch):
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_batch, axis=1)
    return np.sum(true_classes == predicted_classes)
