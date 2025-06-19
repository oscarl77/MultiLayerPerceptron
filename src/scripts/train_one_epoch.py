import numpy as np
from src.data_tools.process_dataset import generate_batches
from src.mlp_utils.loss_fns import cross_entropy_loss

def train_one_epoch(model, x_train, y_train, batch_size):
    epoch_train_loss = 0.0
    correct_train_predictions = 0
    num_train_samples_in_epoch = 0

    for x_batch, y_batch in generate_batches(x_train, y_train, batch_size=batch_size, shuffle=True):
        n = x_batch.shape[0]

        # Forward pass
        predictions, cache = model.forward(x_batch)

        # Compute cross-entropy loss
        batch_loss = cross_entropy_loss(predictions, y_batch)
        epoch_train_loss += batch_loss * n
        num_train_samples_in_epoch += n

        # Compute training accuracy
        correct_train_predictions += compute_training_accuracy(predictions, y_batch)

        # Backward pass
        gradients = model.backward(x_batch, y_batch, cache)


    train_accuracy = correct_train_predictions / num_train_samples_in_epoch * 100
    avg_train_loss = epoch_train_loss / num_train_samples_in_epoch
    print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}")


def compute_training_accuracy(predictions, y_batch):
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_batch, axis=1)
    return np.sum(true_classes == predicted_classes)
