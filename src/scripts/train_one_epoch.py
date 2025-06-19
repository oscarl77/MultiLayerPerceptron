import numpy as np
from src.data_tools.process_dataset import generate_batches

def train_one_epoch(model, x_train, y_train, batch_size):
    correct_train_predictions = 0
    num_train_samples_in_epoch = 0

    for x_batch, y_batch in generate_batches(x_train, y_train, batch_size=batch_size, shuffle=True):
        predictions, cache = model.forward(x_batch)

        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_batch, axis=1)
        correct_train_predictions += np.sum(predicted_classes == true_classes)
        num_train_samples_in_epoch += x_batch.shape[0]

    train_accuracy = correct_train_predictions / num_train_samples_in_epoch * 100
    print(f"Train Acc: {train_accuracy:.2f}%")
