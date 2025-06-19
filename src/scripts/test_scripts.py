import numpy as np

from src.data_tools.process_dataset import generate_batches


def test_model(model, x_test, y_test, batch_size):
    model.eval()
    total_correct = 0
    total_samples = 0

    for x_batch, y_batch in generate_batches(x_test, y_test, batch_size, shuffle=False):
        n = x_batch.shape[0]

        # Forward pass
        predictions = model.forward(x_batch)

        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_batch, axis=1)

        total_correct += np.sum(predicted_classes == true_classes)
        total_samples += n

    overall_accuracy = total_correct / total_samples * 100
    return overall_accuracy

def compute_accuracy(predictions, y_batch):
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_batch, axis=1)
    return np.sum(true_classes == predicted_classes)
