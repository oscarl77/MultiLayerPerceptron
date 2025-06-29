# NumPy MLP for MNIST

This is a project to build a Multi-Layer Perceptron (MLP) neural network entirely from scratch using only NumPy, with the aim of classifying handwritten digits from the MNIST dataset.

## Project Goal

The primary aim of this project is to gain a deep, fundamental understanding of how neural networks operate. This involves implementing all core components – from forward and backward passes to activation functions, loss calculations, and optimizers – without relying on the high-level abstractions provided by deep learning frameworks like TensorFlow or PyTorch for the model's definition and training process.

## Model Architecture

The core components of the architecture are:
  1. Weights are initialised using He initialisation.
  2. Input Layer: 28x28 images flattened into 784-dimensional vector
  3. Hidden Layers:
     - First dense layer has 64 neurons
     - Second dense layer has 32 neurons
     Each hidden layer is followed by a ReLU activation function
  4. Output Layer: Final dense layer with 10 neurons
     A softmax function follows the output layer.

### Additional Architectural Features:
  1. Batch normalisation layers
  2. Dropout layers
 - These layers are implemented, but as they did not provide an increase in model performance, they are not included in the final model architecture.

## Training the model
 1. Optimiser: Stochastic Gradient Descent with momentum
 2. Loss function: Cross-entropy loss
 3. Training details:
    - Learning rate: 0.005
    - Batch size: 64
    - Epochs: The model trained in 3 epochs due to early stopping.
    - Data split: Standard MNIST training set with 10% split off for validation. The standard MNIST test set was used for performance evaluation.

## Results

On a set random seed of 20, the model achieved 96.30% test accuracy on the MNIST dataset, with an average test accuracy of 96.22% across 5 random seeds.
