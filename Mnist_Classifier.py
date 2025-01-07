import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np

from keras import utils
from keras.datasets import mnist
from deeplearn import Model, backend
from deeplearn.layers import Dense, Convolutional, Reshape
from deeplearn.layers.activations import Sigmoid
from deeplearn.loss import binary_cross_entropy, binary_cross_entropy_prime


def preprocess_data(x, y, limit):
    """
    Preprocess MNIST data to extract and normalize images with specific labels.

    Args:
        x: Input data (images).
        y: Labels.
        limit: Number of samples to extract for each label.

    Returns:
        Preprocessed input data and one-hot encoded labels in backend-compatible format.
    """
    # Get indices of images with label 0 and 1
    zero_index = backend.to_numpy(backend.xp.where(y == 0)[0][:limit])
    one_index = backend.to_numpy(backend.xp.where(y == 1)[0][:limit])
    
    # Stack the indices and shuffle them
    all_indices = backend.to_numpy(backend.xp.hstack((zero_index, one_index)))
    backend.xp.random.shuffle(all_indices)

    # Select the images and labels with the selected indices (labels 0 and 1)
    x, y = x[all_indices], y[all_indices]

    # Normalize images and reshape
    x = x.reshape(len(x), 1, 28, 28).astype('float32') / 255

    # One-hot encode the labels
    y = utils.to_categorical(y, num_classes=2).reshape(len(y), 2, 1)

    # Convert to the backend array format
    x = backend.from_numpy(x)
    y = backend.from_numpy(y)

    return x, y


def main():
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess training and test data
    x_train, y_train = preprocess_data(x_train, y_train, 100)
    x_test, y_test = preprocess_data(x_test, y_test, 100)

    # Set backend to use CUDA if available
    backend.set_cuda(use_cuda=True)

    # Define the neural network
    network = [
        Convolutional((1, 28, 28), 3, 5),
        Sigmoid(),
        Reshape((5, 26, 26), (5 * 26 * 26, 1)),
        Dense(5 * 26 * 26, 100),
        Sigmoid(),
        Dense(100, 2),
        Sigmoid()
    ]

    # Initialize the model
    model = Model(network)

    # Train the model
    model.train(binary_cross_entropy, binary_cross_entropy_prime, x_train, y_train, epochs=20, learning_rate=0.1, verbose=False)

    # Test the model
    for x, y in zip(x_test, y_test):
        # Predict and convert outputs to NumPy for compatibility
        output = backend.to_numpy(model.predict(x))
        y = backend.to_numpy(y)

        # Display predictions
        print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")


if __name__ == "__main__":
    main()
