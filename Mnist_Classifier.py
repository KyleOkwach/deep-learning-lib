import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from keras import utils
from keras.datasets import mnist

from deeplearn import Model
from deeplearn.layers import Dense, Convolutional, Reshape
from deeplearn.layers.activations import Sigmoid
from deeplearn.loss import binary_cross_entropy, binary_cross_entropy_prime

def preprocess_data(x, y, limit):
    # get indices of images with label 0 and 1
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    two_index = np.where(y == 2)[0][:limit]

    # stack the indices and shuffle them
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)

    # get the images and labels with the selected indices (0 and 1)
    x, y = x[all_indices], y[all_indices]

    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype('float32') / 255

    y = utils.to_categorical(y)  # create one-hot encoded labels
    y = y.reshape(len(y), 2, 1)

    return x, y

def main():
    # load mnist data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 100)
    x_test, y_test = preprocess_data(x_test, y_test, 100)

    # neural network
    network = [
        Convolutional((1, 28, 28), 3, 5),
        Sigmoid(),
        Reshape((5, 26, 26), (5 * 26 * 26, 1)),
        Dense(5 * 26 * 26, 100),
        Sigmoid(),
        Dense(100, 2),
        Sigmoid()
    ]

    model = Model(network)
    model.train(binary_cross_entropy, binary_cross_entropy_prime, x_train, y_train, 20, 0.1, False)

    # test the model
    for x, y in zip(x_test, y_test):
        output = model.predict(x)
        print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")

if __name__ == "__main__":
    main()