import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from deeplearn import Model
from deeplearn.layers import Dense, Convolutional, Reshape
from deeplearn.layers.activations import Tanh, Sigmoid
from deeplearn.loss import binary_cross_entropy, binary_cross_entropy_prime

def preprocess_data(x, y, limit):
    # get indices of images with label 0 and 1
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]

    # stack the indices and shuffle them
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)

    x, y = x[all_indices], y[all_indices]

    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype('float32') / 255

    y = np_utils.to_categorical(y)  # create one-hot encoded labels
    y = y.reshape(len(y), 2, 1)

    return x, y

def main():
    # load mnist data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 100)
    x_test, y_test = preprocess_data(x_test, y_test, 100)

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
    model.train(binary_cross_entropy, binary_cross_entropy_prime, x_train, y_train, epochs=20, learning_rate=0.1)

    # test the model
    for x, y in zip(x_test, y_test):
        output = model.predict(x)
        print(f'X: {x.flatten()}, y: {y.flatten()}, predicted: {output.flatten()}')

if __name__ == "__main__":
    main()