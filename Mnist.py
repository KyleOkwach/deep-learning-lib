import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from keras import utils
from keras.datasets import mnist

from deeplearn import Model
from deeplearn.layers import Dense
from deeplearn.layers.activations import Tanh
from deeplearn.loss import mse, mse_prime
from deeplearn.utils import backend, save_model, load_model

def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]

def main():
    # load mnist data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 1000)
    x_test, y_test = preprocess_data(x_test, y_test, 20)

    # neural network
    network = [
        Dense(28 * 28, 40),
        Tanh(),
        Dense(40, 10),
        Tanh()
    ]

    model = Model(network)
    model.train(mse, mse_prime, x_train, y_train, 100, 0.001)

    # test the model
    for x, y in zip(x_test, y_test):
        output = model.predict(x)
        print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")

if __name__ == "__main__":
    main()