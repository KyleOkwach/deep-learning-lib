from deeplearn.model import Model
from deeplearn.layers import Dense
from deeplearn.layers.activations import Tanh
from deeplearn.loss import mse, mse_prime

import numpy as np

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3),  # input layer
    Tanh(),  # activation function
    Dense(3, 1),  # output layer
    Tanh()  # activation function
]

model = Model(network)

# model.train(loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True)
model.train(mse, mse_prime, X, Y, 10000, 0.1, False)

# testing
for x, y in zip(X, Y):
    output = model.predict(x)
    print(f'X: {x.flatten()}, y: {y.flatten()}, predicted: {output.flatten()}')