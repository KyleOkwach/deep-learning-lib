from dense import Dense
from activations import Tanh
from losses import mse, mse_prime
from network import Network

import numpy as np

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3),  # input layer
    Tanh(),  # activation function
    Dense(3, 1),  # output layer
    Tanh(),  # activation function
]

epochs = 10000  # number of training iterations
learning_rate = 0.1  # step size for gradient descent

model = Network(network)
model.train(mse, mse_prime, X, Y, epochs, learning_rate, False)

# testing
for x, y in zip(X, Y):
    output = model.predict(x)
    print(f'X: {x.flatten()}, y: {y.flatten()}, predicted: {output.flatten()}')