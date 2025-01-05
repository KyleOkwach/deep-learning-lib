from network import Network
from dense import Dense
from dropout import Dropout
from activations import Tanh
from losses import mse, mse_prime

import numpy as np

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3),  # input layer
    Tanh(),  # activation function
    Dense(3, 1),  # output layer
    Tanh()  # activation function
]

model = Network(network)
model.train(mse, mse_prime, X, Y, 10000, 0.1, False)

# testing
for x, y in zip(X, Y):
    output = model.predict(x)
    print(f'X: {x.flatten()}, y: {y.flatten()}, predicted: {output.flatten()}')