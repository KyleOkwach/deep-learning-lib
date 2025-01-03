from dense import Dense
from activations import Tanh
from losses import mse, mse_prime

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

# training
for e in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        # forward
        output = x
        for layer in network:
            output = layer.forward(output)
        
        # error
        error += mse(y, output)
        
        # backward pass
        output_grad = mse_prime(y, output)
        for layer in reversed(network):
            output_grad = layer.backward(output_grad, learning_rate)
    
    error /= len(X)
    if e % 1000 == 0:
        print(f'epoch {e}, error {error}')

# testing
for x, y in zip(X, Y):
    output = x
    for layer in network:
        output = layer.forward(output)
    print(f'X: {x.flatten()}, y: {y.flatten()}, predicted: {output.flatten()}')