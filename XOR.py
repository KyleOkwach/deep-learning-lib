from deeplearn.model import Model
from deeplearn.layers import Dense
from deeplearn.layers.activations import Tanh
from deeplearn.utils import backend, save_model
from deeplearn.loss import mse, mse_prime

import numpy as np

backend.set_cuda(use_cuda=False)

X = backend.from_numpy(np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1)))
Y = backend.from_numpy(np.reshape([[0], [1], [1], [0]], (4, 1, 1)))

network = [
    Dense(2, 3),  # input layer
    Tanh(),  # activation function
    Dense(3, 1),  # output layer
    Tanh()  # activation function
]

model = Model(network)


# model.train(loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True)
model.train(mse, mse_prime, X, Y, epochs=10000, learning_rate=0.1, verbose=False)

model.save('models/xor.pkl')

# testing
for x, y in zip(X, Y):
    output = model.predict(x)
    print(f'X: {x.flatten()}, y: {y.flatten()}, predicted: {output.flatten()}')