# _____ ACTIVATION LAYER _____
"""
    Applies an activation function to the input.
"""
from deeplearn.layers.layer import Layer
import numpy as np  # for matrix multiplication

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        # Save the input for the backward pass
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_grad, learning_rate):
        # return output_grad * self.activation_prime(self.input)
        return np.multiply(output_grad, self.activation_prime(self.input))

# _____ ACTIVATION FUNCTIONS _____

"""
    Tanh activation function.
    (exp(x) - exp(-x)) / (exp(x) + exp(-x))
"""
class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x)**2
        super().__init__(tanh, tanh_prime)


"""
    Sigmoid activation function.
    1 / (1 + exp(-x))
"""
class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        sigmoid_prime = lambda x: sigmoid(x) * (1 - sigmoid(x))
        super().__init__(sigmoid, sigmoid_prime)