# _____ DENSE LAYER _____
"""
    Connects a set of i input neuron to a set of j output neuron.
    A weight wji connects the jth input neuron to the ith output neuron.
"""
from layer import Layer
import numpy as np  # for matrix multiplication

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(output_size, input_size)
        self.bias = np.random.rand(output_size, 1)
    
    def forward(self, input):
        # Save the input for the backward pass
        self.input = input
        return np.dot(self.weights, self.input) + self.bias  # y = xw + b
    
    def backward(self, output_grad, learning_rate):
        weight_gradient = np.dot(output_grad, self.input.T)  # dL/dw = dL/dy * dy/dw
        self.weights -= learning_rate * weight_gradient  # w = w - lr * dL/dw
        self.bias -= learning_rate * output_grad  # b = b - lr * dL/db
        return np.dot(self.weights.T, output_grad)  # dL/dx = dL/dy * dy/dx