import numpy as np
from scipy import signal

# _____ BASE LAYER _____
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        pass

    def backward(self, output_grad, learning_rate):
        pass


# _____ DENSE LAYER _____
"""
    Connects a set of i input neuron to a set of j output neuron.
    A weight wji connects the jth input neuron to the ith output neuron.
"""
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


# _____ CONVOLUTIONAL LAYER _____
"""
    A convolutional layer applies a set of kernels to the input.
"""
class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        """
            input shape: (depth, height, width)
            kernel size: size of the kernel
            depth: number of kernels
        """
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)
    
    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                # Correlate the input with the kernel
                # The correlate function is not commutative, therefore the order of the arguments matters
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], mode='valid')
        return self.output

    def backward(self, output_grad, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_grad[i], mode='valid')
                input_gradient[j] += signal.convolve2d(output_grad[i], self.kernels[i, j], mode='full')
        
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_grad
        return input_gradient


# _____ RESHAPE LAYER _____
"""
    Reshape the input tensor.
"""
class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_grad, learning_rate):
        return np.reshape(output_grad, self.input_shape)
