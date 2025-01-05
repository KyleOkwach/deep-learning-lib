# _____ CONVOLUTIONAL LAYER _____
from layer import Layer
import numpy as np
from scipy import signal

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