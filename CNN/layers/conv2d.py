from layer import Layer

from utils.convolution import correlate2d, convolve2d
from utils.random import PseudoRandom

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        pseudo_random = PseudoRandom()
        self.kernels = [[pseudo_random.gauss(0, 1) for _ in range(kernel_size * kernel_size * input_depth)] for _ in range(depth)]
        self.bias = [[pseudo_random.gauss(0, 1)] for _ in range(depth)]

    def forward(self, input):
        self.input = input
        output_height, output_width = self.output_shape[1], self.output_shape[2]
        self.output = [[[self.bias[i][0] for _ in range(output_width)] for _ in range(output_height)] for i in range(self.depth)]
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = [[[[0 for _ in range(self.kernels_shape[3])] 
                     for _ in range(self.kernels_shape[2])]
                     for _ in range(self.kernels_shape[1])]
                     for _ in range(self.kernels_shape[0])]
        input_gradient = [[[0 for _ in range(self.input_shape[2])]
                  for _ in range(self.input_shape[1])]
                  for _ in range(self.input_shape[0])]

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient