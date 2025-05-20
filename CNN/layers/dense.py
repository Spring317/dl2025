import math
from CNN.utils.random import PseudoRandom
from CNN.utils.matrix_calculation import dot_product, matrix_transpose
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        pseudo_random = PseudoRandom()
        self.weights = [[pseudo_random.gauss(0, 1) for _ in range(input_size)] for _ in range(output_size)]
        self.bias = [[pseudo_random.gauss(0, 1)] for _ in range(output_size)]

    def forward(self, input):
        self.input = input
        # Use dot product utility function
        output = dot_product(self.weights, input)
        
        # Add bias
        for i in range(len(output)):
            output[i][0] += self.bias[i][0]
            
        return output

    def backward(self, output_gradient, learning_rate):
        # Calculate weights gradient
        weights_gradient = [[0 for _ in range(len(self.weights[0]))] for _ in range(len(self.weights))]
        for i in range(len(self.weights)):
            for j in range(len(self.weights[0])):
                weights_gradient[i][j] = output_gradient[i][0] * self.input[j][0]
        
        # Calculate input gradient using transpose and dot product
        weights_T = matrix_transpose(self.weights)
        input_gradient = dot_product(weights_T, output_gradient)
        
        # Update weights and bias
        for i in range(len(self.weights)):
            for j in range(len(self.weights[0])):
                self.weights[i][j] -= learning_rate * weights_gradient[i][j]
            self.bias[i][0] -= learning_rate * output_gradient[i][0]
        
        return input_gradient