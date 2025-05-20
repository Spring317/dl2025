from layer import Layer
from activation import Activation
from math import exp

class Sigmoid(Activation):
    """Sigmoid activation function."""
    def __init__(self):
        super().__init__(self.sigmoid, self.sigmoid_derivative)

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of the sigmoid function."""
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    
class ReLU(Activation):
    """ReLU activation function."""
    def __init__(self):
        super().__init__(self.relu, self.relu_derivative)

    def relu(self, x):
        """ReLU activation function."""
        return max(0, x)

    def relu_derivative(self, x):
        """Derivative of the ReLU function."""
        return 1 if x > 0 else 0
    
class Softmax(Activation):
    """Softmax activation function."""
    def __init__(self):
        super().__init__(self.softmax, self.softmax_derivative)

    def softmax(self, x):
        """Softmax activation function."""
        exp_x = [exp(i) for i in x]
        sum_exp_x = sum(exp_x)
        return [i / sum_exp_x for i in exp_x]

    def softmax_derivative(self, x):
        """Derivative of the softmax function."""
        s = self.softmax(x)
        return [[s[i] * (1 - s[i]) if i == j else -s[i] * s[j] for j in range(len(s))] for i in range(len(s))]

