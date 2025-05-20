from layer import Layer

class Activation(Layer): 
    """Abstract class for an activation layer in the CNN model."""
    def __init__(self, activation, derivative):
        self.activation = activation
        self.derivative = derivative
        
    def forward(self, input):
        """Forward pass through the activation layer."""
        self.input = input
        return self.activation(input)

    def backward(self, output_gradient):
        """Backward pass through the activation layer."""
        return self.derivative(self.input) * output_gradient