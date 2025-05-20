class Layer:
    """Abstract class for a layer in the CNN (not Cable News Network but Convolution Neural Network) model."""
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, input):
        """Forward pass through the layer."""
        pass
    def backward(self, output_gradient):
        """Backward pass through the layer."""
        pass