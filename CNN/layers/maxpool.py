from layer import Layer

class MaxPool(Layer):
    """Max pooling layer in the CNN model."""
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, input):
        """Forward pass through the max pooling layer."""
        self.input = input
        output = []
        for i in range(0, len(input), self.pool_size):
            row = []
            for j in range(0, len(input[0]), self.pool_size):
                max_value = max(max(input[i+k][j:j+self.pool_size]) for k in range(self.pool_size))
                row.append(max_value)
            output.append(row)
        return output
    
    def backward(self, output_gradient):
        """Backward pass through the max pooling layer."""
        input_gradient = [[0 for _ in range(len(self.input[0]))] for _ in range(len(self.input))]
        for i in range(0, len(self.input), self.pool_size):
            for j in range(0, len(self.input[0]), self.pool_size):
                max_value = max(max(self.input[i+k][j:j+self.pool_size]) for k in range(self.pool_size))
                for k in range(self.pool_size):
                    for l in range(self.pool_size):
                        if self.input[i+k][j+l] == max_value:
                            input_gradient[i+k][j+l] += output_gradient[i//self.pool_size][j//self.pool_size]
        return input_gradient