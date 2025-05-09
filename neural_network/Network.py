from Neuron import Neuron
from pseudo_random import PseudoRandom


class Network:
    """
    Forming the network with each Neuron is a logistic regression
    
    Parameters:
    * bias (should be 1)
    * w: initial weights
    * x: input data
    * y: target values 
    * lr: learning rate 
    * iter: the number of iteration needed 
    * config_path: path to network configuration file
    """
    
    def __init__(self, bias, w, x, y, lr, iteration, config_path):
        self.bias = bias 
        self.w = w
        self.x = x 
        self.y = y 
        self.lr = lr 
        self.iter = iteration
        self.config_path = config_path
        self.layers = []
        self.config = self.get_config(config_path)
        self.weights = self.initialize_weights()
        
    def get_config(self, path):
        """Parse the neural network configuration
        
        Parameters:
        path: path of the config.txt
        
        Returns:
        List of integers representing number of nodes in each layer
        """
        with open(path, 'r') as f:
            lines = f.readlines()
        
        config = []
        for line in lines:
            if line.strip():
                config.append(int(line.strip()))
        
        return config
    
    def initialize_weights(self):
        """Initialize weights for all connections in the network"""
        random = PseudoRandom()
        weights = []
        
        for i in range(len(self.config) - 1):
            layer_weights = []
            for j in range(self.config[i + 1]):  # Neurons in next layer
                neuron_weights = [random.random() for _ in range(self.config[i])]  # Weights from previous layer
                layer_weights.append(neuron_weights)
            weights.append(layer_weights)
            
        return weights
    
    def forward(self, input_data):
        """Perform forward propagation through the network
        
        Parameters:
        input_data: Input features
        
        Returns:
        List of activations for each layer
        """
        activations = [input_data]
        current_input = input_data
        
        for i in range(len(self.weights)):
            layer_activations = []
            for j in range(len(self.weights[i])):
                # Remove the None parameter
                neuron = Neuron(self.bias, self.weights[i][j], current_input, self.lr)
                z = neuron.feed_fwd()
                a = neuron.sigmoid(z)
                layer_activations.append(a)
            
            activations.append(layer_activations)
            current_input = layer_activations
            
        return activations
    
    def predict(self, input_data):
        """Make predictions using the network (since feedforward means prediction)
        
        Parameters:
        input_data: Input features
        
        Returns:
        Prediction output
        """
        activations = self.forward(input_data)
        return activations[-1]
