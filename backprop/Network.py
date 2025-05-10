from Neuron import Neuron
from pseudo_random import PseudoRandom
from math import log
import matplotlib.pyplot as plt

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
    
    def __init__(self, bias, x, y, lr, iteration, config_path):
        self.bias = bias 
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
    
    # def forward(self, input_data, pred = False):
    #     """Perform feed forward through the network
        
    #     Parameters:
    #     input_data: Input features
        
    #     Returns:
    #     List of activations for each layer
    #     """
    #     if pred:
    #         self.weights, losses = self.train()
    #         return self.weights, losses

    
    def backprop(self, input_data, target):
        """Perform back-propagation to train the network

        Parameters:
        input_data: Input features
        target: Target values

        Returns:
        loss: The loss for this sample
        """
        activations = [input_data]
        current_input = input_data

        for i in range(len(self.weights)):
            layer_activations = []
            for j in range(len(self.weights[i])):
                neuron = Neuron(self.bias, self.weights[i][j], current_input, self.lr)
                z = neuron.feed_fwd()
                a = neuron.sigmoid(z)
                layer_activations.append(a)
            activations.append(layer_activations)
            current_input = layer_activations

        output_layer = activations[-1]
        output_error = []

        loss = 0
        for j in range(len(output_layer)):
            y_pred = output_layer[j]
            y_true = target[j]
            
            loss += -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
            
            # Error = predicted - actual
            error = y_pred - y_true
            output_error.append(error)

        current_error = output_error

        for layer_idx in range(len(self.weights) - 1, -1, -1):
            prev_activations = activations[layer_idx]
            current_activations = activations[layer_idx + 1]
            next_error = [0] * len(prev_activations) if layer_idx > 0 else None
            
            for j in range(len(self.weights[layer_idx])):
                delta = current_error[j] * current_activations[j] * (1 - current_activations[j])
                
                for k in range(len(self.weights[layer_idx][j])):
                    # Weight update = learning_rate * delta * activation
                    self.weights[layer_idx][j][k] -= self.lr * delta * prev_activations[k]
                
                if layer_idx > 0:
                    for k in range(len(self.weights[layer_idx][j])):
                        next_error[k] += delta * self.weights[layer_idx][j][k]
            
            current_error = next_error

        return layer_activations, loss

    def train(self):
        """Train the network using backpropagation"""
        for i in range(self.iter):
            total_loss = 0
            avg_losses = []
            for x, y in zip(self.x, self.y):
                layer_activation, loss = self.backprop(x, y)
                total_loss += loss
            
            avg_loss = total_loss / len(self.x)
            avg_losses.append(avg_loss)
            print(f"Iteration {i + 1}/{self.iter}, Loss: {avg_loss:.6f}")

            if avg_loss < 1e-2:
                break
        
        weights = self.weights
        print("Training complete.")
        print(f"Final Weights: {weights}")
        print(f"Final Loss: {avg_loss:.6f}")
        
        return avg_losses, weights
         
                
    def predict(self, input_data, weights):
        """Make predictions using the network (since feedforward means prediction)
        
        Parameters:
        input_data: Input features
        
        Returns:
        Prediction output
        """
        activations = [input_data]
        current_input = input_data
            
        for i in range(len(self.weights)):
            layer_activations = []

            for j in range(len(self.weights[i])):
                neuron = Neuron(self.bias, self.weights[i][j], current_input, self.lr)
                z = neuron.feed_fwd()
                a = neuron.sigmoid(z)
                layer_activations.append(a)
            
            activations.append(layer_activations)
            current_input = layer_activations
            
        return activations[-1]       
    
    
    def plot_loss(self, losses):
        """Plot the loss over iterations"""
        plt.plot(losses)
        plt.title('Loss over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.show()