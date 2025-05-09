from Network import Network
import numpy as np
import os


import matplotlib.pyplot as plt

def main():
    # Define parameters
    bias = 1
    learning_rate = 0.01
    iterations = 1000
    config_path = os.path.join(os.path.dirname(__file__), 'config.txt')
    
    # Sample data for XOR problem
    x_data = np.array([
        [0, 0, 1],  
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    
    y_data = np.array([
        [0],  
        [1],
        [1],
        [0]
    ])
    
    # Initialize network with None for weights (will be initialized by the network)
    network = Network(
        bias=bias,
        w=None,
        x=x_data,
        y=y_data,
        lr=learning_rate,
        iteration=iterations,
        config_path=config_path
    )
    
  
    results = []
    for x in x_data:
        prediction = network.predict(x)
        results.append(prediction)
    
    # Print results
    print("Neural Network Results:")
    print("----------------------")
    for i, (x, y, pred) in enumerate(zip(x_data, y_data, results)):
        print(f"Input: {x[:-1]}, Expected: {y[0]}, Predicted: {pred[0]:.4f}")
    
    
def plot_results(actual, predicted):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(actual)), [y[0] for y in actual], color='blue', label='Actual')
    plt.scatter(range(len(predicted)), [y[0] for y in predicted], color='red', label='Predicted')
    plt.xlabel('Sample Index')
    
    plt.ylabel('Value')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.savefig('results.png')
    plt.close()

if __name__ == "__main__":
    main()