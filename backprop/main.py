from Network import Network
# import numpy as np
# import os
from data_reader import DataReader


import matplotlib.pyplot as plt

def main():
    # Define parameters
    bias = 1
    learning_rate = 0.1
    iterations = 1000000
    config_path = 'config.txt'
    # datareader = DataReader('/home/spring/dl2025/loan.csv')
    x_data = [
        [0, 0, 1],  
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    
    y_data = [
        [0],  
        [1],
        [1],
        [0]
    ]
    
    # x_data, y_data = datareader.read_csv_file()
    # y_data = [[y] for y in y_data]
    # print(y_data)
    
    # Initialize network with None for weights (will be initialized by the network)
    network = Network(
        bias=bias,
        x=x_data,
        y=y_data,
        lr=learning_rate,
        iteration=iterations,
        config_path=config_path
    )
    
  
    results = []
    weights = network.train()
    
    for x in x_data:
        prediction = network.predict(x, weights=weights)
        results.append(prediction)
    
    # print(f"results: {results}")  
    # network.plot_loss(loss)
    # # Print results
    print("Neural Network Results:")
    print("----------------------")
    
    for i, (x, y, pred) in enumerate(zip(x_data, y_data, results)):
        print(f"Input: {x[:-1]}, Expected: {y[0]}, Predicted: {pred[0]:.4f}")
    with open('result.txt', 'w') as f:
        for i, (x, y, pred) in enumerate(zip(x_data, y_data, results)):
            f.write(f"Input: {x[:-1]}, Expected: {y[0]}, Predicted: {pred[0]:.4f}\n")
    
    
# def plot_results(actual, predicted):
#     """Plot actual vs predicted values"""
#     plt.figure(figsize=(10, 6))
#     plt.scatter(range(len(actual)), [y[0] for y in actual], color='blue', label='Actual')
#     plt.scatter(range(len(predicted)), [y[0] for y in predicted], color='red', label='Predicted')
#     plt.xlabel('Sample Index')
    
#     plt.ylabel('Value')
#     plt.title('Actual vs Predicted Values')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('results.png')
#     plt.close()

if __name__ == "__main__":
    main()