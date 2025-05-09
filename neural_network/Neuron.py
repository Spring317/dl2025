import math

class Neuron:
    """This is the re-implementation of a Neuron inside a neural network"""
    def __init__(self, bias, w, x, lr):
        self.bias = bias
        self.w = w
        self.x = x
        self.lr = lr 
        
    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def feed_fwd(self, lr=0.01, iterations=1000):
        # w0, w1, w2 = 0.0, 0.0, 0.0
        # loss_history = []
        zi = 0
        for x, w in zip (self.x, self.w):
            zi += x*w
        zi = zi + 1
        return zi
        # for step in range(iterations):
        #     total_loss = 0.0
        #     dw0, dw1, dw2 = 0.0, 0.0, 0.0

        # for x, y in zip(x_data, y_data):
        #     x0, x1 = x
        #     z = self.w1 * x0 + self.w2 * x1 + self.w0
        #     pred = self.sigmoid(z)
            
        #     return pred
            
            # error = pred - y

            # dw0 += error
            # dw1 += error * x0
            # dw2 += error * x1

            # eps = 1e-15
            # total_loss += - (y * math.log(pred + eps) + (1 - y) * math.log(1 - pred + eps))

        # n = len(x_data)
        # w0 -= lr * dw0 / n
        # w1 -= lr * dw1 / n
        # w2 -= lr * dw2 / n

        # avg_loss = total_loss / n
        # loss_history.append(avg_loss)

        # print(f"Step {step}: Loss = {avg_loss:.6f}, Weights = [{w0:.5f}, {w1:.5f}, {w2:.5f}]")

        # if step > 0 and abs(loss_history[-2] - avg_loss) < 1e-6:
        #     break

        # return loss_history, w0, w1, w2

    