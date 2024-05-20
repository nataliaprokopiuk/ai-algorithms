import numpy as np

class ActivationFunc:
    def __init__(self):
        self.alpha = 0.01

    def identity(self, x):
        return x
    
    def identity_prime(self, x):
        return np.ones_like(x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_prime(self, x):
        return np.where(x > 0, 1, 0)
    
    def leaky_relu(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def leaky_relu_prime(self, x):
        return np.where(x > 0, 1, self.alpha)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return np.exp(-x) / np.power((1 + np.exp(-x)), 2)