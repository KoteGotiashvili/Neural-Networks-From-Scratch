import numpy as np
from nnfs.datasets import spiral_data


class Dense:
    def __init__(self, input, neuron):
        self.weights = 0.01 * np.random.rand(input, neuron)
        self.biases = np.zeros((1, neuron))
        self.output = 0

    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.biases
