import numpy as np
from nnfs.datasets import spiral_data


class Dense:
    def __init__(self, input, neuron):
        self.weights = 0.01 * np.random.rand(input, neuron)
        self.biases = np.zeros((1, neuron))
        self.output = 0

    def forward(self, inputs):

        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):

        """
         Calculate the gradient of the cost with respect to weights and biases.
         dvalues: gradient of the cost with respect to the output of the layer.
         input: input of the layer.

         Returns:
         dweights: gradient of the cost with respect to the weights
         dbiases: gradient of the cost with respect to the biases

         Note: The dinputs of the previous layer is not calculated in this method.
         It will be calculated in the method `update_weights` in the `NeuralNetwork` class.


        """
        # Calculate gradient of weights and biases

        self.dweights = np.dot(self.inputs.T, dvalues)

        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Calculate gradient of the input
        self.dinputs = np.dot(dvalues, self.weights.T)