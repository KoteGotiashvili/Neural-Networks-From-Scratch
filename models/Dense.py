import numpy as np
from nnfs.datasets import spiral_data


class Dense:
    def __init__(self, input, neuron,
                 weight_regularization_L1=0,
                 weight_regularization_L2=0,
                 bias_regularization_L1=0, bias_regularization_L2=0):
        self.weights = 0.01 * np.random.rand(input, neuron)
        self.biases = np.zeros((1, neuron))
        self.output = 0
        self.weight_regularization_L1 =weight_regularization_L1
        self.weight_regularization_L2 = weight_regularization_L2
        self.bias_regularization_L1 = bias_regularization_L1
        sel.f.bias_regularization_L2 = bias_regularization_L2

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
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)