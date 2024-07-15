import numpy as np
class Sigmoid:


    def forward(self, inputs):
        """
        Applies the sigmoid activation function to the input.

        :param inputs: Input array.
        :return: Output array after applying the sigmoid activation function.
        """

        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        """
        Calculates the gradient of the sigmoid activation function with respect to its input.

        :param dvalues: Gradient of the loss with respect to the output.
        :return: Gradient of the loss with respect to the input.
        """
        # multiply gradients by sigmoid derivative to measure change in backward pass
        self.dinputs = dvalues * (1- self.output) * self.output

