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

    # let's create prediction method that will choose the appropriate method for out model
    def predictions(self, outputs):
        return (outputs >= 0.5) * 1

