import numpy as np
class Dropout:

    def __init__(self, rate):
        """
              Initialize Dropout with a given dropout rate.

              Args:
              - rate (float): Dropout rate, where 0 <= rate < 1.
        """
        #self.rate = 0.7 means that during each training iteration,
        #each neuron has a 70% chance of being retained (not dropped out) and a 30% chance of being dropped out.
        self.rate = 1-rate

    def forward(self, inputs, training):
        """
               Perform forward propagation with dropout.

               Args:
               - inputs (ndarray): Input data or activations from the previous layer.
        """

        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape)/ self.rate

        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        """
               Perform backward propagation through dropout.

               Args:
               - dvalues (ndarray): Gradient of the loss with respect to the outputs.
        """

        self.dinputs = dvalues * self.binary_mask
