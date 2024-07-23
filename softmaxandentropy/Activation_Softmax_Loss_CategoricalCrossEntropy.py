import numpy as np
from activation.SoftMax import SoftMax
from loss.CategoricalCrossEntropy import CategoricalCrossEntropy

class Activation_Softmax_Loss_CategoricalCrossentropy:
    # Creates activation and loss function objects
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = SoftMax()
        self.loss = CategoricalCrossEntropy()

    # Forward pass
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples
