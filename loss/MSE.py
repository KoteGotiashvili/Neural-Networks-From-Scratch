import numpy as np
from loss.Loss import Loss
#Mean Squared Error/Loss
class MSE(Loss):

    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_pred - y_true)**2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):

        # number of samples
        samples = len(dvalues)
        # number of output in every sample, use first sample to count
        outputs = len(dvalues[0])

        # gradient on values
        self.dinputs = -2 * (y_true - dvalues) / samples

        # Normalize gradient
        self.dinputs = self.dinputs / samples