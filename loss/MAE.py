import numpy as np

# Mean Absolute Error
class MAE:

    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_pred - y_true), axis=-1)
        return sample_losses


    def backward(self, dvalues, y_true):

        # number of samples
        samples = len(dvalues)
        # number of output in every sample, use first sample to count
        outputs = len(dvalues[0])

        # gradient on values
        self.dinputs = np.sign(y_true - dvalues) / outputs

        # normalize gradient
        self.dinputs = self.dinputs / samples