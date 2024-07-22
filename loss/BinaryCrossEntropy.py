import Loss
import numpy as np
from loss.Loss import Loss

class BinaryCrossEntropy(Loss):

    def forward(self, y_pred, y_true):

        #clip data to prevent division by zero
        # Clip both sides in order to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # calculate binary cross entropy loss
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):

        #num of samples
        samples = len(dvalues)

        #num of labels in every sample
        outputs = len(dvalues[0])

        # clip to prevent divison by zero
        dvalues_clipped = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # calculate gradientdvalues_clipped
        self.dinputs = - (y_true / dvalues_clipped - (1-y_true)/(1 - dvalues_clipped)) / outputs

        # normalize gradient
        self.dinputs = self.dinputs / samples
