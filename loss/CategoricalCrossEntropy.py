from loss.Loss import Loss
import numpy as np
class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):

        samples=len(y_pred)
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Check if y_true is one-hot encoded or class indices
        if len(y_true.shape) == 1:
            # Class indices, pick the correct class probabilities
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            # One-hot encoded, mask the correct class probabilities
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        else:
            raise ValueError("Unsupported shape for y_true")

        # Compute the losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):

        samples=len(dvalues)

        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues

        #normalize gradient
        self.dinputs = self.dinputs/samples
