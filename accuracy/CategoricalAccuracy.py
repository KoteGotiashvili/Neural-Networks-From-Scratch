from accuracy.Accuracy import Accuracy
import numpy as np
# accuracy calculation for classification model
class CategoricalAccuracy(Accuracy):
    def __init__(self, binary=False):
        # check binary mode
        self.binary = binary

    # no init is needed
    def init(self, y):
        pass

    # compare predictions to truth values
    def compare(self, predictions, y):
        """
        If binary mode is not set, we assume we have one-hot encoded labels
        In this case, we compare the predicted class index with the actual class index.
        Otherwise, we compare the predicted probabilities with the actual class probabilities.
        """
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        return predictions == y
