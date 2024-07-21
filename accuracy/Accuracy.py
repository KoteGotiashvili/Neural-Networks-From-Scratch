import numpy as np

class Accuracy:

    # calculate accuracy given predictions and ground truth values
    def calculate(self, predictions, y):

        #get comparison results
        comparisons = self.compare(predictions, y)

        # calculate an accuracy
        accuracy = np.mean(comparisons)

        return accuracy