import numpy as np

class Accuracy:

    # calculate accuracy given predictions and ground truth values
    def calculate(self, predictions, y):

        #get comparison results
        comparisons = self.compare(predictions, y)

        # calculate an accuracy
        accuracy = np.mean(comparisons)

        # add accumalted sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy

    def calculate_accumulated(self):
        """"
        Calculates the accumulated accuracy.

        :return: Accumulated accuracy.
        """

        #calculate mean accuracy
        accuracy = self.accumulated_sum / self.accumulated_count

        return accuracy

