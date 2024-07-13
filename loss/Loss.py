import numpy as np


class Loss:

    def calculate(self,output,y):

        sample_losses = self.forward(output,y)

        data_loss=np.mean(sample_losses)

        return data_loss
    def regularzation_loss(self, layer):

        """
        Calculate regularization loss both L1 and L2

        :return: float
        """

        regularization_loss = 0

        if layer.weight_regularization_L1 > 0:
            regularization_loss += layer.weight_regularization_L1 * np.sum(np.abs(layer.weights))

        if layer.weight_regularization_L2 > 0:
            regularization_loss += layer.weight_regularization_L2 * np.sum(layer.weights ** 2)

        if layer.bias_regularization_L1 > 0:
            regularization_loss += layer.bias_regularization_L1 * np.sum(np.abs(layer.biases))

        if layer.bias_regularization_L2 > 0:
            regularization_loss += layer.bias_regularization_L2 * np.sum(layer.biases ** 2)

        return regularization_loss