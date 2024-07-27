import numpy as np


class Loss:



    def remember_trainable_layers(self, trainable_layers):
        """
        Stores the trainable layers for later use.

        :param trainable_layers: List of trainable layers.
        :return: None.

        """
        self.trainable_layers = trainable_layers

    # Regularization loss calculation
    def regularization_loss(self):

        # 0 by default
        regularization_loss = 0
        # calculate regularization loss, itarate only on trainable layers
        for layer in self.trainable_layers:

            # L1 regularization - weights
            # calculate only when factor greater than 0
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                                       np.sum(np.abs(layer.weights))

            # L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                                   np.sum(layer.weights *
                                          layer.weights)

            # L1 regularization - biases
            # calculate only when factor greater than 0
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                                   np.sum(np.abs(layer.biases))

            # L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                                   np.sum(layer.biases *
                                          layer.biases)

        return regularization_loss

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y, include_regularization=False):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        #if just data loss return it
        if not include_regularization:
            return data_loss

        # Return loss
        return data_loss, self.regularization_loss()

    def calculate_accumulated(self, *, include_regularization=False):
        """"
        Calculates the accumulated loss and regularization loss.

        :param include_regularization: Whether to include regularization loss in the calculation.
        :return: Tuple containing accumulated loss and regularization loss.
        """

        #calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        # if just data loss return it
        if not include_regularization:
            return data_loss

        #return the data and regularization losses
        return data_loss, self.regularization_loss()

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0