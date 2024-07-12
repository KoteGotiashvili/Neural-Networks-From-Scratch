import numpy as np

class AdaGrad:
    def __init__(self, lr=0.001, decay=0, epsilon=1e-8):
        self.lr = lr
        self.current_lr=lr
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before any parameters update
    def pre_update_params(self):
        """

        :return:
        """
        if self.decay:
            self.current_lr = self.lr * (1./(1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        """"
         Update the weights and biases using AdaGrad

         Step one: Check if weigth and bias is initialized if not make them zeros using numpy

         Step two: create cache of weights and biases, make 2th power of gradient

         Steo three: update weights and biases based on AdaGrad "formula", approach
        """

        if not hasattr(layer,'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # squared gradients for update
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += - self.current_lr * layer.dweights / (np.sqrt(layer.weight_cache)+ self.epsilon)
        layer.biases -= self.current_lr * layer.dbiases / (np.sqrt(layer.bias_cache)+ self.epsilon)

    def post_update_params(self):
        """
        Reset the squared gradients for the next iteration.

        :return: None
        """
        self.iterations += 1