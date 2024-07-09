import numpy as np


class SGD:
    def __init__(self, lr=1, decay=0., momentum=0.):
        """
        Args:
        lr (float): The initial learning rate.
        decay (float): The decay rate for the learning rate.
        current_lr(float): updated current learning rate
        iterations: epochs/steps
        """
        self.lr = lr
        self.current_lr=lr
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum


    def pre_update_params(self):
        """
        Update the learning rate at the beginning of each iteration.
        The learning rate is decayed by a factor of 1/(1+decay*iterations)
        to prevent the learning rate from becoming too small, or large.

        Returns:
        None
        """
        if self.decay:
            self.current_lr = self.lr * (1./(1. + self.decay * self.iterations))

    def update_params(self, layer):
        """
        Update weights and biases of the layer using stochastic gradient descent.
        Args:
        layer (Dense): The layer whose weights and biases need to be updated.

        Returns:
        None
        """
        if self.momentum:

            if not hasattr(layer,'weight_momentums'):
                # if layer does not contain momentum arrays, create them filled with zeros
                layer.weight_momentums = np.zeros_like(layer.weights)
                # same for bias
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_lr * layer.dweights
            layer.weight_momentums = weight_updates

            #bias update
            bias_updates = self.momentum * layer.bias_momentums - self.current_lr * layer.dbiases
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.current_lr * layer.dweights
            bias_updates = -self.current_lr * layer.dbiases

        # update weights and biases
        layer.weights += weight_updates
        layer.biases += bias_updates




    def post_update_params(self):
        """
        Reset the gradients of the layer for the next iteration.

        Returns:
        None
        """
        self.iterations += 1