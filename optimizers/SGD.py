
class SGD:
    def __init__(self, lr=1, decay=0.):
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
        # multiply learning rate by derivatives of weights and biases
        layer.weights += -self.current_lr * layer.dweights
        layer.biases += -self.current_lr * layer.dbiases

    def post_update_params(self):
        """
        Reset the gradients of the layer for the next iteration.

        Returns:
        None
        """
        self.iterations += 1