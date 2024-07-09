
class SGD:
    def __init__(self, lr=1):
        """
        Initialize the Stochastic Gradient Descent optimizer.
        Args:
        lr (float): The learning rate for the optimizer. Default is 1.
        """
        self.lr = lr

    def update_params(self, layer):
        """
        Update weights and biases of the layer using stochastic gradient descent.
        Args:
        layer (Dense): The layer whose weights and biases need to be updated.

        Returns:
        None
        """
        # multiply learning rate by derivatives of weights and biases
        layer.weights+= -self.lr * layer.dweights
        layer.biases+= -self.lr * layer.dbiases