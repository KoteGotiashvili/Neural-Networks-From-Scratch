import numpy as np

class Model:

    def __init__(self):

        # Initialize an empty list to store the layers of the model.
        # This will be used for forward and backward propagation.
        # layers list
        self.layers = []

    def add (self, layer):
        """
        Add a layer to the model.

        :param layer: Dense layer
        :return: None, just append layer
        """

        self.layers.append(layer)
    def set(self, *, loss, optimizer):
        """
        After adding layers we also want to set loss and optimizer functions

        Set the loss and optimizer for the model.

        :param loss: Loss function to be used
        :param optimizer: Optimizer to be used
        :return: None, just set loss and optimizer
        """
        self.loss = loss
        self.optimizer = optimizer


    def train(self, X, y, *, epochs=1, print_every=100):

        for epoch in range(1, epochs):

            #temporary
            pass