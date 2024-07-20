import numpy as np
from model.LayerInput import LayerInput
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


    def train(self, X, y, *, epochs=1, print_every=10):

        for epoch in range(1, epochs):

            #temporary
            pass

    def finalize(self):

        """
        This code creates an input layer and sets next and prev references
        for each layer contained within self.layers

        This method should be called after all layers have been added to the model.

        :return: None, just finalize the model
        """

        #create and set the input layer
        self.inpute_layer = LayerInput()

        # count all objects, layers
        layer_count = len(self.layers)

        #iterate through objects

        for i in range(layer_count):

            # if it is the first layer, the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.inpute_layer
                self.layers[i].next = self.layers[i+1]

            # all layer expect first and last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i+1]
                self.layers[i].next = self.layers[i+1]

            # the last layer - the next object is the loss
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss