import numpy as np
from model.LayerInput import LayerInput
from sys import exit
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
            #perform forward pass
            output = self.forward(X)
            print(output)
            #temporary
            exit()

    def finalize(self):

        """
        This code creates an input layer and sets next and prev references
        for each layer contained within self.layers

        This method should be called after all layers have been added to the model.

        :return: None, just finalize the model
        """

        #create and set the input layer
        self.input_layer = LayerInput()

        # count all objects, layers
        layer_count = len(self.layers)

        #iterate through objects

        for i in range(layer_count):

            # if it is the first layer, the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            # all layer expect first and last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            # the last layer - the next object is the loss
            # output is models outpit
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # check if layer contains attribute called "weights", if yes it is trainable layer, and add to the list of trainable layers
            # checking weights is enough, we do not need biases
            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])



    def forward(self, X):
        """
        We have information about layers(next,prev), now we can perform forward pass

        """
        # call forward on input layer
        self.input_layer.forward(X)


        # call forward on each layer
        for layer in self.layers:
            layer.forward(layer.prev.output)

        # layer is last objet from list, return its output
        return layer.output
