import numpy as np
from model.LayerInput import LayerInput
from sys import exit
from activation.SoftMax import SoftMax
from loss.CategoricalCrossEntropy import CategoricalCrossEntropy
from softmaxandentropy.Activation_Softmax_Loss_CategoricalCrossEntropy import Activation_Softmax_Loss_CategoricalCrossentropy
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
    def set(self, *, loss, optimizer, accuracy):
        """
        After adding layers we also want to set loss and optimizer, accuracy functions

        Set the loss and optimizer, accuracy for the model.

        :param loss: Loss function to be used
        :param optimizer: Optimizer to be used
        :param accuracy: Accuracy to be used
        :return: None
        """
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):

        """
        This code creates an input layer and sets next and prev references
        for each layer contained within self.layers

        This method should be called after all layers have been added to the model.

        :return: None, just finalize the model
        """

        # create and set the input layer
        self.input_layer = LayerInput()

        self.trainable_layers = []
        # count all objects, layers
        layer_count = len(self.layers)
        # iterate through objects
        for i in range(layer_count):

            # if it is the first layer, the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            # all layer expect first and last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            # the last layer - the next object is the loss
            # output is models outpit
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # check if layer contains attribute called "weights", if yes it is trainable layer, and add to the list of trainable layers
            # checking weights is enough, we do not need biases
            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])

        # update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)

        # if output activation is softmax and loss function categorical cross entropy
        # create and object of combined activation and loss function containing faster gradient calculation
        if isinstance(self.layers[-1], SoftMax) and isinstance(self.loss, CategoricalCrossEntropy):
            # create and object of combined activation and loss func
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

    def train(self, X, y, *, epochs=1, print_every=10, validation_data=None):

        #initalize accuracy object
        self.accuracy.init(y)

        for epoch in range(1, epochs+1):
            #perform forward pass
            output = self.forward(X, training=True)

            #calculate loss
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
            loss = data_loss + regularization_loss

            # get predictions and calculate accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            accuracy = self.accuracy.calculate(predictions, y)

            #perform backward pass
            self.backward(output, y)

            # Optimizer Update parameters
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            # print summary
            if not epoch % print_every == 0:
                print(f"Epoch: {epoch}, "
                      f"Loss: {loss:.4f}, "
                      f"Accuracy: {accuracy:.4f},"
                      f"Data Loss:  {data_loss:.4f}",
                      f"Reg_loss: {regularization_loss:.3f}",
                      f"lr:{self.optimizer.current_learning_rate}")
            # for validation data
            if validation_data is not None:
                X_val, y_val = validation_data

                #perform forward pass
                output = self.forward(X_val, training=False)
                # calculate loss
                loss = self.loss.calculate(output, y_val)
                #get prerdicions and caluclate accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, y_val)

                #print summary
                print(f"Validation, Epoch: {epoch}, "
                      f"Loss: {loss:.4f}, "
                      f"Accuracy: {accuracy:.4f}")




    def forward(self, X, training):
        """
        We have information about layers(next,prev), now we can perform forward pass

        """
        # call forward on input layer
        self.input_layer.forward(X, training)


        # call forward on each layer
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # layer is last objet from list, return its output
        return layer.output

    def backward(self, output, y):
        # last is set loss, so first call backward on loss
        self.loss.backward(output, y)

        # then call backward on each layer, reversed order passing dinpuds as parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
