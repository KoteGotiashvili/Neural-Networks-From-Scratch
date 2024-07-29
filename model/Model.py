import numpy as np
from model.LayerInput import LayerInput
from sys import exit
from activation.SoftMax import SoftMax
from loss.CategoricalCrossEntropy import CategoricalCrossEntropy
from softmaxandentropy.Activation_Softmax_Loss_CategoricalCrossEntropy import Activation_Softmax_Loss_CategoricalCrossentropy
import pickle
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
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
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
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        # if output activation is softmax and loss function categorical cross entropy
        # create and object of combined activation and loss function containing faster gradient calculation
        if isinstance(self.layers[-1], SoftMax) and isinstance(self.loss, CategoricalCrossEntropy):
            # create and object of combined activation and loss func
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()


    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
            # Initialize accuracy object
            self.accuracy.init(y)

            # Default value if batch size is not being set
            train_steps = 1

            # Calculate number of steps
            if batch_size is not None:
                train_steps = len(X) // batch_size
                # Dividing rounds down. If there are some remaining
                # data but not a full batch, this won't include it
                # Add `1` to include this not full batch
                if train_steps * batch_size < len(X):
                    train_steps += 1

            # Main training loop
            for epoch in range(1, epochs + 1):

                # Print epoch number
                print(f'epoch: {epoch}')

                # Reset accumulated values in loss and accuracy objects
                self.loss.new_pass()
                self.accuracy.new_pass()

                # Iterate over steps
                for step in range(train_steps):

                    # If batch size is not set -
                    # train using one step and full dataset
                    if batch_size is None:
                        batch_X = X
                        batch_y = y

                    # Otherwise slice a batch
                    else:
                        batch_X = X[step * batch_size:(step + 1) * batch_size]
                        batch_y = y[step * batch_size:(step + 1) * batch_size]

                    # Perform the forward pass
                    output = self.forward(batch_X, training=True)

                    # Calculate loss
                    data_loss, regularization_loss = \
                        self.loss.calculate(output, batch_y,
                                            include_regularization=True)
                    loss = data_loss + regularization_loss

                    # Get predictions and calculate an accuracy
                    predictions = self.output_layer_activation.predictions(
                        output)
                    accuracy = self.accuracy.calculate(predictions,
                                                       batch_y)

                    # Perform backward pass
                    self.backward(output, batch_y)

                    # Optimize (update parameters)
                    self.optimizer.pre_update_params()
                    for layer in self.trainable_layers:
                        self.optimizer.update_params(layer)
                    self.optimizer.post_update_params()

                    # Print a summary
                    if not step % print_every or step == train_steps - 1:
                        print(f'step: {step}, ' +
                              f'acc: {accuracy:.3f}, ' +
                              f'loss: {loss:.3f} (' +
                              f'data_loss: {data_loss:.3f}, ' +
                              f'reg_loss: {regularization_loss:.3f}), ' +
                              f'lr: {self.optimizer.current_learning_rate}')

                # Get and print epoch loss and accuracy
                epoch_data_loss, epoch_regularization_loss = \
                    self.loss.calculate_accumulated(
                        include_regularization=True)
                epoch_loss = epoch_data_loss + epoch_regularization_loss
                epoch_accuracy = self.accuracy.calculate_accumulated()

                print(f'training, ' +
                      f'acc: {epoch_accuracy:.3f}, ' +
                      f'loss: {epoch_loss:.3f} (' +
                      f'data_loss: {epoch_data_loss:.3f}, ' +
                      f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')

                # If there is the validation data
                if validation_data is not None:
                    # Evaluate the model:
                    self.evaluate(*validation_data,
                                  batch_size=batch_size)

        # Evaluates the model using passed-in dataset

    def evaluate(self, X_val, y_val, *, batch_size=None):

        # Default value if batch size is not being set
        validation_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            # Dividing rounds down. If there are some remaining
            # data but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        # Reset accumulated values in loss
        # and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()

        # Iterate over steps
        for step in range(validation_steps):

            # If batch size is not set -
            # train using one step and full dataset
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val

            # Otherwise slice a batch
            else:
                batch_X = X_val[
                          step * batch_size:(step + 1) * batch_size
                          ]
                batch_y = y_val[
                          step * batch_size:(step + 1) * batch_size
                          ]

            # Perform the forward pass
            output = self.forward(batch_X, training=False)

            # Calculate the loss
            self.loss.calculate(output, batch_y)

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(
                output)
            self.accuracy.calculate(predictions, batch_y)

        # Get and print validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        # Print a summary
        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')

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


    def get_parameters(self):

        """
        Returns all trainable layers parameters in form of list of tuples (weights, biases)

        :return: list of tuples
        """
        #list of parameters
        params = []

        for layer in self.trainable_layers:
            params.append(layer.get_parameters())

        return params

    def set_parameters(self, parameters):

        """
        Sets all trainable layers parameters from given list of tuples (weights, biases)

        :param parameters: list of tuples
        """

        #iterate over the parameters and layers and update each layers with each set of the parameters
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    def save_parameters(self, path):

        # open file in the binary write mode and save parameters to it
        with open(path, 'wb') as file:
            pickle.dump(self.get_parameters(), file)
    def load_parameters(self, path):
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

