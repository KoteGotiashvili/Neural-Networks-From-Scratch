import numpy  as np

class Linear:

    def forward(self, inputs):

        # does not modify input and pass to output y=x
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        # derivative is 1 1 * dvalues = dvaluess + the chain rule
        self.dinputs = dvalues.copy()

    # let's create prediction method that will choose the appropriate method for out model
    def predictions(self, outputs):
        return outputs