import numpy as np

class RelU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):

        # create copy since we need to modify original one
        self.dinputs=dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.dinputs<=0] = 0

