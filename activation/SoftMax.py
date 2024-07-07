import numpy as np

class SoftMax:


    def forward(self, inputs):
        """ Softmax first exponentiates the inputs the power of e(euler number)
        Then it sum ups all exponentiated values and divides each input in order to get probability distribution

        axis=1 specifies column-wise use of list

        keepdims=True makes the shape of the output same as input shape

        We substract the maximum number from inputs in order to avoid exponential function overflow, or dead neurons
        """
        #first substract then exponentiate the power of e
        exp_values=np.exp(inputs - np.max(inputs, axis=1,
                                                keepdims=True))
        print(exp_values)
        probabilities= exp_values/np.sum(exp_values, axis=1,
                                                     keepdims=True)
        self.output=probabilities

    def backward(self, dvalues):
        """
        The backward pass computes the gradient of the loss with respect to the input values
        """

        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):

            single_output = single_output.reshape(-1,1)

            #calculate jacobian matrix of the output

            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # multiply the jacobian matrix by the gradient

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
