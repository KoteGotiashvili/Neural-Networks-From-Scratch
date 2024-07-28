import numpy as np
from realdataset.DataLoading import DataLoading

class PreProcessData:

    def scale_features(self, X, X_test):

        """
        Scale the features of the MNIST dataset between -1 and 1.

        :return: Scaled image
        """
        X = (X.astype(np.float32) - 127.5) / 127.5
        X_test = (X_test.astype(np.float32) - 127.5) / 127.5

        # reshape to vectors, currently it is 2d
        X = X.reshape(X.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        return X, X_test
