import numpy as np
import nnfs

class DataShuffling:
    @staticmethod
    def shuffle_data(X, y):
        nnfs.init()
        keys = np.array(range(X.shape[0]))
        np.random.shuffle(keys)

        X = X[keys]
        y = y[keys]

        return X, y

