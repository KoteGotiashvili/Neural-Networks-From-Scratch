from realdataset.DataLoading import DataLoading
from realdataset.DataShuffling import DataShuffling
from realdataset.PreProcessData import PreProcessData
data_loader = DataLoading()

data_shuffling = DataShuffling()

data_preprocess = PreProcessData()

X, y, X_test, y_test = data_loader.load_mnist_dataset('fashion_mnist_images')

# shuffle data
X, y = data_shuffling.shuffle_data(X, y)

# then flatten sample-wise and scale to the range of -1 to 1
X, X_test = data_preprocess.scale_features(X, X_test)



