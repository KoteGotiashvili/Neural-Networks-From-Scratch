from realdataset.DataLoading import DataLoading
from realdataset.PreProcessData import PreProcessData
from realdataset.DataShuffling import DataShuffling
import matplotlib.pyplot as plt
DIR = "./fashion_mnist_images/train"
GET_SOME_IMAGE_FROM_DIR = "./fashion_mnist_images/train/7/0017.png"

# initialize dataloading
dl = DataLoading()

# get image labels, and find how many classes are there
print("image labels, class count")
#DataLoading.get_labels(DIR)

print("visualize data")
#DataLoading.visualize_data(GET_SOME_IMAGE_FROM_DIR)

X, y, X_test, y_test = dl.create_data_mnist('./fashion_mnist_images')

# preprocess data
preprocess_data = PreProcessData()

# scale data between -1 and 1, this is very tough approach but works fine for this case
X, y, X_test, y_test = preprocess_data.scale_features(X, y, X_test, y_test)
# print("get min&max")
#print(X.min(), X.max())

# test suffle data
X, y = DataShuffling.shuffle_data(X, y)
#get labels
print(y[:10])
# now test if it right
plt.imshow(X[1].reshape(28,28))
plt.show()
