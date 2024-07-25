from realdataset.DataLoading import DataLoading
from realdataset.PreProcessData import PreProcessData
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
print(X.min(), X.max())