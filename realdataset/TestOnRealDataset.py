from realdataset.DataLoading import DataLoading
from realdataset.DataShuffling import DataShuffling
from realdataset.PreProcessData import PreProcessData
from model.Model import Model
from models.Dense import Dense
from activation.ReLU import ReLU
from activation.SoftMax import SoftMax
from loss.CategoricalCrossEntropy import CategoricalCrossEntropy
from optimizers.Adam import Adam
from accuracy.CategoricalAccuracy import CategoricalAccuracy
import numpy as np
import cv2
import matplotlib.pyplot as plt

data_loader = DataLoading()

data_shuffling = DataShuffling()

data_preprocess = PreProcessData()

X, y, X_test, y_test = data_loader.create_data_mnist('fashion_mnist_images')

# shuffle data
X, y = data_shuffling.shuffle_data(X, y)

# then flatten sample-wise and scale to the range of -1 to 1
X, X_test = data_preprocess.scale_features(X, X_test)

# #lets construct model
# model = Model()
#
# # add layers
# model.add(Dense(X.shape[1], 64))
# model.add(ReLU())
# model.add(Dense(64, 64))
# model.add(ReLU())
# model.add(SoftMax())
#
# # set otpimizer, loss and accuracy objects
# model.set(
#     loss=CategoricalCrossEntropy(),
#     optimizer=Adam(),
#     accuracy=CategoricalAccuracy()
# )
#
# # lets finalize and train
# model.finalize()
#
# model.train(X, y, epochs=3, print_every=10, validation_data=(X_test, y_test), batch_size=64)
#model.evaluate(X_test, y_test)

#retrieve and print parameters
# params = model.get_parameters()
# print(params)

#save parameters
#model.save_parameters('fashion_mnist.parms')
# model = Model.load('fashion_mnist.model')
# model.evaluate(X_test, y_test)
#
# confidences = model.predict(X_test[:5])
# print(confidences)

# Label index to label name relation
fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# Read an image
image_data = cv2.imread('coat.jpg', cv2.IMREAD_GRAYSCALE)

# Resize to the same size as Fashion MNIST images
image_data = cv2.resize(image_data, (28, 28))

# Invert image colors
image_data = 255 - image_data

# Reshape and scale pixel data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

# Load the model
model = Model.load('fashion_mnist.model')

# Predict on the image
confidences = model.predict(image_data)

# Get prediction instead of confidence levels
predictions = model.output_layer_activation.predictions(confidences)

# Get label name from label index
prediction = fashion_mnist_labels[predictions[0]]

print(prediction)
