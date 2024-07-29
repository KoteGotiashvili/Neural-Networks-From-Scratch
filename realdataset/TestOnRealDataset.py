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
data_loader = DataLoading()

data_shuffling = DataShuffling()

data_preprocess = PreProcessData()

X, y, X_test, y_test = data_loader.create_data_mnist('fashion_mnist_images')

# shuffle data
X, y = data_shuffling.shuffle_data(X, y)

# then flatten sample-wise and scale to the range of -1 to 1
X, X_test = data_preprocess.scale_features(X, X_test)

#lets construct model
model = Model()

# add layers
model.add(Dense(X.shape[1], 64))
model.add(ReLU())
model.add(Dense(64, 64))
model.add(ReLU())
model.add(SoftMax())

# set otpimizer, loss and accuracy objects
model.set(
    loss=CategoricalCrossEntropy(),
    optimizer=Adam(),
    accuracy=CategoricalAccuracy()
)

# lets finalize and train
model.finalize()

#model.train(X, y, epochs=3, print_every=10, validation_data=(X_test, y_test), batch_size=64)
#model.evaluate(X_test, y_test)

#retrieve and print parameters
# params = model.get_parameters()
# print(params)

#save parameters
#model.save_parameters('fashion_mnist.parms')
model.load_parameters('./fashion_mnist.parms')
model.evaluate(X_test, y_test)


