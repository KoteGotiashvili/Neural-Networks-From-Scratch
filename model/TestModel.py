from model.Model import Model
from models.Dense import Dense
from activation.ReLU import ReLU
from activation.Linear import Linear
from loss.MSE import MSE
from optimizers.Adam import Adam
from nnfs.datasets import sine_data
from accuracy.RegressionAccuracy import RegressionAccuracy
from loss.BinaryCrossEntropy import BinaryCrossEntropy
from nnfs.datasets import spiral_data
from activation.Sigmoid import Sigmoid
from accuracy.CategoricalAccuracy import CategoricalAccuracy
# create dataset
X, y = sine_data()
model = Model()


model.add(Dense(1, 64))
model.add(ReLU())
model.add(Dense(64,64))
model.add(ReLU())
model.add(Dense(64,1))
model.add(Linear())

# layers
#print(model.layers)


model.set(
    loss=MSE(),
    optimizer=Adam(learning_rate=0.05, decay=1e-3),
    accuracy=RegressionAccuracy()
)

#finalize the model, get info about layers
model.finalize()


#model.train(X, y, epochs=1000, print_every=100)



## Let's do on regression task

X, y = spiral_data(samples=100, classes=2)
X_test, y_test = spiral_data(samples=100, classes=2)

# reshape labels to be a list of lists
# inner list contains either 1 or 0
#print(y)
y = y.reshape(-1, 1)
#print(y)
y_test = y_test.reshape(-1, 1)

#init model
model = Model()

model.add(Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(ReLU())
model.add(Dense(64, 1))
model.add(Sigmoid())

# set loss, optimzier and accuracy
model.set(
    loss=BinaryCrossEntropy(),
    optimizer=Adam(learning_rate=0.05, decay=5e-7),
    accuracy=CategoricalAccuracy(binary=True)
)

#finalize model means remember layer next, prevt, etc
model.finalize()

#train the model
model.train(X, y, epochs=1000, print_every=100, validation_data=(X_test, y_test))




