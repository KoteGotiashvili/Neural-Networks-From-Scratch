from model.Model import Model
from models.Dense import Dense
from activation.ReLU import ReLU
from activation.Linear import Linear
from loss.MSE import MSE
from optimizers.Adam import Adam
from nnfs.datasets import sine_data

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
print(model.layers)


model.set(
    loss=MSE(),
    optimizer=Adam(learning_rate=0.05, decay=1e-3)
)

#finalize the model, get info about layers
model.finalize()


model.train(X, y, epochs=1000, print_every=100)




