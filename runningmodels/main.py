import numpy as np
from nnfs.datasets import spiral_data
from models.Dense import Dense
from activation.ReLU import RelU
from activation.SoftMax import SoftMax
X, y = spiral_data(samples=100,
                   classes=3)
# Create Dense Layer for ReLU
dense1 = Dense(2,3)

#Create dense 2 for softmax, (3,3) because we take output from previous layer (2,3) -> (2,3) @ (3,3)
dense2 = Dense(3,3)

# Initialize Relu to be used with dense
activation1=RelU()

#Initialize SoftMa
activation2=SoftMax()

# make forward pass through training spiral data
dense1.forward(X)

# Forward pass through activation function
activation1.forward(dense1.output)

#Make forward through second dense layer
dense2.forward(activation1.output)

#make forward pass through activation function(softmax)
activation2.forward(dense2.output)

# pirnt ReLU results
print(activation1.output[:5])

# Print Softmax Results, After That developer in many cases apply ARGMAX(to get most probable output, just choose max prob val)
print(activation2.output[:5])

