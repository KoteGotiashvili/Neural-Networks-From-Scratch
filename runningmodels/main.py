import numpy as np
from nnfs.datasets import spiral_data
from models.Dense import Dense
from activation.ReLU import RelU
from activation.SoftMax import SoftMax
X, y = spiral_data(samples=100,
                   classes=3)
# Create Dense Layer with 2 input and 3 output
dense1 = Dense(2,3)

# Initialize Relu to be used with dense
activation1=RelU()

# make forward pass through training spiral data
dense1.forward(X)

# Forward pass through activation function
activation1.forward(dense1.output)

print(activation1.output[:5])

softmax=SoftMax()
softmax.forward([[1,2,3]])
print(softmax.output)