import numpy as np
from nnfs.datasets import spiral_data
from models.Dense import Dense
from activation.ReLU import RelU
from activation.SoftMax import SoftMax
from loss.CategoricalCrossEntropy import CategoricalCrossEntropy
from softmaxandentropy.Activation_Softmax_Loss_CategoricalCrossEntropy import Activation_Softmax_Loss_CategoricalCrossentropy
from optimizers.SGD import SGD
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

#loss
loss_function=CategoricalCrossEntropy()

# Initialize SGD optimizer
optimizer = SGD(lr=0.1)

# make forward pass through training spiral data
dense1.forward(X)

# Forward pass through activation function
activation1.forward(dense1.output)

#Make forward through second dense layer
dense2.forward(activation1.output)

#make forward pass through activation function(softmax)
activation2.forward(dense2.output)

# pirnt ReLU results
print("ReLU results")
print(activation1.output[:5])

# Print Softmax Results, After That developer in many cases apply ARGMAX(to get most probable output, just choose max prob val)
print("Softmax Results")
print(activation2.output[:5])

#loss
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

#perform a forward pass through the activation/loss function
#takes the output of secod dense here and returns loss
loss = loss_activation.forward(dense2.output, y)

# output of few samples
print("output of loss_activation")
print(loss_activation.output[:5])

#loss
print("loss: ", loss)

#calculate accuracy from output of activation 2 and targets
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y=np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

#print accuracy
print('acc: ', accuracy)

#Backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# Update weights and biases
optimizer.update_params(dense1)
optimizer.update_params(dense2)


# Print gradients
print("Print gradients")
print(dense1.dweights)
print()
print(dense1.dbiases)
print()
print(dense2.dweights)
print()
print(dense2.dbiases)





