# Neural Networks from Scratch

Welcome to the **Neural Networks from Scratch** repository. This project is a comprehensive guide for building neural networks from the ground up, providing detailed explanations and code implementations for every component. This repository is ideal for learners and educators in the fields of artificial intelligence and machine learning.

## Table of Contents

1. [Introducing Neural Networks](#introducing-neural-networks)
2. [Coding Our First Neurons](#coding-our-first-neurons)
3. [Adding Layers](#adding-layers)
4. [Activation Functions](#activation-functions)
5. [Calculating Network Error with Loss](#calculating-network-error-with-loss)
6. [Introducing Optimization](#introducing-optimization)
7. [Derivatives](#derivatives)
8. [Gradients, Partial Derivatives, and the Chain Rule](#gradients-partial-derivatives-and-the-chain-rule)
9. [Backpropagation](#backpropagation)
10. [Optimizers](#optimizers)
11. [Testing with Out-of-Sample Data](#testing-with-out-of-sample-data)
12. [Dropout](#dropout)
13. [Binary Logistic Regression](#binary-logistic-regression)
14. [Regression](#regression)
15. [Model Object](#model-object)
16. [A Real Dataset](#a-real-dataset)
17. [Model Evaluation](#model-evaluation)
18. [Saving and Loading Models and Their Parameters](#saving-and-loading-models-and-their-parameters)
19. [Prediction / Inference](#prediction-inference)


## Introducing Neural Networks
Originally known as artificial neural networks, neural networks are computational 
models inspired by the human brain. They consist of interconnected layers of nodes 
or neurons that process data, enabling machines to learn patterns and make decisions.
They find applications in almost everything.

This is pretty visual representation of neural net (:

![pretty image ](images/neuralnet.jpg)

But this is actual (:

![pretty image ](images/actualneuralnet.png)

More actual is just numbers (:

## Coding Our First Neurons
Step-by-step guide to Fully Connected Artificial Neural Network, layers of neurons, 
and operations like dot products and matrix multiplications using pure python. 
For detailed code examples, refer to the [FullyConnectedNN From Scratch](models/FullyConnectedNN.py)

## Adding Layers
Learn how to add and train dense layers in your neural network models. 
Implementing dense layers, which are fully connected layers where each neuron 
receives input from all neurons in the previous layer, has been pivotal in 
deep learning advancements.Dense layers allow for complex pattern 
learning and model flexibility. It is implemented from scratch using Numpy on there [Dense From Scratch](models/Dense.py)

## Activation Functions
**ReLU** - is widely used for its simplicity, computational efficiency, and effectiveness in handling vanishing gradients(Gradients become extremely small) in deep networks. [ReLU From Scratch](activation/ReLU.py)

**SoftMax** - is essential for multi-class classification problems, converting logits(numbers) into probabilities, making it easier to interpret model outputs and compare class scores. [SoftMax From Scratch](activation/SoftMax.py)
## Calculating Network Error with Loss
**Categorical Cross-Entropy** -Categorical cross-entropy loss is widely used in classification tasks, it measures the difference between true and predicted probabilities.It is beneficial for multi-class problems. Lower loss indicates higher confidence in correct predictions (e.g., 0.99 probability yields low loss, approaching zero).
[Categorical Cross-Entropy From Scratch](loss/CategoricalCrossEntropy.py)
## Introducing Optimization
The next step is determining how to adjust the weights and biases 
to decrease the loss. Initially, one might think of changing them randomly, 
but experiments show this is not effective. Instead, 
we use calculus to optimize these adjustments. 😊 

You can see the results after using the random approach
![train neural net using random approach](images/randomapproach.png)

## Derivatives
The random approach does not yield good results. 
Instead, we use calculus, particularly derivatives,
to measure changes in variables. This method involves using the
slopes of functions to optimize our neural networks. Basically we calculate
how much each singular weight and bias change the loss value. For that we are going to
use **Gradient Descent** which is result of the calculation of the partial derivatives and then
we'll **Backpropagate** it using **Chain Rule**.

More About Calculus: https://en.wikipedia.org/wiki/Calculus

## Gradients, Partial Derivatives, and the Chain Rule
**Gradient**: The gradient is a vector consisting of the partial derivatives with respect to each of the inputs.

**Partial Derivatives**: Measures how much impact a single input has on a function's output

If the function is following ![gradient](images/function.png)

Then Partial Derivatives and gradient will be following
![gradient](images/gradient.png) 

**Chain Rule**: The chain rule is used in neural networks during backpropagation 
to compute gradients of the loss function. Loss is a big function
or a chain of functions of multiple inputs.

For instance, if we have two function z=f(x) and y=g(z)
We can represent this equation as y=g(f(x)), As you can see there 
is function inside the function
and as calculus says derivative of y will be y'=g(f(x))' * f(x)'

## Backpropagation
Backpropagation is a key algorithm in training neural networks, 
enabling weight adjustment to minimize error. It measures the gradient of 
the loss function with respect to each weight by propagating errors backward 
through the network. Essential for gradient descent optimization, it enhances 
model accuracy. 

Step 1 -> Compute Gradient of Loss Function

Step 2 -> Backpropagate the Gradient

step 3 -> Adjust Weights

**Categorical Cross-Entropy Loss Derivative:**

Full solution ![gradient](images/entropy.png)
Code On there: [CategoricalCrossEntropy](loss/CategoricalCrossEntropy.py)

**Softmax Derivative:** 

Full Solution:
![gradient](images/softmax.png)
Code On there: [SoftMax](activation/SoftMax.py)

## Optimizers

**Stochastic Gradient Descent (SGD):** Once we have calculated gradients we can use 
this information to adjust weights and biases to decrease the loss. I'm going to implement
SGD where after each step/epoch gradient is changed by small portion in order to decrease loss.
[SGD](optimizers/SGD.py)

**Learning Rate:** Lr is hyperparameter if it is too low small updates to
derivatives cause the model to stuck in a local minimum, If it is high 
loss will start jumping, and results weired loss for each epoch, 
No more words just look at the image (:

![Learning Rate](images/leraningrate.png)

In most cases we want to start with a larger learning rate and decrease the learning rate over time/steps/epochs.
Commonly used solution explore various learning rate during training is to implement **Learning Rate Decay**

This is "formula" for that: ![LRDecay](images/lrdecay.png)

step increase causes learning rate decrease, lets visualize it for 20 epoch
![after20epoch](images/20stepdecay.png)

This approach is fair but only follows gradient updates without any additional 
logic that could potentially help the model find the global minimum to the loss function. 
One option for improving SGD is to introduce **MOMENTUM**

**Momentum:** Creates a rolling average of gradients over some number 
of updates and uses this average with the unique gradient at each step.
Momentum may still point toward the global gradient descent direction. The 
update contains a portion of the gradient from preceding steps as our momentum
and only a portion of the current gradient.

**AdaGrad:** AdaGrad is short for adaptive gradient. Main idea is to
normalize updates made to the features. During the training process, some weights
can rise significantly, while others tend to not change by much. AdaGrad
provides a way to normalize parameter updates by keeping a history of previous updates
**The bigger the sum of the update is, in either direction, the smaller updates are made further in training.**
Which lets less-frequently updated parameters to keep-up with changes, utilizing more neurons for training.

This Is AdaGrad In Nutshell:
![after20epoch](images/adagrad.png)
**Cache:** holds history of squared gradients.
**parm_updates:** is function of the learning rate multiplied by the gradient and then 
divided by the sqrt of root of the cache plus some **epsilon**(epsilon value is usually a small such as 1e-8, avoid division by zero)

AdaGrad(Adaptive Gradient) From Scratch:[AdaGrad](optimizers/AdaGrad.py)


**RMSProp:** Full name is **Root Mean Square Propagation**. It is similar to AdaGrad but calculation
happens in different way, where AdaGrad calculates cache as squared gradients, RMSProp have following approach:
![rmsprop](images/rmsprop.png)

In this way learning rate changes are smoother. **rho(hyperparameter):** is the cache memory decay rate. 


**Adam:** Short for **Adaptive Momentum** is currently most widely-used optimizer and is built
atop RMSProp, with the momentum concept from SGD added back in.Adam optimizer works well with large 
datasets and sparse data, helping models learn faster and more reliably. 

**It is usually good to try Adam optimizer first and then others. Choosing optimizer
depends on problem type but if problem is too complex just try different ones
and choose the best (:**
## Testing with Out-of-Sample Data
Out-of sample data is just unseen data by neural net. Our task is to minimize loss
and **generalize** the model, this is where regularization comes into play. L1 and L2 regularization
is used like penalty and added to overall loss.

**L1 Regularization:** It is sum of all the absolute values for weights and biases.

**L2 Regularization:** Is the sum of the squared weights and biases.

**L1 regularization is rarely used alone because of its nature of linearity, It 
mostly used with L2.**

**Overall Loss = DataLoss + L1weight + L2bias + L2weight + L2bias**

**Regularization losses are calculated separately **

This is for forward pass: [L1 and L2 regularization loss](loss/Loss.py)

Implementing this means that overall loss is changed which must be accounting in to backpropagation.
We need to cover derivatives of L1 nad L2 regularization.

L2 regularization derivative is straightforward: ![rmsprop](images/L2.png)

L1 is a bit of complicated: ![rmsprop](images/L1.png) This simply happens because of absolute value.

There is L1 and L2 taking into consideration for backward pass: [backward](models/Dense.py)


## Dropout
Another approach for neural net regularization is adding a **Dropout Layer**. Idea
is that to prevent neural net from becoming too dependent on any neuron. This is very straightforward
you should disable some percentage neurons during every froward pass. Dropout does not
disable neurons it just set them to zero. **While dropout helps neural net generalize it's not 
something that we want to utilize when predicting. **

**Dropout From Scratch:** [Dropout](dropout/Dropout.py)

## Binary Logistic Regression
**Binary Logistic Regression:** Is used when outcome is binary. For example, Spam detection,
I help us to distinguish between whether message is spam or not. 0 or 1. It is used with Sigmoid 
Activation function, which approaches both maximum and minimum values exponentially fast, this property 
makes the Sigmoid activation function a good candidate to apply to the final layer's output.

Graph and function of Sigmoid:

![Sigmoid](images/sigmoid.png)

Derivative of sigmoid:

![SigmoidDerivative](images/sigmoidderivative.png)

Sigmoid Implementation: [Sigmoid From Scratch](activation/Sigmoid.py)

**Binary Cross-Entropy Loss:** Binary cross entropy loss measures how well
predicted probabilities match true binary labels. It penalizes deviations 
using logarithmic loss. After loss is calculated, we need to find mean of this loss
to find average loss.

For backward pass there is Binary Cross-Entropy Loss Derivative:

![binarycrossentropy](images/binarycrossentropy.png)

Then we need to find mean of this, and using chain rule we will multiply it by 1/J.

Full code for forward/backward pass: [Binary Cross-Entropy From Scratch](loss/BinaryCrossEntropy.py)



## Regression
I've been working with classification models so far, where we try to determine
what something is based on probabilities. In case of **Regression** we are trying to
predict some number, for example what stock price will be in 5 years from now. Approach 
is different, result is different, and consequently we need different ways to measure loss, also
we have different data, and we have to approach it in other way.

**Linear Activation:** I'm going to use linear activation function fo the output layer.

Code: [Binary Cross-Entropy From Scratch](activation/Linear.py)

Two methods for calculating loss in regression are MSE nad MAE

**Mean Squared Error/Loss(MSE):** It is simply square of differences between
predicted and actual, true values of single outputs and average those squared values.
![MSE](images/mse.png)

Derivative is very straightforward:
![MSEDerivative](images/msederivative.png)

**MSE Code:** [Mean Squared Error From Scratch](loss/MSE.py)

**MAE:** Mean Absolute Error is the absolute difference between the predicted and true values in a single output
and average those absolute values.
![MAE](images/MAE.png)

Derivative of MAE:  ![MAE](images/MAEDerivative.png)

**MAE Code**: [Mean Absolute Error From Scratch](loss/MAE.py)]

**In Practice MAE loss is used less frequently than MSE loss**

**Mean Absolute Error (MAE) is simple to calculate and handles outliers well but is not differentiable at zero. Mean Squared Error
(MSE) is sensitive to outliers and penalizes larger errors more due to squaring.**



**Accuracy In Regression:**

**There's no perfect way to show accuracy with regression**

In this case we want to consider predicted value to be correct or close enough to actual value.


## Model Object

It makes more sense to make model class which will be object itself, we want to add layers, activation functions, etc
object class makes it easier to use and understand.

We are going to need modify our code and also we need previous layer output to perform forward/backward pass.

All I'm doing here is to create something like Deep Learning framework, like PyTorch.

This is model class code, which I think I can call **Deep Learning Framework** [Deep Learning Framework Main class](model/Model.py)

## A Real Dataset
**Our Framework is ready to test on real dataset**

In this project, I will test my model using the MNIST clothing dataset. 
I will implement all steps from scratch, including data shuffling and image dimension preparation, 
without relying on frameworks like PyTorch. For image manipulation, I will use OpenCV, and 
for visualization, I will use matplotlib.

Download MNIST dataset: [Download/Unzip Dataset ](realdataset/DataPreparation.py)

Load, visualize, and turn into numbers existing data: [DataLoading](realdataset/DataLoading.py)


**Data Shuffling:** Our dataset currently consists samples and their target classification. But
we need to shuffle it in order to avoid overfitting. If we train neural net seperately on data it will be 
biased towards each class and it will not generalize well.

There is table of data representation:
![MSEDerivative](images/labels_table.png)

### Batches
So far we've trained models by feeding the entire dataset as a single "batch". But in most cases
AI and ML engineers have a lot of dataset to handle, even terabytes and doing one forward pass 
will eat memory (: also it is not good practice, when doing batches it sum ups losses and accuracies and helps
to generalize the model. Common batch sizes range between 32 and 128 samples. Each batch being trained
is referred to as a **step**. We can calculate number of steps **by dividing the number of samples by the batch size.**
**Integer division rounds down, thus, if there are any samples left, we will add 1 to form the last batch with the remainder**

Updated loss: [Updated Loss Class](loss/Loss.py)

Updated Accuracy: [Updated Accuracy Class](accuracy/Accuracy.py)

Modify train method for batch suit:[Updated Train](model/Model.py)

Now let's train model on real dataset: [Test On Real Dataset](model/TestOnRealDataset.py)

## Model Evaluation
**SOON 🔜  ** Methods for evaluating neural network models to ensure they perform well on unseen data.

## Saving and Loading Models and Their Parameters
**SOON 🔜  ** Techniques for saving and loading models and their parameters for reuse.

## Prediction / Inference
**SOON 🔜  ** Guidelines for making predictions and performing inference with trained neural network models.



---

This repository aims to be a thorough, hands-on guide to neural networks. Happy learning and coding!
