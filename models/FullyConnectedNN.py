
def fully_connected_nn(inputs, weights, biases):
    # Output of current layer
    layer_outputs = []

    for neuron_weights, neuron_bias in zip(weights, biases):

        # zerod output because calculate each score differently [1,2,3,4] not summing upp, not cumulative
        neuron_output = 0
        for n_input, weight in zip(inputs, neuron_weights):
            # multiply input by associated weigth and add to the neuron's output
            neuron_output += n_input * weight
        # Add bias
        neuron_output += neuron_bias

        layer_outputs.append(neuron_output)
    print(layer_outputs)


# Lets test this Simple NN
inputs = [1, 2, 3, 4]
weights = [[0.1, 0.2, 0.3, 0.4],
           [0.27, -1.2, -0.276, 1],
           [1, 2, 3, 4]]
biases = [2, 3, 0.7]

fully_connected_nn(inputs, weights, biases)