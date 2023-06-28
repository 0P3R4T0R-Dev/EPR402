"""This is my implementation of code shown in the Neural Network From Scratch in Python book
I will be implementing their coding principles in my own way"""

import numpy as np


class Layer_Dense:
    def __init__(self, num_inputs, num_neurons):
        self.d_inputs = None
        self.d_biases = None
        self.inputs = None
        self.output = None
        self.weights = np.random.rand(num_inputs, num_neurons) - 0.5  # weights is inputs X outputs instead of the
        # other way to allow me to not have to transpose the whole time
        self.biases = np.random.rand(1, num_neurons) - 0.5

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, d_values):
        """The d_ variables represent the (partial) derivatives of the respective variable"""
        self.d_inputs = d_values.copy()
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)
        self.d_inputs = np.dot(d_values, self.weights.T)


class Activation_Softmax:
    def __init__(self):
        self.d_inputs = None
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        # Get un-normalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, d_values):
        """TODO gain better understanding of softmax backpropagation"""
        # Create uninitialized array
        self.d_inputs = np.empty_like(d_values)
        # Enumerate outputs and gradients
        for index, (single_output, single_d_values) in enumerate(zip(self.output, d_values)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.d_inputs[index] = np.dot(jacobian_matrix, single_d_values)


class Activation_ReLU:
    def __init__(self):
        self.inputs = None
        self.d_inputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    def backward(self, d_values):
        self.d_inputs = d_values.copy()
        # Zero gradient where input values were negative
        self.d_inputs[self.inputs <= 0] = 0
