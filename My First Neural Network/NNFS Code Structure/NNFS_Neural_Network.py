"""This is my implementation of code shown in the Neural Network From Scratch in Python book
I will be implementing their coding principles in my own way"""

import numpy as np
from scipy import signal


class Layer_Dense:
    def __init__(self, num_inputs, num_neurons, weight_lambda_l1=0, weight_lambda_l2=0,
                 bias_lambda_l1=0, bias_lambda_l2=0):
        self.d_weights = None
        self.d_inputs = None
        self.d_biases = None
        self.inputs = None
        self.output = None
        self.weights = np.random.rand(num_inputs, num_neurons) - 0.5  # weights is inputs X outputs instead of the
        # other way to allow me to not have to transpose the whole time
        self.biases = np.random.rand(1, num_neurons) - 0.5
        # Set regularization strength
        self.weight_lambda_l1 = weight_lambda_l1
        self.weight_lambda_l2 = weight_lambda_l2
        self.bias_lambda_l1 = bias_lambda_l1
        self.bias_lambda_l2 = bias_lambda_l2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, d_values):
        """The d_ variables represent the (partial) derivatives of the respective variable"""
        self.d_weights = np.dot(self.inputs.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_lambda_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.d_weights += self.weight_lambda_l1 * dL1
        # L2 on weights
        if self.weight_lambda_l2 > 0:
            self.d_weights += 2 * self.weight_lambda_l2 * self.weights
        # L1 on biases
        if self.bias_lambda_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.d_biases += self.bias_lambda_l1 * dL1
        # L2 on biases
        if self.bias_lambda_l2 > 0:
            self.d_biases += 2 * self.bias_lambda_l2 * self.biases

        # Gradient on values
        self.d_inputs = np.dot(d_values, self.weights.T)


class Layer_Dropout:
    def __init__(self, rate):
        self.d_inputs = None
        self.output = None
        self.binary_mask = None
        self.inputs = None
        self.rate = 1 - rate  # Store rate, we invert it as for example for dropout of 0.1 we need success rate of 0.9

    # Forward pass
    def forward(self, inputs):
        # Save input values
        self.inputs = inputs
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, d_values):
        # Gradient on values
        self.d_inputs = d_values * self.binary_mask


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


class Activation_Leaky_ReLU:
    def __init__(self):
        self.inputs = None
        self.d_inputs = None
        self.output = None

    def forward(self, inputs, alpha=0.01):
        self.inputs = inputs
        self.output = np.maximum(alpha * inputs, inputs)

    def backward(self, d_values, alpha=0.01):
        self.d_inputs = d_values.copy()
        # Zero gradient where input values were negative
        self.d_inputs = np.where(self.inputs > 0, 1, alpha)


class Activation_TanH:
    def __init__(self):
        self.outputs = None
        self.inputs = None
        self.d_inputs = None

    def forwardProp(self, inputs):
        self.inputs = inputs
        self.outputs = np.tanh(inputs)

    def backProp(self, d_values):
        self.d_inputs = d_values.copy()
        self.d_inputs = 1 - pow(np.tanh(self.inputs), 2)
        return


class Loss:
    """This is the parent loss class from which the other loss classes will inherit.
    It takes the predicted value and the actual true answer"""

    def __init__(self):
        self.d_inputs = None

    def regularization_loss(self, layer):
        # 0 by default
        regularization_loss = 0
        # L1 regularization - weights
        # calculate only when factor greater than 0
        if layer.weight_lambda_l1 > 0:
            regularization_loss += layer.weight_lambda_l1 * np.sum(np.abs(layer.weights))
        # L2 regularization - weights
        if layer.weight_lambda_l2 > 0:
            regularization_loss += layer.weight_lambda_l2 * np.sum(layer.weights * layer.weights)
        # L1 regularization - biases
        # calculate only when factor greater than 0
        if layer.bias_lambda_l1 > 0:
            regularization_loss += layer.bias_lambda_l1 * np.sum(np.abs(layer.biases))
        # L2 regularization - biases
        if layer.bias_lambda_l2 > 0:
            regularization_loss += layer.bias_lambda_l2 * np.sum(layer.biases * layer.biases)
        return regularization_loss

    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss

    def forward(self, output, y):
        return 0


class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_predicted, y_true):
        num_samples = len(y_predicted)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_predicted_clipped = np.clip(y_predicted, 1e-7, 1 - 1e-7)
        correct_confidences = None
        if len(y_true.shape) == 1:  # 1D arrays
            correct_confidences = y_predicted_clipped[range(num_samples), y_true]
        elif len(y_true.shape) == 2:  # this is true for 2D arrays
            correct_confidences = np.sum(y_predicted_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, d_values, y_true):
        # Number of samples
        samples = len(d_values)
        # Number of labels in every sample
        # We’ll use the first sample to count them
        labels = len(d_values[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.d_inputs = -y_true / d_values
        # Normalize gradient
        self.d_inputs = self.d_inputs / samples


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossEntropy:
    # Creates activation and loss function objects
    def __init__(self):
        self.d_inputs = None
        self.output = None
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        # Output layer’s activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    def backward(self, d_values, y_true):
        num_samples = len(d_values)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.d_inputs = d_values.copy()
        # Calculate gradient
        self.d_inputs[range(num_samples), y_true] -= 1
        # Normalize gradient
        self.d_inputs = self.d_inputs / num_samples


class Optimizer_SGD:
    """This is the very basic optimizer and the one that Samson Zhang uses
    it can accept and optional learning_rate
    default = 1"""

    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.d_weights
        layer.biases += -self.learning_rate * layer.d_biases


class Optimizer_Adam:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentum = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        # Update momentum with current gradients
        layer.weight_momentum = self.beta_1 * layer.weight_momentum + (1 - self.beta_1) * layer.d_weights
        layer.bias_momentum = self.beta_1 * layer.bias_momentum + (1 - self.beta_1) * layer.d_biases
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentum_corrected = layer.weight_momentum / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentum_corrected = layer.bias_momentum / (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.d_weights ** 2

        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.d_biases ** 2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentum_corrected / (
                np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentum_corrected / (
                np.sqrt(bias_cache_corrected) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


class Layer_Convolution:
    """IMPORTANT to match the syntax of the other components the word kernel is replaced with weight
    this allows high cohesion of the class with the other components"""
    def __init__(self, input_shape=(1, 28, 28), kernel_size=3, depth=1):
        self.d_biases = None
        self.d_inputs = None
        self.d_weights = None
        self.output = None
        self.inputs = None
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.weights_shape = (depth, input_depth, kernel_size, kernel_size)
        self.weights = np.random.randn(*self.weights_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.copy(self.biases)  #initialise the output as the biases for now
        for i in range(self.depth):
            for j in range(self.input_depth):
                temp1 = self.inputs[i]
                temp2 = self.weights[i, j]
                self.output[i] += signal.correlate2d(self.inputs[j], self.weights[i, j], "valid")

    def backward(self, d_values):
        self.d_weights = np.zeros(self.weights_shape)
        self.d_inputs = np.zeros(self.input_shape)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)

        for i in range(self.depth):
            for j in range(self.input_depth):
                self.d_weights[i, j] = signal.correlate2d(self.inputs[j], d_values[i], "valid")
                self.d_inputs[j] += signal.convolve2d(d_values[i], self.weights[i, j], "full")


