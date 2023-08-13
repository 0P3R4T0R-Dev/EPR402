# the goal is the implement Samson Zhang code but have it modularised like in the video
# https://www.youtube.com/watch?v=pauPCy_s0Ok

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time


class DenseLayer:
    def __init__(self, inputSize, outputSize):
        self.input = None
        self.Z = None
        self.A = None
        self.W = np.random.rand(outputSize, inputSize) - 0.5
        self.b = np.random.rand(outputSize, 1) - 0.5

    def forwardProp(self, X):
        self.input = X
        self.Z = np.dot(self.W, self.input) + self.b
        return self.Z

    def backProp(self, dZ2, alpha, reverseActivation):
        dZ1 = dZ2 * reverseActivation
        dW1 = 1 / m * np.dot(dZ1, self.input.T)
        db1 = 1 / m * np.sum(dZ1)
        temp_w = self.W
        self.W = self.W - alpha * dW1
        self.b = self.b - alpha * db1
        return np.dot(temp_w.T, dZ1)


class OutputLayer:
    def __init__(self, inputSize, outputSize):
        self.input = None
        self.Z = None
        self.A = None
        self.W = np.random.rand(outputSize, inputSize) - 0.5
        self.b = np.random.rand(outputSize, 1) - 0.5

    def forwardProp(self, X):
        self.input = X
        self.Z = np.dot(self.W, self.input) + self.b
        self.A = np.exp(self.Z) / sum(np.exp(self.Z))
        return self.Z, self.A

    def backProp(self, Y_true_oneHot, alpha):
        dZ = self.A - Y_true_oneHot  #remember this Y needs to be one hotted
        dW = 1 / m * np.dot(dZ, self.input.T)
        db = 1 / m * np.sum(dZ)
        temp_w = self.W
        self.W = self.W - alpha * dW
        self.b = self.b - alpha * db
        return np.dot(temp_w.T, dZ)


class ActivationReLU:
    @staticmethod
    def forwardProp(X):
        return np.maximum(X, 0.0)

    @staticmethod
    def backProp(X):
        return X > 0


class ActivationTanH:
    @staticmethod
    def forwardProp(X):
        return np.tanh(X)

    @staticmethod
    def backProp(X):
        return 1 - pow(np.tanh(X), 2)


def get_predictions(A2):
    """:returns the  index of the max value in A2
    this is used to find out what the networks prediction is"""
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    """checks to see how accurate the network is by comparing the outputs with each label"""
    return np.sum(predictions == Y) / Y.size


def one_hot(Y):
    """allows the output to be in a one hot encoded format, so we can easily see what the ans is"""
    oneHot = np.zeros((Y.size, Y.max() + 1))
    oneHot[np.arange(Y.size), Y] = 1  # TODO need to investigate this line more
    oneHot = oneHot.T
    return oneHot


data = pd.read_csv('train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape


epochs = 500
learning_rate = 0.2
start_time = time.time()

layer1 = DenseLayer(784, 400)
# layer2 = DenseLayer(522, 300)
layerOut = OutputLayer(400, 10)
RelU = ActivationReLU()
tanH = ActivationTanH()
accuracy = []
acc = 0
for i in range(epochs):
    Z1 = layer1.forwardProp(X_train)
    A1 = tanH.forwardProp(Z1)
    # Z2 = layer2.forwardProp(A1)
    # A2 = tanH.forwardProp(Z2)
    Z_out, layer_output = layerOut.forwardProp(A1)

    gradient1 = layerOut.backProp(one_hot(Y_train), learning_rate)
    # derivative_ReLU = tanH.backProp(Z2)
    # gradient_layer2 = layer2.backProp(gradient1, learning_rate, derivative_ReLU)
    derivative_ReLU = tanH.backProp(Z1)
    gradient_layer1 = layer1.backProp(gradient1, learning_rate, derivative_ReLU)

    Z1 = layer1.forwardProp(X_dev)
    A1 = tanH.forwardProp(Z1)
    # Z2 = layer2.forwardProp(A1)
    # A2 = tanH.forwardProp(Z2)
    Z_out, layer_output = layerOut.forwardProp(A1)
    prediction = get_predictions(layer_output)
    acc = get_accuracy(prediction, Y_dev)
    accuracy.append(acc)
    if i % 10 == 0:
        print("Iteration: ", i)
        print(prediction[:10], Y_dev[:10])
        print("Accuracy: ", acc)

end_time = time.time()
print("Elapsed Time: ", end_time - start_time)

x = range(1, epochs+1)
plt.plot(x, accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Iteration')
plt.show()





