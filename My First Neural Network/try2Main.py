# goal is to break the code by Samson Zhang into grouped functions
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time



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


def init_params():
    """
    initialises all the weights and biased to random numbers between -0.5 and 0.5
    """
    W1 = np.random.rand(10, 784) - 0.5  # this is assuming that the weights will be going from 784 inputs to 10 outputs
    b1 = np.random.rand(10, 1) - 0.5  # there are only 10 nodes in the first hidden layer
    W2 = np.random.rand(10, 10) - 0.5  # There are only 10 nodes in the second hidden layer
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


def ReLU(inputs):
    return np.maximum(inputs, 0)  # this returns the max of zero or the input (does the comparison element-wize)


def softMax(inputs):
    return np.exp(inputs) / sum(np.exp(inputs))


def forward_layer1(W, b, X):
    Z = np.dot(W, X) + b
    A = ReLU(Z)
    return Z, A


def forward_layer2(W, b, input_previousLayer):
    Z = np.dot(W, input_previousLayer) + b
    A = softMax(Z)
    return Z, A


def one_hot(Y):
    """allows the output to be in a one hot encoded format, so we can easily see what the ans is"""
    oneHot = np.zeros((Y.size, Y.max() + 1))
    oneHot[np.arange(Y.size), Y] = 1  # TODO need to investigate this line more
    oneHot = oneHot.T
    return oneHot


def derivative_ReLU(inputs):  # this simple function is actually the ReLU derivative for backpropagation
    return inputs > 0


def back_prop_layer2(W2, b2, Z2, A1, A2, Y, alpha):
    oneHot = one_hot(Y)
    dZ2 = A2 - oneHot
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2)
    tempW = W2
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W2, b2, np.dot(tempW.T, dZ2)


def back_prop_layer1(W1, b1, Z1, dZ2, X, alpha):
    dZ1 = dZ2 * derivative_ReLU(Z1)
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1)
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    return W1, b1


def get_predictions(A2):
    """:returns the  index of the max value in A2
    this is used to find out what the networks prediction is"""
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    """checks to see how accurate the network is by comparing the outputs with each label"""
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iterations, alpha):
    """This is the function where the actual network runs in"""
    W1, b1, W2, b2 = init_params()
    acc = 0
    for i in range(iterations):
        Z1, A1 = forward_layer1(W1, b1, X)
        Z2, A2 = forward_layer2(W2, b2, A1)
        acc = get_accuracy(get_predictions(A2), Y)
        accuracy.append(acc)
        if i % 10 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", acc)
        W2, b2, dZ2 = back_prop_layer2(W2, b2, Z2, A1, A2, Y, alpha)
        W1, b1 = back_prop_layer1(W1, b1, Z1, dZ2, X, alpha)
    return acc, alpha



start_time = time.time()
accuracy = []
epochs = 250
accc, learning_rate = gradient_descent(X_train, Y_train, epochs, 0.7)
end_time = time.time()





import csv

data = [
    ['acc', accc],
    ['alpha', learning_rate]
]
filename = 'data.csv'

with open(filename, 'a', newline='') as file:
    writer = csv.writer(file)
    for row in data:
        writer.writerow(row)
print("DONE")


print("Elapsed Time: ", end_time - start_time)

x = range(1, epochs + 1)

plt.plot(x, accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Iteration')
plt.show()
