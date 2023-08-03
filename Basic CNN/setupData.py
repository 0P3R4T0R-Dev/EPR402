import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def setupData(filename):
    print("Loading data...")
    data = pd.read_csv(filename)
    data = np.array(data)
    np.random.shuffle(data)
    print("Data loaded.")
    return data


def kFoldCrossValidation(data, k):
    splitData = np.array_split(data, k)
    training_sets = []
    testing_sets = []

    for i in range(k):
        testing_set = splitData[i]
        training_set = np.concatenate([splitData[j] for j in range(k) if j != i])
        testing_sets.append(testing_set)
        training_sets.append(training_set)

    return training_sets, testing_sets


def separateDataANDLabel(data):
    m, n = data.shape
    data_train = data.T
    Y_lbl = data_train[0]
    X_data = data_train[1:n]
    X_data = X_data / 255.
    return Y_lbl, X_data


def showSomeData(Y_train, X_train, numToDisplay):
    for j in range(numToDisplay):
        toPrint = []
        for i in range(784):
            toPrint.append(X_train[i][j])
        print(toPrint)
        toPrint = np.array(toPrint)
        current_image = toPrint.reshape((28, 28)) * 255
        print(Y_train[j])
        plt.gray()
        plt.title("Label: " + str(Y_train[j]))
        plt.imshow(current_image, interpolation='nearest')
        plt.show()

# print("Loading data...")
# data = pd.read_csv('train.csv')
# data = np.array(data)
# m, n = data.shape
# print("Data shape: ", data.shape)
# np.random.shuffle(data)  # shuffle before splitting into dev and training sets
#
# data_dev = data[0:1000].T
# Y_dev = data_dev[0]
# X_dev = data_dev[1:n]
# X_dev = X_dev / 255.
#
# data_train = data[1000:m].T
# Y_train = data_train[0]
# X_train = data_train[1:n]
# X_train = X_train / 255.
# _, m_train = X_train.shape
# print("Data loaded.")
