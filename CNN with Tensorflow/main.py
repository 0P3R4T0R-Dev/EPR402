import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import time


def setupData(filename):
    print("Loading data...")
    data = pd.read_csv(filename, header=None)
    data = np.array(data)
    np.random.shuffle(data)
    print("Data loaded.")
    return data


def separateDataANDLabel(data):
    m, n = data.shape
    data_train = data.T
    Y_lbl = data_train[0]
    X_data = data_train[1:n]
    X_data = X_data / 255.
    return Y_lbl, X_data


model = np.load("../../TensorFlowModelSaved-6400.npz", allow_pickle=True)
model = model['arr_0']
model = model.item()


testData = setupData("Johan train1225.csv")
predict_labels, predict_dataset = separateDataANDLabel(testData)
predict_dataset_new = []
for image in predict_dataset.T:
    predict_dataset_new.append(np.reshape(image, (35, 35, 1)))
predict_labels = np.reshape(predict_labels, (1, 1))
predict_dataset_new = np.array(predict_dataset_new)

# training=False is needed only if there are layers with different
# behavior during training versus inference (e.g. Dropout).
predictions = model(predict_dataset_new, training=False)
class_names = ['Keegan', 'Stefan', 'Johan', 'David', 'StefanP']
wrongGuesses = 0
for i, logits in enumerate(predictions):
    print(tf.nn.softmax(logits))
    class_idx = tf.math.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[class_idx]
    name = class_names[class_idx]
    if class_idx != int(predict_labels[i]):
        wrongGuesses += 1
    print("Example {} prediction: {} ({:4.1f}%) : ({})".format(i, name, 100 * p, class_names[int(predict_labels[i])]))
print("Number of wrong guesses: ", wrongGuesses)
