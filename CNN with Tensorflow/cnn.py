"""### Import TensorFlow"""

import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import time

"""### Load My data

"""


def setupData(filename):
    print("Loading data...")
    data = pd.read_csv(filename)
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


data = setupData("train6400.csv")

predict_labels, predict_dataset = separateDataANDLabel(data[:100])
y_dev, x_dev = separateDataANDLabel(data[100:2000])
y_train, x_train = separateDataANDLabel(data[2000:])
x_train_new = []
x_dev_new = []
predict_dataset_new = []

print(predict_labels.shape)
print(y_dev.shape)
print(y_train.shape)

y_train = np.reshape(y_train, (22999, 1))
y_dev = np.reshape(y_dev, (1900, 1))
predict_labels = np.reshape(predict_labels, (100, 1))
for i, image in enumerate(x_train.T):
    x_train_new.append(np.reshape(image, (80, 80, 1)))
for image in x_dev.T:
    x_dev_new.append(np.reshape(image, (80, 80, 1)))
for image in predict_dataset.T:
    predict_dataset_new.append(np.reshape(image, (80, 80, 1)))

x_train_new = np.array(x_train_new)
x_dev_new = np.array(x_dev_new)
predict_dataset_new = np.array(predict_dataset_new)
np.savez("../../predict_dataset_new.npz", predict_dataset_new, predict_labels)

print(x_dev_new.shape)
print(x_train_new.shape)

class_names = ['Keegan', 'Stefan', 'Johan', 'David', 'StefanP']

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(x_train_new[i])
#     # The CIFAR labels happen to be arrays,
#     # which is why you need the extra index
#     plt.xlabel(class_names[y_train[i][0]])
# plt.show()

"""### Create the convolutional base

The 6 lines of code below define the convolutional base using a common pattern: a stack of [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D) and [MaxPooling2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D) layers.

As input, a CNN takes tensors of shape (image_height, image_width, color_channels), ignoring the batch size. If you are new to these dimensions, color_channels refers to (R,G,B). In this example, you will configure your CNN to process inputs of shape (32, 32, 3), which is the format of CIFAR images. You can do this by passing the argument `input_shape` to your first layer.
"""

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 1)))
model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5))

"""Here's the complete architecture of your model:"""

model.summary()

"""The network summary shows that (4, 4, 64) outputs were flattened into vectors of shape (1024) before going through two Dense layers.

### Compile and train the model
"""

model.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
start = time.time()
history = model.fit(x_train_new, y_train, batch_size=200, epochs=100,
                    validation_data=(x_dev_new, y_dev))
end = time.time()

"""### Evaluate the model"""

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# # plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
#
# test_loss, test_acc = model.evaluate(x_dev_new, y_dev, verbose=2)

# testData = setupData("train1225_test.csv")
# predict_labels, predict_dataset = separateDataANDLabel(testData)
# predict_dataset_new = []
# for image in predict_dataset.T:
#     predict_dataset_new.append(np.reshape(image, (35, 35, 1)))
# predict_labels = np.reshape(predict_labels, (99, 1))
# predict_dataset_new = np.array(predict_dataset_new)
#
# # training=False is needed only if there are layers with different
# # behavior during training versus inference (e.g. Dropout).
# predictions = model(predict_dataset_new, training=False)
# class_names = ['Keegan', 'Stefan', 'Johan', 'David', 'StefanP']
# wrongGuesses = 0
# for i, logits in enumerate(predictions):
#     class_idx = tf.math.argmax(logits).numpy()
#     p = tf.nn.softmax(logits)[class_idx]
#     name = class_names[class_idx]
#     if class_idx != int(predict_labels[i]):
#         wrongGuesses += 1
#     print("Example {} prediction: {} ({:4.1f}%) : ({})".format(i, name, 100 * p, class_names[int(predict_labels[i])]))
# print("Number of wrong guesses: ", wrongGuesses)
print("Time to train: ", end - start, " seconds")
# np.savez("../../TensorFlowModelSaved-6400", model)
