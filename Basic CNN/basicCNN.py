import pandas as pd
from matplotlib import pyplot as plt
import time
from NN_components import *

print("Loading data...")
data = pd.read_csv('train.csv')
data = np.array(data)
m, n = data.shape
print("Data shape: ", data.shape)
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
print("Data loaded.")

# for j in range(5):
#     toPrint = []
#     for i in range(784):
#         toPrint.append(X_train[i][j])
#     print(toPrint)
#     toPrint = np.array(toPrint)
#     current_image = toPrint.reshape((28, 28)) * 255
#     print(Y_train[j])
#     plt.gray()
#     plt.title("Label: " + str(Y_train[j]))
#     plt.imshow(current_image, interpolation='nearest')
#     plt.show()


epochs = 200
learning_rate = 0.2
start_time = time.time()

conv1 = Layer_MyConvolution(784, 6, weight_lambda_l2=5e-4, bias_lambda_l2=5e-4)
activation1 = Activation_ReLU()
dropout1 = Layer_Dropout(0.25)
conv2 = Layer_MyConvolution(conv1.num_neurons, 6, weight_lambda_l2=5e-4, bias_lambda_l2=5e-4)
activation2 = Activation_ReLU()
dropout2 = Layer_Dropout(0.25)
dense2 = Layer_Dense(conv2.num_neurons, 10)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
optimizer = Optimizer_Adam(learning_rate=0.001, decay=50e-5)

accuracy = []
acc = 0
counter = 0
print("Training...")
for i in range(epochs):
    conv1.forward(X_train.T)
    activation1.forward(conv1.output)
    dropout1.forward(activation1.output)
    conv2.forward(dropout1.output)
    activation2.forward(conv2.output)
    dropout2.forward(activation2.output)
    dense2.forward(dropout2.output)
    data_loss = loss_activation.forward(dense2.output, Y_train)

    regularization_loss = \
        loss_activation.loss.regularization_loss(conv1) + \
        loss_activation.loss.regularization_loss(conv2) + \
        loss_activation.loss.regularization_loss(dense2)
    loss = data_loss + regularization_loss

    loss_activation.backward(loss_activation.output, Y_train)
    dense2.backward(loss_activation.d_inputs)
    dropout2.backward(dense2.d_inputs)
    activation2.backward(dropout2.d_inputs)
    conv2.backward(activation2.d_inputs)
    dropout1.backward(conv2.d_inputs)
    activation1.backward(dropout1.d_inputs)
    conv1.backward(activation1.d_inputs)

    optimizer.pre_update_params()
    optimizer.update_params(conv1)
    optimizer.update_params(conv2)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    conv1.forward(X_dev.T)
    activation1.forward(conv1.output)
    # dropout1.forward(activation1.output)
    conv2.forward(activation1.output)
    activation2.forward(conv2.output)
    # dropout2.forward(activation2.output)
    dense2.forward(activation2.output)
    loss_activation.forward(dense2.output, Y_dev)
    prediction = get_predictions(loss_activation.output)
    acc = get_accuracy(prediction, Y_dev)
    accuracy.append(acc)
    if i % 10 == 0:
        print("Iteration: ", i)
        print("Loss: ", loss)
        print(prediction[:20])
        print(Y_dev[:20])
        print("Accuracy: ", acc)
    np.savez("my_data", conv1, activation1, dropout1, conv2, activation2, dropout2, dense2, loss_activation, optimizer)
end_time = time.time()
print("Elapsed Time: ", end_time - start_time)

x = range(1, epochs + 1)
plt.plot(x, accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Iteration')
plt.show()

