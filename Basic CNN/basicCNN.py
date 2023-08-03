import pandas as pd
from matplotlib import pyplot as plt
import time
from NN_components import *
from setupData import *

data = setupData('train784.csv')
training_sets, testing_sets = kFoldCrossValidation(data, 5)

totalTime = []
epochs = 100

for p in range(len(training_sets)):
    print("Fold: ", p, " of ", len(training_sets))
    Y_train, X_train = separateDataANDLabel(training_sets[p])
    Y_dev, X_dev = separateDataANDLabel(testing_sets[p])
    np.clip(Y_train, None, 2)
    np.clip(Y_dev, None, 2)
    numberOfOutputOptions = len(np.unique(Y_train))
    start_time = time.time()

    conv1 = Layer_MyConvolution(784, 3, weight_lambda_l2=5e-4, bias_lambda_l2=5e-4)
    activation1 = Activation_ReLU()
    dropout1 = Layer_Dropout(0.25)
    # print("conv1.num_neurons: ", conv1.num_neurons)
    conv2 = Layer_MyConvolution(conv1.num_neurons, 3, weight_lambda_l2=5e-4, bias_lambda_l2=5e-4)
    activation2 = Activation_ReLU()
    dropout2 = Layer_Dropout(0.25)
    # print("conv2.num_neurons: ", conv2.num_neurons)
    dense2 = Layer_Dense(conv2.num_neurons, numberOfOutputOptions)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
    optimizer = Optimizer_Adam(learning_rate=0.001, decay=50e-5)

    accuracy = []
    overallLoss = []
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
        loss_activation.forward(dense2.output, Y_train)

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
        data_loss = loss_activation.forward(dense2.output, Y_dev)
        regularization_loss = \
            loss_activation.loss.regularization_loss(conv1) + \
            loss_activation.loss.regularization_loss(conv2) + \
            loss_activation.loss.regularization_loss(dense2)
        loss = data_loss + regularization_loss
        prediction = get_predictions(loss_activation.output)
        acc = get_accuracy(prediction, Y_dev)
        accuracy.append(acc)
        overallLoss.append(data_loss)
        if i % 10 == 0:
            print("Iteration: ", i, " : ", p)
            print("Loss: ", loss)
            print("data_loss: ", data_loss)
            print("regularization_loss: ", regularization_loss)
            # print(prediction[:20])
            # print(Y_dev[:20])
            print("Accuracy: ", acc)
        np.savez("my_data", conv1, activation1, dropout1, conv2, activation2, dropout2, dense2, loss_activation, optimizer)
    end_time = time.time()
    totalTime.append(end_time - start_time)
    print("Elapsed Time: ", end_time - start_time)

    x = range(1, epochs + 1)
    plt.plot(x, overallLoss)
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
print()
print("Average Time: ", sum(totalTime) / len(totalTime))
print("Average Time: ", sum(totalTime) / len(totalTime) / epochs, " per iteration")
print("Total Time: ", sum(totalTime), totalTime)
plt.show()

