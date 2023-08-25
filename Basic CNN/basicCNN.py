import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
from NN_components import *
from setupData import *
from graphStuff import plot_graphs

data = setupData('train6400.csv')
training_sets, testing_sets = kFoldCrossValidation(data, 5)

totalTime = []
epochs = 300
batchSize = 100
totalAccuracy = []
totalLoss = []
totalDataLoss = []
totalRegLoss = []
totalTrainAccuracy = []
totalTrainLoss = []
totalTrainDataLoss = []
totalTrainRegLoss = []

for p in range(len(training_sets)):
    print("Fold: ", p, " of ", len(training_sets))
    Y_train, X_train = separateDataANDLabel(training_sets[p])
    print("Y_train.shape: ", np.array(Y_train).shape)
    print("X_train.shape: ", np.array(X_train).shape)
    Y_train = split_into_groups(Y_train, batchSize)
    print("Y_train.shape: ", np.array(Y_train).shape)
    X_train = split_2d_into_groups(X_train.T, batchSize)
    print("X_train.shape: ", np.array(X_train).shape)
    Y_dev, X_dev = separateDataANDLabel(testing_sets[p])
    # np.clip(Y_train, None, 2)
    # np.clip(Y_dev, None, 2)
    numberOfOutputOptions = len(np.unique(Y_train))
    start_time = time.time()

    conv1 = Layer_MyConvolution(6400, 3, weight_lambda_l2=5e-4, bias_lambda_l2=5e-4)
    activation1 = Activation_ReLU()
    dropout1 = Layer_Dropout(0.2)
    # print("conv1.num_neurons: ", conv1.num_neurons)
    conv2 = Layer_MyConvolution(conv1.num_neurons, 3, weight_lambda_l2=5e-4, bias_lambda_l2=5e-4)
    activation2 = Activation_ReLU()
    dropout2 = Layer_Dropout(0.2)
    # print("conv2.num_neurons: ", conv2.num_neurons)
    dense2 = Layer_Dense(conv2.num_neurons, numberOfOutputOptions)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
    optimizer = Optimizer_Adam(learning_rate=0.001, decay=50e-5)

    accuracy = []
    overallLoss = []
    dataLoss = []
    regLoss = []
    trainingLoss = []
    trainingAcc = []
    trainingDataLoss = []
    trainingRegLoss = []

    acc = 0
    counter = 0
    print("Training...")
    for i in range(epochs):
        for batch in range(len(X_train)):
            perBatchTimeStart = time.time()
            conv1.forward(X_train[batch])
            activation1.forward(conv1.output)
            dropout1.forward(activation1.output)
            conv2.forward(dropout1.output)
            activation2.forward(conv2.output)
            dropout2.forward(activation2.output)
            dense2.forward(dropout2.output)
            trainingDataLossVar = loss_activation.forward(dense2.output, Y_train[batch])
            trainingRegularization_loss = \
                loss_activation.loss.regularization_loss(conv1) + \
                loss_activation.loss.regularization_loss(conv2) + \
                loss_activation.loss.regularization_loss(dense2)
            trainingLossVar = trainingDataLossVar + trainingRegularization_loss
            prediction = get_predictions(loss_activation.output)
            trainingAccVar = get_accuracy(prediction, Y_train[batch])
            trainingLoss.append(trainingLossVar)
            trainingAcc.append(trainingAccVar)
            trainingDataLoss.append(trainingDataLossVar)
            trainingRegLoss.append(trainingRegularization_loss)

            loss_activation.backward(loss_activation.output, Y_train[batch])
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
            # optimizer2.update_params(conv1)
            # optimizer2.update_params(conv2)
            # optimizer2.update_params(dense2)

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
            overallLoss.append(loss)
            dataLoss.append(data_loss)
            regLoss.append(regularization_loss)
            perBatchTimeEnd = time.time()
            print("#################################################################")
            print("Iteration: ", i, " : ", p)
            print("Batch: ", batch, " of ", len(X_train))
            print("Time taken for this batch: ", perBatchTimeEnd - perBatchTimeStart)
            print("Loss: ", loss)
            print("data_loss: ", data_loss)
            print("regularization_loss: ", regularization_loss)
            print("Accuracy: ", acc)
            if data_loss < 0.35:
                break
        if patience_five(np.array(dataLoss)):
            break

    totalAccuracy.append(accuracy)
    totalLoss.append(overallLoss)
    totalDataLoss.append(dataLoss)
    totalRegLoss.append(regLoss)
    totalTrainLoss.append(trainingLoss)
    totalTrainAccuracy.append(trainingAcc)
    totalTrainDataLoss.append(trainingDataLoss)
    totalTrainRegLoss.append(trainingRegLoss)
    np.savez("../../" + str(p) + " this_was_done_with_big_images", conv1, activation1, dropout1, conv2, activation2,
             dropout2, dense2, loss_activation, optimizer)
    end_time = time.time()
    totalTime.append(end_time - start_time)
    print("Elapsed Time: ", end_time - start_time)

    # plt.plot(x, overallLoss)
    # plt.ylabel('Loss')
    # plt.xlabel('Iteration')
print()
print("Average Time: ", sum(totalTime) / len(totalTime))
print("Average Time: ", sum(totalTime) / len(totalTime) / epochs, " per iteration")
print("Total Time: ", sum(totalTime), totalTime)
Y = [totalAccuracy, totalLoss, totalDataLoss, totalRegLoss]
print("totalAccuracy: ", np.array(totalAccuracy).shape)
print("totalLoss: ", np.array(totalLoss).shape)
print("length of totalAccuracy: ", len(totalAccuracy))
print("length of totalAccuracy[0]: ", len(totalAccuracy[0]))
print("shape of totalAccuracy[0]: ", np.array(totalAccuracy[0]).shape)
x = np.arange(len(totalAccuracy[0]))
plot_graphs(x, Y)
Y = [totalTrainAccuracy, totalTrainLoss, totalTrainDataLoss, totalTrainRegLoss]
plot_graphs(x, Y)
