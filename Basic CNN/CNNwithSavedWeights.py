import pandas as pd
from matplotlib import pyplot as plt
import time
from NN_components import *

model = np.load("my_data.npz", allow_pickle=True)

print("Loading data...")
data = pd.read_csv('test.csv', header=None)
data = np.array(data)
m, n = data.shape
print("Data shape: ", data.shape)

data_dev = data.T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.
print("Data loaded.")


start_time = time.time()
print("Loading model...")

conv1 = model["arr_0"].item()
activation1 = model["arr_1"].item()
dropout1 = model["arr_2"].item()
conv2 = model["arr_3"].item()
activation2 = model["arr_4"].item()
dropout2 = model["arr_5"].item()
dense2 = model["arr_6"].item()
loss_activation = model["arr_7"].item()
optimizer = model["arr_8"].item()
print("Model loaded.")

accuracy = []
counter = 0
print("Running...")

for i in range(20):
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

    print("Accuracy: ", acc)

end_time = time.time()
print("Elapsed Time: ", end_time - start_time)
