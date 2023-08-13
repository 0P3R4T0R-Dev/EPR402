import numpy as np
from NN_components import *

data = np.load("my_data.npz", allow_pickle=True)
for name in data:
    print("name:", name, "| array:", data[name])
    hey = data[name].item()
    print(hey.weights)
