import numpy as np

image = np.random.rand(33, 33, 64)
print(image)
blank_row = np.zeros((1, image.shape[1], 64))
blank_col = np.zeros((image.shape[0] + 1, 1, 64))

image = np.concatenate((image, blank_row), axis=0)
image = np.concatenate((image, blank_col), axis=1)


print(image)



