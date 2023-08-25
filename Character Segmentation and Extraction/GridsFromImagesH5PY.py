import numpy as np
from PIL import Image, ImageFilter
from Grid import *
import os
import h5py

num_samples = 5000
filename = "Stefan_BigParagraph"

folder_path = "C:/EPR402 REPO/DATA/"

flag = True
counter = 0
images = []

while flag:
    myimage = None
    try:
        myimage = Image.open(folder_path + filename + "/" + str(counter) + ".jpg").convert("L").resize((25, 25))
    except IOError:
        break
    counter += 1
    myimage = np.array(myimage)
    images.append(myimage)

arrToSaveWithH5PY = np.zeros((num_samples, 35, 35))
for i in range(num_samples):
    array = constructSmallerGridRandomly(images)
    imageToSave = Image.fromarray(array)
    imageToSave = imageToSave.resize((35, 35))
    imageToSave = imageToSave.filter(ImageFilter.SHARPEN)
    imageToSave = np.array(imageToSave)
    arrToSaveWithH5PY[i] = imageToSave

with h5py.File(folder_path + "/" + filename + ".h5", 'w') as hf:
    hf.create_dataset("dataset", data=arrToSaveWithH5PY)
