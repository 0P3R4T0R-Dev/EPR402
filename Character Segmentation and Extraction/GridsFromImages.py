import numpy as np
from PIL import Image, ImageFilter
from Grid import *
import os

num_samples = 5000
filename = "StefanP_BigParagraph"

folder_path = "C:/EPR402 REPO/DATA/" + filename

flag = True
counter = 0
images = []

while flag:
    myimage = None
    try:
        myimage = Image.open(folder_path + "/" + str(counter) + ".jpg").convert("L").resize((25, 25))
    except IOError:
        break
    counter += 1
    myimage = np.array(myimage)
    images.append(myimage)

for i in range(num_samples):
    array = constructSmallerGridRandomly(images)
    imageToSave = Image.fromarray(array)
    imageToSave = imageToSave.filter(ImageFilter.SHARPEN)
    imageToSave.save(folder_path + "/" + str(i) + ".jpg")
