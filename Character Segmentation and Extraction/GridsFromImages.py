import numpy as np
from PIL import Image, ImageFilter
from Grid import *

num_samples = 5000
filename = "Johan_Sentences"

flag = True
counter = 0
images = []

while flag:
    myimage = None
    try:
        myimage = Image.open(filename + "/" + str(counter) + ".jpg").convert("L").resize((25, 25))
    except IOError:
        break
    counter += 1
    myimage = np.array(myimage)
    images.append(myimage)

for i in range(num_samples):
    array = constructSmallerGridRandomly(images)
    imageToSave = Image.fromarray(array)
    imageToSave = imageToSave.filter(ImageFilter.SHARPEN)
    imageToSave.save(filename + "/" + str(i) + ".jpg")
