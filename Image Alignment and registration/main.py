# import the necessary packages
import random
import math
from ImageAlignment import align_images
from collections import namedtuple
import cv2
from PIL import Image
import numpy as np
from Grid import *

person = "Karen"
ID = "0"
num_samples = 10000

# create a named tuple which we can use to create locations of the
# input document which we wish to OCR
OCRLocation = namedtuple("OCRLocation", ["id", "bbox"])  # x, y, w, h
y_values = [608, 706, 804, 902, 999, 1097, 1195, 1293, 1391, 1489, 1587, 1685, 1783, 1881, 1979, 2077, 2175, 2273, 2371,
            2469, 2567, 2665, 2763, 2861, 2959, 3057]
x_values = [790, 909, 1028, 1147, 1266, 1385]
OCR_LOCATIONS = []
for y_num, char in enumerate(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
                              "s", "t", "u", "v", "w", "x", "y", "z"]):
    for num in range(6):
        OCR_LOCATIONS.append(OCRLocation(char + "_" + str(num), (x_values[num], y_values[y_num], 114, 93)))

image = cv2.imread("Forms/Form"+person+".jpg")
template = cv2.imread("User Input Form V3 template.jpg")

aligned = align_images(image, template)

grouped_characters = {name: [None for _ in range(5)] for name in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
                                                                  "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
                                                                  "w", "x", "y", "z"]}
# loop over the locations of the document we are going to OCR
for loc in OCR_LOCATIONS:
    (x, y, w, h) = loc.bbox
    roi = aligned[y:y + h, x:x + w]
    roi = Image.fromarray(roi)
    roi = roi.resize((25, 25))
    roi = np.array(roi)

    for i, element in enumerate(grouped_characters[loc.id[0]]):
        if element is None:
            grouped_characters[loc.id[0]][i] = roi
            break

arrayToSave = np.array(constructGridRandomly(grouped_characters))
for i in range(5):
    array = constructGridRandomly(grouped_characters, debug=True)
    arrayToSave = np.hstack((arrayToSave, array))

imageToSave = Image.fromarray(arrayToSave)
imageToSave.show()
imageToSave.save("output samples 5 V4.jpg")
# for i in range(num_samples):
#     array = constructGridRandomly(grouped_characters)
#     imageToSave = Image.fromarray(array)
#     imageToSave.save("../"+person+"_TrainingData_"+ID+"/"+person+"Grid-" + str(i) + "-.jpg")
