# import the necessary packages
import random
import math
from ImageAlignment import align_images
from collections import namedtuple
import cv2
from PIL import Image
import numpy as np
from Grid import *

# create a named tuple which we can use to create locations of the
# input document which we wish to OCR
OCRLocation = namedtuple("OCRLocation", ["id", "bbox"])  # x, y, w, h
y_values = [577, 675, 773, 871, 969, 1067, 1165, 1263, 1361, 1459, 1557, 1655, 1753, 1851, 1949, 2047, 2145, 2243, 2341,
            2439, 2537, 2635, 2733, 2831, 2929, 3027]
x_values = [790, 909, 1028, 1147, 1266, 1385]
OCR_LOCATIONS = []
for y_num, char in enumerate(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
                              "s", "t", "u", "v", "w", "x", "y", "z"]):
    for num in range(6):
        OCR_LOCATIONS.append(OCRLocation(char + "_" + str(num), (x_values[num], y_values[y_num], 114, 93)))

image = cv2.imread("test upscaled with EDSR.jpg")
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
imageToSave.save("output samples 5 with EDSR upscaled test.jpg")
