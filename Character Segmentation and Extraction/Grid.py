import numpy as np
import random
from PIL import Image
import math


def constructGridRandomly(images, debug=False):
    basicGrid = np.array([])
    previousRow = 0
    counter = 0
    tempGrid = np.array([])
    for i in range(len(images)):
        row = math.floor((i / 5))
        if row == previousRow:
            if counter == 0:
                tempGrid = images[random.randint(0, len(images) - 1)]
            else:
                tempGrid = np.hstack((tempGrid, images[random.randint(0, len(images) - 1)]))
            counter += 1
            previousRow = row
        else:
            previousRow = row
            counter = 0
            if basicGrid.size == 0:
                basicGrid = tempGrid
            else:
                basicGrid = np.vstack((basicGrid, tempGrid))
            tempGrid = np.array([])
            if counter == 0:
                tempGrid = images[random.randint(0, len(images) - 1)]
            else:
                tempGrid = np.hstack((tempGrid, images[random.randint(0, len(images) - 1)]))
            counter += 1
            previousRow = row
            if i == 25:
                break

    if debug:
        image_array = np.array(basicGrid)
        image_to_show = Image.fromarray(image_array)
        # b, g, r = image_to_show.split()
        # image_to_show = Image.merge("RGB", (r, g, b))
        image_to_show.show()
    return basicGrid


def constructSmallerGridRandomly(images, debug=False):
    basicGrid = np.array([])
    previousRow = 0
    counter = 0
    tempGrid = np.array([])
    for i in range(len(images)):
        row = math.floor((i / 2))
        if row == previousRow:
            if counter == 0:
                tempGrid = images[random.randint(0, len(images) - 1)]
            else:
                tempGrid = np.hstack((tempGrid, images[random.randint(0, len(images) - 1)]))
            counter += 1
            previousRow = row
        else:
            previousRow = row
            counter = 0
            if basicGrid.size == 0:
                basicGrid = tempGrid
            else:
                basicGrid = np.vstack((basicGrid, tempGrid))
            tempGrid = np.array([])
            if counter == 0:
                tempGrid = images[random.randint(0, len(images) - 1)]
            else:
                tempGrid = np.hstack((tempGrid, images[random.randint(0, len(images) - 1)]))
            counter += 1
            previousRow = row
            if i == 4:
                break

    if debug:
        image_array = np.array(basicGrid)
        image_to_show = Image.fromarray(image_array)
        # b, g, r = image_to_show.split()
        # image_to_show = Image.merge("RGB", (r, g, b))
        image_to_show.show()
    return basicGrid

