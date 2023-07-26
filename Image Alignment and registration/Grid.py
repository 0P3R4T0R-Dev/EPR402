import numpy as np
import random
from PIL import Image
import math


def constructGrid(characters, pos=0, debug=False):
    if pos == -1:
        pos = random.randint(0, 4)
    basicGrid = np.array([])
    previousRow = 0
    counter = 0
    tempGrid = np.array([])
    for i, character in enumerate(characters):
        # print(grouped_characters[character][random.randint(0, 4)].shape)
        row = math.floor((i / 5))
        if row == previousRow:
            if counter == 0:
                tempGrid = characters[character][pos]
            else:
                tempGrid = np.hstack((tempGrid, characters[character][pos]))
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
                tempGrid = characters[character][pos]
            else:
                tempGrid = np.hstack((tempGrid, characters[character][pos]))
            counter += 1
            previousRow = row
            if i == 25:
                tempGrid = characters[character][pos]
                tempGrid = np.hstack((tempGrid, np.zeros((25, 100, 3), dtype=np.uint8)))
                basicGrid = np.vstack((basicGrid, tempGrid))

    if debug:
        image_array = np.array(basicGrid)
        image_to_show = Image.fromarray(image_array)
        image_to_show.show()
    return basicGrid


def constructGridRandomly(characters, noForm=True, debug=False):
    basicGrid = np.array([])
    previousRow = 0
    counter = 0
    tempGrid = np.array([])
    for i, character in enumerate(characters):
        # print(grouped_characters[character][random.randint(0, 4)].shape)
        row = math.floor((i / 5))
        if row == previousRow:
            if counter == 0:
                tempGrid = characters[character][random.randint(noForm, 4)]
            else:
                tempGrid = np.hstack((tempGrid, characters[character][random.randint(noForm, 4)]))
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
                tempGrid = characters[character][random.randint(noForm, 4)]
            else:
                tempGrid = np.hstack((tempGrid, characters[character][random.randint(noForm, 4)]))
            counter += 1
            previousRow = row
            if i == 25:
                tempGrid = characters[character][random.randint(noForm, 4)]
                tempGrid = np.hstack((tempGrid, np.zeros((25, 100, 3), dtype=np.uint8)))
                basicGrid = np.vstack((basicGrid, tempGrid))

    if debug:
        image_array = np.array(basicGrid)
        image_to_show = Image.fromarray(image_array)
        b, g, r = image_to_show.split()
        image_to_show = Image.merge("RGB", (r, g, b))
        image_to_show.show()
    return basicGrid
