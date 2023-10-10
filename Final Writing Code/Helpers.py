import numpy as np
import os
import random


def imageToBinary(image, threshold=128, flipped=False):
    upper = 255
    lower = 0
    if flipped:
        upper = 0
        lower = 255
    imageArr = np.array(image)
    copyImage = np.copy(imageArr)
    imageArr[copyImage > threshold] = upper
    imageArr[copyImage <= threshold] = lower
    return imageArr


def findMiddle(x, y, w, h):
    return x + w // 2, y + h // 2


def overlap(source, target):
    # unpack points
    x1, y1, w1, h1 = source
    x2, y2, w2, h2 = target

    # checks
    if x1 >= x2 + w2 or x2 >= x1 + w1:
        return False
    if y1 >= y2 + h2 or y2 >= y1 + h1:
        return False
    return True


def getAllOverlaps(boxes, bounds, index):
    overlaps = []
    for a in range(len(boxes)):
        if a != index:
            if overlap(bounds, boxes[a]):
                overlaps.append(a)
    return overlaps


def doesContain(source, target):
    # unpack points
    x1, y1, w1, h1 = source
    x2, y2, w2, h2 = target

    # checks
    if x1 <= x2 and x1 + w1 >= x2 + w2:
        if y1 <= y2 and y1 + h1 >= y2 + h2:
            return True
    return False


def getAllContains(boxes, bounds, index):
    contains = []
    for a in range(len(boxes)):
        if a != index:
            if doesContain(bounds, boxes[a]):
                contains.append(a)
    return contains


def addLetterToCanvas(canvas, letter, x, y):
    # canvas and letter are both only 2D numpy arrays

    x = x - letter.shape[1] // 2
    y = y - letter.shape[0] // 2

    for row in range(letter.shape[0]):
        for col in range(letter.shape[1]):
            canvas[row + y][col + x] = letter[row][col]
    return canvas


def getImageOfLetter(letter, folderName):
    import PIL.Image as Image
    length = len(os.listdir(folderName + "/" + letter)) - 1
    image = Image.open(folderName + "/" + letter + "/" + str(random.randint(0, length)) + ".jpg").convert("L")
    image = imageToBinary(image, 100)
    image = np.array(image)
    return image


def getName(folderPath):
    file_names = os.listdir(folderPath)
    if len(file_names) == 0:
        return "0"
    return str(len(file_names))


def addLettersToCanvas(letters, sentence, x=20):
    # this takes the letters and puts them on a canvas
    canvas = np.ones((120, 1112), np.uint8) * 255

    for i, image in enumerate(letters):  # down shift is lower number
        if sentence[i] in ["h", "l", "k", "b", "d", "f"]:
            addLetterToCanvas(canvas, image, x + image.shape[1] // 2, 40)
        elif sentence[i] in ["g", "j", "p", "q", "y"]:
            addLetterToCanvas(canvas, image, x + image.shape[1] // 2, 60)
        elif sentence[i].isupper():
            addLetterToCanvas(canvas, image, x + image.shape[1] // 2, 40)
        else:
            addLetterToCanvas(canvas, image, x + image.shape[1] // 2, 50)
        # Helpers.addLetterToCanvas(canvas, image, x + image.shape[1] // 2, 100)
        x += image.shape[1] - 5
    return canvas



