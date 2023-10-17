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


def addLetterToCanvas(canvas, letter, x, y, base="middle"):
    # canvas and letter are both only 2D numpy arrays
    if base == "middle":
        x = x - letter.shape[1] // 2
        y = y - letter.shape[0] // 2
    if base == "top":
        x = x - letter.shape[1] // 2
        y = y
    if base == "bottom":
        x = x - letter.shape[1] // 2
        y = y - letter.shape[0]

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
    """this takes the letters and puts them on a canvas"""
    max_riser = 0
    max_faller = 0
    for i, letter in enumerate(letters):
        if sentence[i] in ["b", "d", "f", "h", "k", "l"]:
            if letter.shape[0] > max_riser:
                max_riser = letter.shape[0]
        elif sentence[i] in ["g", "j", "p", "q", "y"]:
            if letter.shape[0] > max_faller:
                max_faller = letter.shape[0]
    upperLine = 0
    lowerLine = 0
    for i, letter in enumerate(letters):
        if sentence[i] in ["a", "c", "e", "i", "m", "n", "o", "r", "s", "u", "v", "w", "x", "z"]:
            if letter.shape[0] > lowerLine:
                lowerLine = letter.shape[0]
    heightOfCanvas = (upperLine+max_faller)-(lowerLine-max_riser)
    canvas = np.ones((heightOfCanvas, 1112), np.uint8) * 255  # Here is the height of the canvas(line) that needs to
    # be adjusted to add things like a signature which could be much taller than other letters in the handwriting

    centreLine = abs(lowerLine-max_riser) + lowerLine//2
    upperLine = centreLine - lowerLine//2
    lowerLine = centreLine + lowerLine//2

    for i, image in enumerate(letters):
        if sentence[i] in ["h", "l", "k", "b", "d", "f"]:
            addLetterToCanvas(canvas, image, x + image.shape[1] // 2, lowerLine, base="bottom")
        elif sentence[i] in ["g", "j", "p", "q", "y"]:
            addLetterToCanvas(canvas, image, x + image.shape[1] // 2, upperLine, base="top")
        elif sentence[i].isupper():
            addLetterToCanvas(canvas, image, x + image.shape[1] // 2, lowerLine, base="bottom")
        else:
            addLetterToCanvas(canvas, image, x + image.shape[1] // 2, centreLine, base="middle")
        x += image.shape[1] - 5
    return canvas



