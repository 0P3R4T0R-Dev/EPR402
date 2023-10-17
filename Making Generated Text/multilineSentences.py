# the page is 1112 pixels wide(second) and 1247 pixels tall(first)
# total of 9 lines of writing given that each line is 140 pixels tall

import numpy as np
import Helpers
import PIL.Image as Image
import random

maxCharsPerLine = 35

sentence = "something is afoot and its not at the end of my leg"
# sentence = "something is aoo"
hand_writer = "dewaldCapital"

words = sentence.split(" ")
lines = []
currentLine = ""
for i, word in enumerate(words):
    if len(currentLine) + len(word) < maxCharsPerLine:
        if i == len(words) - 1:
            currentLine += word
            lines.append(currentLine)
            break
        currentLine += word + " "
    else:
        lines.append(currentLine)
        currentLine = word + " "
        if i == len(words) - 1:
            lines.append(currentLine)
            break

totalCanvas = []
for line in lines:
    sentenceImages = []

    for letter in line:
        if letter == " ":
            sentenceImages.append(np.ones((random.randint(10, 20), 35), np.uint8) * 255)
        else:
            if letter.isupper():
                letter = letter + "_"
            sentenceImages.append(Helpers.getImageOfLetter(letter, "../../FONTS/" + hand_writer))

    canvas = Helpers.addLettersToCanvas(sentenceImages, line)
    if len(totalCanvas) == 0:
        totalCanvas = canvas
    else:
        totalCanvas = np.vstack((totalCanvas, canvas))

if totalCanvas.shape[0] > 1247:
    print()
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Too many lines please use less words in the sentence")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    exit(1)
totalCanvas = Image.fromarray(totalCanvas)
totalCanvas.show()
# totalCanvas.save(sentence + ".jpg")
