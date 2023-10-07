import numpy as np
import Helpers
import PIL.Image as Image


sentence = "this is a generated fffkllkhh"  # length max 30
hand_writer = "myburgh"
sentenceImages = []

for letter in sentence:
    if letter == " ":
        sentenceImages.append(np.ones((15, 45), np.uint8) * 255)
    else:
        sentenceImages.append(Helpers.getImageOfLetter(letter, "../../FONTS/" + hand_writer))

canvas = np.ones((120, 1112), np.uint8) * 255

# the page is 1112 pixels wide(second) and 1247 pixels tall(first)
# total of 9 lines of writing given that each line is 140 pixels tall

x = 20
for i, image in enumerate(sentenceImages):
    if sentence[i] in ["h", "l", "k", "b", "d", "f"]:
        Helpers.addLetterToCanvas(canvas, image, x + image.shape[1]//2, 40)
    elif sentence[i] in ["g", "j", "p", "q", "y"]:
        Helpers.addLetterToCanvas(canvas, image, x + image.shape[1]//2, 60)
    else:
        Helpers.addLetterToCanvas(canvas, image, x + image.shape[1]//2, 50)
    # Helpers.addLetterToCanvas(canvas, image, x + image.shape[1] // 2, 100)
    x += image.shape[1]
canvas = Image.fromarray(canvas)
canvas.show()
# canvas.save("this is a generated sentence.jpg")











