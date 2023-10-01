import numpy as np
import Helpers
import PIL.Image as Image


sentence = "i ate a fish today"  # length max 30
hand_writer = "matthew"
sentenceImages = []

for letter in sentence:
    if letter == " ":
        sentenceImages.append(np.ones((35, 45), np.uint8) * 255)
    else:
        sentenceImages.append(Helpers.getImageOfLetter(letter, "../../FONTS/" + hand_writer))

canvas = np.ones((200, 2500), np.uint8) * 255


x = 20
for i, image in enumerate(sentenceImages):
    if sentence[i] in ["h", "l", "k", "b", "d", "f"]:
        Helpers.addLetterToCanvas(canvas, image, x + image.shape[1]//2, 90)
    elif sentence[i] in ["g", "j", "p", "q", "y"]:
        Helpers.addLetterToCanvas(canvas, image, x + image.shape[1]//2, 110)
    else:
        Helpers.addLetterToCanvas(canvas, image, x + image.shape[1]//2, 100)
    # Helpers.addLetterToCanvas(canvas, image, x + image.shape[1] // 2, 100)
    x += image.shape[1]
canvas = Image.fromarray(canvas)
canvas.show()
canvas.save("i ate a fish today.jpg")











