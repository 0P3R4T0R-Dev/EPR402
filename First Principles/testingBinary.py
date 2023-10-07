import Helpers
import PIL.Image as Image
import numpy as np


image = Image.open("someWeird.jpg").convert("F")
# image.show()

imageArr = Helpers.imageToBinary(np.array(image), threshold=90, flipped=True)

imageArr = np.array(imageArr, dtype=np.uint8)
imageToShow = Image.fromarray(imageArr)
imageToShow.save("someWeird.jpg")
imageToShow.show()






