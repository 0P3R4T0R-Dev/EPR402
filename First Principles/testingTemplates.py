import Helpers
import PIL.Image as Image
import numpy as np


image = Image.open("TEMP.jpg").convert("F")
# image.show()

imageArr = Helpers.imageToBinary(np.array(image), threshold=90)

imageArr = np.array(imageArr, dtype=np.uint8)
imageToShow = Image.fromarray(imageArr)
imageToShow.save("HiThereEveryoneBinary.jpg")
# imageToShow.show()






