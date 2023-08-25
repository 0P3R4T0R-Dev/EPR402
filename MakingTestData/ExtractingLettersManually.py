from collections import namedtuple
from PIL import Image, ImageFilter
import numpy as np
import math

person = "JohanTest"
ID = "2"
num_samples = 100


def constructGrid(characters, debug=False):
    basicGrid = np.array([])
    previousRow = 0
    counter = 0
    tempGrid = np.array([])
    for i, character in enumerate(characters):
        # print(grouped_characters[character][random.randint(0, 4)].shape)
        row = math.floor((i / 5))
        if row == previousRow:
            if counter == 0:
                tempGrid = characters[character][0]
            else:
                tempGrid = np.hstack((tempGrid, characters[character][0]))
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
                tempGrid = characters[character][0]
            else:
                tempGrid = np.hstack((tempGrid, characters[character][0]))
            counter += 1
            previousRow = row
            if i == 25:
                tempGrid = characters[character][0]
                tempGrid = np.hstack((tempGrid, np.zeros((25, 100, 3), dtype=np.uint8)))
                basicGrid = np.vstack((basicGrid, tempGrid))

    if debug:
        image_array = np.array(basicGrid)
        image_to_show = Image.fromarray(image_array)
        # b, g, r = image_to_show.split()
        # image_to_show = Image.merge("RGB", (r, g, b))
        image_to_show.show()
    return basicGrid


grouped_characters = {name: [None for _ in range(6)] for name in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
                                                                  "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
                                                                  "w", "x", "y", "z"]}

tempImage = Image.open("JohanLetters/a.jpg")
grouped_characters["a"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/b.jpg")
grouped_characters["b"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/c.jpg")
grouped_characters["c"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/d.jpg")
grouped_characters["d"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/e.jpg")
grouped_characters["e"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/f.jpg")
grouped_characters["f"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/g.jpg")
grouped_characters["g"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/h.jpg")
grouped_characters["h"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/i.jpg")
grouped_characters["i"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/j.jpg")
grouped_characters["j"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/k.jpg")
grouped_characters["k"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/l.jpg")
grouped_characters["l"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/m.jpg")
grouped_characters["m"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/n.jpg")
grouped_characters["n"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/o.jpg")
grouped_characters["o"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/p.jpg")
grouped_characters["p"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/q.jpg")
grouped_characters["q"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/r.jpg")
grouped_characters["r"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/s.jpg")
grouped_characters["s"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/t.jpg")
grouped_characters["t"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/u.jpg")
grouped_characters["u"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/v.jpg")
grouped_characters["v"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/w.jpg")
grouped_characters["w"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/x.jpg")
grouped_characters["x"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/y.jpg")
grouped_characters["y"][0] = tempImage.resize((25, 25))
tempImage = Image.open("JohanLetters/z.jpg")
grouped_characters["z"][0] = tempImage.resize((25, 25))


grid = constructGrid(grouped_characters, debug=True)

# save grid using pillow
image_array = np.array(grid)
image_to_show = Image.fromarray(image_array)
image_to_show.save("JohanTestGridGrid-0-.jpg")
