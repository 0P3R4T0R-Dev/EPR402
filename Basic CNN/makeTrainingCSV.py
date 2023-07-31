from PIL import Image
import numpy as np
import math


num_pixels = 28 ** 2
num_samples = 10000

for i in range(num_samples):
    if i % 100 == 0:
        print(i, "/", num_samples)
    # 1. Read image
    img = Image.open('../Karen_TrainingData_0/KarenGrid-' + str(i) + '-.jpg')
    img = img.resize((int(math.sqrt(num_pixels)), int(math.sqrt(num_pixels))))  # consider resizing image to not have MANY MANY pixels
    img = img.convert("L")
    img = Image.eval(img, lambda px: 255 - px)
    # 2. Convert image to NumPy array
    arr = np.array(img)
    img.close()

    # 3. Convert 3D array to 2D list of lists
    lst = arr.flatten()
    # 4. Save list of lists to CSV
    with open('train' + str(num_pixels) + '.csv', 'a') as f:
        f.write("0,")
        for i, el in enumerate(lst):
            if i == 0:
                f.write(str(el))
            else:
                f.write(',' + str(el))
        f.write('\n')

for i in range(num_samples):
    if i % 100 == 0:
        print(i, "/", num_samples)
    # 1. Read image
    img = Image.open('../Jarrod_TrainingData_2/JarrodGrid-' + str(i) + '-.jpg')
    img = img.resize((int(math.sqrt(num_pixels)), int(math.sqrt(num_pixels))))  # consider resizing image to not have MANY MANY pixels
    img = img.convert("L")
    img = Image.eval(img, lambda px: 255 - px)
    # 2. Convert image to NumPy array
    arr = np.array(img)
    img.close()

    # 3. Convert 3D array to 2D list of lists
    lst = arr.flatten()
    # 4. Save list of lists to CSV
    with open('train' + str(num_pixels) + '.csv', 'a') as f:
        f.write("2,")
        for i, el in enumerate(lst):
            if i == 0:
                f.write(str(el))
            else:
                f.write(',' + str(el))
        f.write('\n')

for i in range(num_samples):
    if i % 100 == 0:
        print(i, "/", num_samples)
    # 1. Read image
    img = Image.open('../Juliet_TrainingData_1/JulietGrid-' + str(i) + '-.jpg')
    img = img.resize((int(math.sqrt(num_pixels)), int(math.sqrt(num_pixels))))  # consider resizing image to not have MANY MANY pixels
    img = img.convert("L")
    img = Image.eval(img, lambda px: 255 - px)
    # 2. Convert image to NumPy array
    arr = np.array(img)
    img.close()

    # 3. Convert 3D array to 2D list of lists
    lst = arr.flatten()
    # 4. Save list of lists to CSV
    with open('train' + str(num_pixels) + '.csv', 'a') as f:
        f.write("1,")
        for i, el in enumerate(lst):
            if i == 0:
                f.write(str(el))
            else:
                f.write(',' + str(el))
        f.write('\n')