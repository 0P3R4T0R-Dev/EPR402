from PIL import Image
import numpy as np
import math


num_pixels = 80 ** 2
num_samples = 5000

person = ["Keegan", "Stefan", "Johan", "David", "StefanP"]
ID = ["0", "1", "2", "3", "4"]

for P in range(len(person)):
    print("Person:", person[P])
    print("ID:", ID[P])
    for i in range(num_samples):
        if i % 100 == 0:
            print(i, "/", num_samples)
        # 1. Read image
        img = Image.open('../../' + person[P] + '_TrainingData_' + ID[P] + '/' + person[P] + 'Grid-' + str(i) + '-.jpg')
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
            f.write(ID[P] + ",")
            for i, el in enumerate(lst):
                if i == 0:
                    f.write(str(el))
                else:
                    f.write(',' + str(el))
            f.write('\n')

