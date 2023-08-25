from PIL import Image
import numpy as np
import math


num_pixels = 35 ** 2

# person = ["Stefan", "Johan", "David"]
# ID = ["0", "1", "2"]
person = ["Stefan_Sentences", "Johan_Sentences", "David_Sentences", "StefanP_Sentences", "Brian_Sentences"]
# person = ["Stefan_BigParagraph", "Johan_BigParagraph", "David_BigParagraph", "StefanP_BigParagraph", "Brian_BigParagraph"]
# person = ["David_Sentences"]
ID = ["0", "1", "2", "3", "4"]
# ID = ["2"]
# person = ["StefanTest grids"]
# ID = ["0"]
folder_path = "C:/EPR402 REPO/DATA/"

for P in range(len(person)):
    flag = True
    m = 0
    print("Person:", person[P])
    print("ID:", ID[P])
    while flag:
        if m % 500 == 0:
            print(m)
        try:
            img = Image.open(folder_path + person[P] + '/' + str(m) + '.jpg')
        except IOError:
            flag = False
            break
        m += 1
        img = img.resize((int(math.sqrt(num_pixels)), int(math.sqrt(num_pixels))))
        img = img.convert("L")
        img = Image.eval(img, lambda px: 255 - px)
        # 2. Convert image to NumPy array
        arr = np.array(img)
        img.close()

        # 3. Convert 3D array to 2D list of lists
        lst = arr.flatten()
        # 4. Save list of lists to CSV
        with open('C:/EPR402 REPO/csv/Sentences_5thPerson ' + str(num_pixels) + '.csv', 'a') as f:
            f.write(ID[P] + ",")
            for i, el in enumerate(lst):
                if i == 0:
                    f.write(str(el))
                else:
                    f.write(',' + str(el))
            f.write('\n')

