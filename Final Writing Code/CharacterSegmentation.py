import numpy as np
import cv2
import PIL.Image as Image
from Helpers import *
from cannyEdgeDetection import myCanny


# filename = "dewaldFox"
# folderName = "../../FONTS/" + "temp"
# sentence = "thequickbrownfoxjumpsoverthelazydog"

def extract_and_label_letters(filename, folderName, sentence):
    image = Image.open(filename + ".jpg")
    image = np.array(image)
    orig = np.copy(image)
    gray = np.array(Image.fromarray(image).convert("F"))
    edged = myCanny(gray, lowThresholdRatio=0.12, highThresholdRatio=0.22)
    edged = np.array(edged, dtype=np.uint8)
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    boxes = []
    for c in contours:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        boxes.append((x, y, w, h))

    # for (x, y, w, h) in boxes:
    #     # draw the prediction on the image
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    merge_margin = 50
    merge_margin_x = 5
    merge_margin_y = 10

    # make each of the bounding boxes bigger based on the margins
    for i, (x, y, w, h) in enumerate(boxes):
        x -= merge_margin_x
        y -= merge_margin_y
        w += 2 * merge_margin_x
        h += 2 * merge_margin_y
        boxes[i] = (x, y, w, h)

    finished = False
    while not finished:
        finished = True

        index = len(boxes) - 1
        while index >= 0:
            x, y, w, h = boxes[index]

            # get matching boxes
            overlaps = getAllOverlaps(boxes, (x, y, w, h), index)

            # check if box has no overlaps
            if len(overlaps) > 0:
                con = []
                overlaps.append(index)
                for ind in overlaps:
                    x, y, w, h = boxes[ind]
                    con.append([x, y])
                    con.append([x + w, y + h])
                con = np.array(con)

                # get bounding rect
                x, y, w, h = cv2.boundingRect(con)

                w -= 1
                h -= 1
                merged = (x, y, w, h)

                # remove boxes from list
                overlaps.sort(reverse=True)
                for ind in overlaps:
                    del boxes[ind]
                boxes.append(merged)

                finished = False
                break

            index -= 1

    copy = np.copy(image)
    for (x, y, w, h) in boxes:
        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 1)
    resized_image = cv2.resize(copy, (850, 650))
    cv2.imshow("Final", resized_image)
    cv2.waitKey(0)

    img = Image.open(filename + ".jpg")
    imgArr = np.array(img)

    boxesDictionary = {0: []}
    for (x, y, w, h) in boxes:
        rangePoint = 60
        middleX, middleY = findMiddle(x, y, w, h)
        keys_in_range = [key for key in boxesDictionary if middleY - rangePoint <= key <= middleY + rangePoint]
        # is there an element in the dictionary with a similar y value?
        if keys_in_range:
            # if so add it to the list
            boxesDictionary[keys_in_range[0]].append((x, y, w, h))
        else:
            # if not create a new list
            boxesDictionary[middleY] = [(x, y, w, h)]

    for key in boxesDictionary.keys():
        boxesDictionary[key].sort(key=lambda tup: tup[0])

    sortedBoxes = []
    for key in sorted(boxesDictionary.keys()):
        for val in boxesDictionary[key]:
            sortedBoxes.append(val)

    sortedBoxes = np.array(sortedBoxes)

    # this part labels and saves the letters ########
    groupedLetters = []

    counter = 0
    for (x, y, w, h) in sortedBoxes:
        crop = imgArr[y:y + h, x:x + w]
        crop = Image.fromarray(crop)
        letterVariable = sentence[counter]
        if letterVariable.isupper():
            letterVariable = letterVariable + "_"
        if not os.path.exists(folderName + "/" + str(letterVariable)):
            os.makedirs(folderName + "/" + str(letterVariable))
        crop.save(
            folderName + "/" + str(letterVariable) + "/" + getName(folderName + "/" + str(letterVariable)) + ".jpg")
        counter += 1
