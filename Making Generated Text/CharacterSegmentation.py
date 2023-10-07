from imutils.contours import sort_contours
import numpy as np
import imutils
import cv2
from PIL import Image
import os
from Helpers import *

filename = "dewaldFox"
folderName = "../../FONTS/" + "dewaldCapital"


image = cv2.imread(filename + ".jpg")
orig = np.copy(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert the input image to grayscale (0-255)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # performs standard blurring on grayscale image
# perform edge detection, find contours in the edge map, and sort the
# resulting contours from left-to-right
edged = cv2.Canny(blurred, 30, 150)
# cv2.imshow("Edged", edged)
contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sort_contours(contours, method="left-to-right")[0]

# initialize the list of contour bounding boxes and associated
# characters that we'll be OCR'ing
chars = []

# loop over the contours
for c in contours:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
    chars.append((x, y, w, h))

# extract the bounding box locations and padded characters
boxes = [b for b in chars]
# loop over the predictions and bounding box locations together
for (x, y, w, h) in boxes:
    # draw the prediction on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

merge_margin = 50
merge_margin_x = 5
merge_margin_y = 10


for i, (x, y, w, h) in enumerate(boxes):
    x -= merge_margin_x
    y -= merge_margin_y
    w += 2 * merge_margin_x
    h += 2 * merge_margin_y
    boxes[i] = (x, y, w, h)


# this is gonna take a long time
finished = False
while not finished:
    # set end con
    finished = True

    # loop through boxes
    index = len(boxes) - 1
    while index >= 0:
        # grab current box
        x, y, w, h = boxes[index]

        # # add margin
        # x -= merge_margin_x
        # y -= merge_margin_y
        # w += 2 * merge_margin_x
        # h += 2 * merge_margin_y

        # get matching boxes
        overlaps = getAllOverlaps(boxes, (x, y, w, h), index)

        # check if empty
        if len(overlaps) > 0:
            # combine boxes
            # convert to a contour
            con = []
            overlaps.append(index)
            for ind in overlaps:
                x, y, w, h = boxes[ind]
                con.append([x, y])
                con.append([x + w, y + h])
            con = np.array(con)

            # get bounding rect
            x, y, w, h = cv2.boundingRect(con)

            # stop growing
            w -= 1
            h -= 1
            merged = (x, y, w, h)

            # remove boxes from list
            overlaps.sort(reverse=True)
            for ind in overlaps:
                del boxes[ind]
            boxes.append(merged)

            # set flag
            finished = False
            break

        # increment
        index -= 1

# show final
copy = np.copy(image)
for (x, y, w, h) in boxes:
    if w * h < 300:
        continue
    cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 1)
cv2.imshow("Final", copy)
cv2.waitKey(0)

img = Image.open(filename + ".jpg")
imgArr = np.array(img)

boxesDictionary = {0: []}
for (x, y, w, h) in boxes:
    rangePoint = 60
    middleX, middleY = findMiddle(x, y, w, h)
    keys_in_range = [key for key in boxesDictionary if middleY-rangePoint <= key <= middleY+rangePoint]
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

sentence = "thequickbrownfoxjumpsoverthelazydogTHEQUICKBROWNFOXJUMPSOVERTHELAZYDOG"
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
    crop.save(folderName + "/" + str(letterVariable) + "/" + getName(folderName + "/" + str(letterVariable)) + ".jpg")
    counter += 1
