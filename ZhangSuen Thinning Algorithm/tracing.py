import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import math
import serial
from collections import deque


def returnNeighbors(imageArr, pixelX, pixelY):
    # returns the 8 neighbors of a pixel in a list
    pixels = []
    for x in range(pixelX - 1, pixelX + 2):
        for y in range(pixelY - 1, pixelY + 2):
            if x == pixelX and y == pixelY:
                continue
            else:
                if x < 0 or x >= imageArr.shape[0] or y < 0 or y >= imageArr.shape[1]:
                    continue
                pixels.append([x, y])
    return pixels


def findBlackPixel(imageArr, pixelX=0, pixelY=0, oldMethod=True):
    if oldMethod:
        blackPixels = np.where(imageArr == 0)
        if len(blackPixels[0]) == 0:
            return -1, -1
        return blackPixels[0][0], blackPixels[1][0]
    else:
        # do the breadth first search for looking for the next pixel
        blackPixels = np.where(imageArr == 0)
        if len(blackPixels[0]) == 0:
            return -1, -1
        my_queue = deque()
        visited = np.zeros(imageArr.shape)
        currentPoint = [pixelX, pixelY]
        my_queue.append(currentPoint)
        found = False
        while not found and len(my_queue) > 0:
            if len(my_queue) >= 1000:
                return findBlackPixel(imageArr)
            currentPoint = my_queue.popleft()
            visited[currentPoint[0], currentPoint[1]] = 1
            if imageArr[currentPoint[0], currentPoint[1]] == 0:
                found = True
            else:
                # add the neighbors that are not in visited into the queue
                neighbors = returnNeighbors(imageArr, currentPoint[0], currentPoint[1])
                for neighbor in neighbors:
                    if visited[neighbor[0]][neighbor[1]] == 0 and neighbor not in my_queue:
                        my_queue.append(neighbor)
        return currentPoint[0], currentPoint[1]


def diagonalMakeWhite(imageArr, pixelX, pixelY, direction):
    if direction == 1:
        imageArr[pixelX, pixelY - 1] = 255
        imageArr[pixelX + 1, pixelY] = 255
    elif direction == 2:
        imageArr[pixelX, pixelY + 1] = 255
        imageArr[pixelX + 1, pixelY] = 255
    elif direction == 3:
        imageArr[pixelX - 1, pixelY] = 255
        imageArr[pixelX, pixelY + 1] = 255
    elif direction == 4:
        imageArr[pixelX, pixelY - 1] = 255
        imageArr[pixelX - 1, pixelY] = 255


def nextPixel(imageArr, pixelX, pixelY):
    """
    This funtion returns the next x and y pixels of the next black pixel
    Returns -1, -1 if there are no more black pixels
    """
    p2 = [imageArr[pixelX, pixelY - 1], pixelX, pixelY - 1]
    p3 = [imageArr[pixelX + 1, pixelY - 1], pixelX + 1, pixelY - 1]
    p4 = [imageArr[pixelX + 1, pixelY], pixelX + 1, pixelY]
    p5 = [imageArr[pixelX + 1, pixelY + 1], pixelX + 1, pixelY + 1]
    p6 = [imageArr[pixelX, pixelY + 1], pixelX, pixelY + 1]
    p7 = [imageArr[pixelX - 1, pixelY + 1], pixelX - 1, pixelY + 1]
    p8 = [imageArr[pixelX - 1, pixelY], pixelX - 1, pixelY]
    p9 = [imageArr[pixelX - 1, pixelY - 1], pixelX - 1, pixelY - 1]

    p10 = [imageArr[pixelX, pixelY - 2], pixelX, pixelY - 2]
    p11 = [imageArr[pixelX + 1, pixelY - 2], pixelX + 1, pixelY - 2]
    p12 = [imageArr[pixelX + 2, pixelY - 2], pixelX + 2, pixelY - 2]
    p13 = [imageArr[pixelX + 2, pixelY - 1], pixelX + 2, pixelY - 1]
    p14 = [imageArr[pixelX + 2, pixelY], pixelX + 2, pixelY]
    p15 = [imageArr[pixelX + 2, pixelY + 1], pixelX + 2, pixelY + 1]
    p16 = [imageArr[pixelX + 2, pixelY + 2], pixelX + 2, pixelY + 2]
    p17 = [imageArr[pixelX + 1, pixelY + 2], pixelX + 1, pixelY + 2]
    p18 = [imageArr[pixelX, pixelY + 2], pixelX, pixelY + 2]
    p19 = [imageArr[pixelX - 1, pixelY + 2], pixelX - 1, pixelY + 2]
    p20 = [imageArr[pixelX - 2, pixelY + 2], pixelX - 2, pixelY + 2]
    p21 = [imageArr[pixelX - 2, pixelY + 1], pixelX - 2, pixelY + 1]
    p22 = [imageArr[pixelX - 2, pixelY], pixelX - 2, pixelY]
    p23 = [imageArr[pixelX - 2, pixelY - 1], pixelX - 2, pixelY - 1]
    p24 = [imageArr[pixelX - 2, pixelY - 2], pixelX - 2, pixelY - 2]
    p25 = [imageArr[pixelX - 1, pixelY - 2], pixelX - 1, pixelY - 2]

    pixels = [-1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
              p16, p17, p18, p19, p20, p21, p22, p23, p24, p25]

    for i in range(1, len(pixels)):
        if pixels[i][0] == 0:
            if i == 2:
                diagonalMakeWhite(imageArr, pixelX, pixelY, 1)
            elif i == 4:
                diagonalMakeWhite(imageArr, pixelX, pixelY, 2)
            elif i == 6:
                diagonalMakeWhite(imageArr, pixelX, pixelY, 3)
            elif i == 8:
                diagonalMakeWhite(imageArr, pixelX, pixelY, 4)
            return pixels[i][1], pixels[i][2]
    return -1, -1

# keep following along the black pixel until you can can't
# for each black pixel, add a white pixel to a black image in the same location
# display the image often to show the tracing process
# once you have found a black pixel, make it white so you don't find it again


# if you can't, then you have a black pixel with no neighbors in that direction
# find the next black pixel and repeat
# (this will trigger a lift pen up event to move to the next black pixel)


def readImage(filename):
    return Image.open(filename).convert('L')


def imageToBinary(image, threshold=128, flipped=False):
    """the higher the threshold the more black pixels there will be"""
    upper = 255
    lower = 0
    if flipped:
        upper = 0
        lower = 255
    imageArr = np.array(image)
    copyImage = np.copy(imageArr)
    imageArr[copyImage > threshold] = upper
    imageArr[copyImage <= threshold] = lower
    return imageArr


def displayImage(imageArr, final=False):
    # imageArray = np.array(imageArr, dtype=np.uint8)
    # img = Image.fromarray(imageArray, 'L')
    # img.show()
    plt.imshow(imageArr, cmap='gray')  # Use 'gray' colormap for grayscale images
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show(block=False)  # Display the images
    if final:
        plt.show()
    else:
        plt.pause(0.0001)


def controlPenZ(up="UP"):
    if up == "UP":
        print("Pen up")
        serial_connection.write(("0,0,0,0," + str(1) + "\n").encode())
    else:
        print("Pen down")
        serial_connection.write(("0,0,0,0," + str(0) + "\n").encode())
    line1 = serial_connection.readline().decode().strip()
    print("Arduino: ", line1)
    while line1 != "OK updown":
        print("Arduino: ", line1)
        line1 = serial_connection.readline().decode().strip()
        time.sleep(0.001)
    print("Arduino: ", line1)


def euclidean_distance(point1, point2):
    squared_distance = sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))
    distance = math.sqrt(squared_distance)
    return distance


def movePen(fromX, fromY, toX, toY):
    print("Move pen from", fromX, fromY, "to", toX, toY)
    serial_connection.write((str(fromX) + "," + str(fromY) + "," + str(toX) + "," + str(toY) + ",-1\n").encode())
    line1 = serial_connection.readline().decode().strip()
    print("Arduino: ", line1)
    while line1 != "OK":
        print("Arduino: ", line1)
        line1 = serial_connection.readline().decode().strip()
        time.sleep(0.001)
    print("Arduino: ", line1)


serial_connection = serial.Serial('COM5', 115200)
line = serial_connection.readline().decode().strip()
while line != "Ready":
    print("Arduino: ", line)
    line = serial_connection.readline().decode().strip()
    time.sleep(0.001)
print("Arduino: ", line)
startTime = time.time()
image = readImage('AnneliseThinned.jpg')
imArr = imageToBinary(image, threshold=90)
# imArr = np.matrix(imArr)
# imArr = imArr.transpose()
# imArr = np.array(imArr)
canvas = np.zeros((imArr.shape[0], imArr.shape[1]))
counter = 0
pointer = findBlackPixel(imArr, 0, 0, oldMethod=False)
# pointer = findBlackPixel(imArr)
imArr[pointer[0], pointer[1]] = 255
controlPenZ("DOWN")
while pointer != (-1, -1):
    prevPointer = pointer
    pointer = nextPixel(imArr, pointer[0], pointer[1])
    if pointer != (-1, -1):
        movePen(prevPointer[0], prevPointer[1], pointer[0], pointer[1])
        canvas[pointer[0], pointer[1]] = 255
    else:
        controlPenZ("UP")
        print("Looking for next black pixel")
        pointer = findBlackPixel(imArr, prevPointer[0], prevPointer[1], oldMethod=False)
        # pointer = findBlackPixel(imArr)
        if pointer != (-1, -1):
            canvas[pointer[0], pointer[1]] = 255
            movePen(prevPointer[0], prevPointer[1], pointer[0], pointer[1])
            controlPenZ("DOWN")
        else:
            print("No black pixels found")
            break
    imArr[pointer[0], pointer[1]] = 255
    counter += 1
endTime = time.time()
print("Time taken: ", endTime - startTime)
#     if counter % 10 == 0:
#         displayImage(canvas)
# displayImage(canvas, final=True)

# add a function that checks if the pixels has 2 neighbours in each direction








