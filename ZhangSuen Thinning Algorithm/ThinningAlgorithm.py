import numpy as np
from PIL import Image


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


def displayImage(imageArr):
    imageArray = np.array(imageArr, dtype=np.uint8)
    img = Image.fromarray(imageArray, 'L')
    img.show()


def saveImage(imageArr, filename):
    imageArray = np.array(imageArr, dtype=np.uint8)
    img = Image.fromarray(imageArray, 'L')
    img.save(filename)


def A(imageArr, pixelX, pixelY):
    p2 = imageArr[pixelX, pixelY - 1]
    p3 = imageArr[pixelX + 1, pixelY - 1]
    p4 = imageArr[pixelX + 1, pixelY]
    p5 = imageArr[pixelX + 1, pixelY + 1]
    p6 = imageArr[pixelX, pixelY + 1]
    p7 = imageArr[pixelX - 1, pixelY + 1]
    p8 = imageArr[pixelX - 1, pixelY]
    p9 = imageArr[pixelX - 1, pixelY - 1]
    pixels = [-1, p2, p3, p4, p5, p6, p7, p8, p9]

    numberOfTransitionsFromWhiteToBlack = 0
    PreviousPixel = pixels[1]
    for i in range(1, 10):
        if i == 9:
            if pixels[8] == 255 and pixels[1] == 0:
                numberOfTransitionsFromWhiteToBlack += 1
        else:
            if pixels[i] == 0 and PreviousPixel == 255:  # black pixel is 0, white pixel is 255
                numberOfTransitionsFromWhiteToBlack += 1
            PreviousPixel = pixels[i]
    return numberOfTransitionsFromWhiteToBlack


def B(imageArr, pixelX, pixelY):
    p2 = imageArr[pixelX, pixelY - 1]
    p3 = imageArr[pixelX + 1, pixelY - 1]
    p4 = imageArr[pixelX + 1, pixelY]
    p5 = imageArr[pixelX + 1, pixelY + 1]
    p6 = imageArr[pixelX, pixelY + 1]
    p7 = imageArr[pixelX - 1, pixelY + 1]
    p8 = imageArr[pixelX - 1, pixelY]
    p9 = imageArr[pixelX - 1, pixelY - 1]
    pixels = [-1, p2, p3, p4, p5, p6, p7, p8, p9]

    numberOfBlackPixels = 0
    for i in range(1, 9):
        if pixels[i] != 0:
            numberOfBlackPixels += 1
    return numberOfBlackPixels


def has8Neighbors(imageArr, pixelX, pixelY):
    if pixelX - 1 < 0 or pixelY - 1 < 0 or pixelX + 1 >= imageArr.shape[0] or pixelY + 1 >= imageArr.shape[1]:
        return False
    else:
        return True


def checkNESW(N, E, S, W, imageArr, pixelX, pixelY):
    """
    This checks if the pixel has at least one neighbor in the directions specified
    """
    if N:
        if imageArr[pixelX, pixelY - 1] == 255:
            return True
    if E:
        if imageArr[pixelX + 1, pixelY] == 255:
            return True
    if S:
        if imageArr[pixelX, pixelY + 1] == 255:
            return True
    if W:
        if imageArr[pixelX - 1, pixelY] == 255:
            return True
    return False


def pass1(imageArr, pixelX, pixelY):
    pixelMarkedForDeletion = False
    if A(imageArr, pixelX, pixelY) == 1:
        if 2 <= B(imageArr, pixelX, pixelY) <= 6:
            if checkNESW(True, True, True, False, imageArr, pixelX, pixelY):
                if checkNESW(False, True, True, True, imageArr, pixelX, pixelY):
                    pixelMarkedForDeletion = True
    return pixelMarkedForDeletion


def pass2(imageArr, pixelX, pixelY):
    pixelMarkedForDeletion = False
    if A(imageArr, pixelX, pixelY) == 1:
        if 2 <= B(imageArr, pixelX, pixelY) <= 6:
            if checkNESW(True, True, False, True, imageArr, pixelX, pixelY):
                if checkNESW(True, False, True, True, imageArr, pixelX, pixelY):
                    pixelMarkedForDeletion = True
    return pixelMarkedForDeletion


def zhangSuenThinning(imageArr):
    somethingChanged = True
    counter = 0
    while somethingChanged:
        somethingChanged = False
        markerImage = np.zeros(imageArr.shape)
        blackPixels = np.where(imageArr == 0)  # if the pixel is white then skip it
        for (x, y) in zip(blackPixels[0], blackPixels[1]):
            if not has8Neighbors(imageArr, x, y):
                continue
            if pass1(imageArr, x, y):
                markerImage[x, y] = 255
                somethingChanged = True
        imageArr = imageArr + markerImage

        markerImage = np.zeros(imageArr.shape)
        blackPixels = np.where(imageArr == 0)  # if the pixel is white then skip it
        for (x, y) in zip(blackPixels[0], blackPixels[1]):
            if not has8Neighbors(imageArr, x, y):
                continue
            if pass2(imageArr, x, y):
                markerImage[x, y] = 255
                somethingChanged = True
        imageArr = imageArr + markerImage

        if not somethingChanged:
            return imageArr
        counter += 1
        print("something changed: ", counter)


if __name__ == '__main__':
    image = readImage('i ate a fish today.jpg')
    image = imageToBinary(image, threshold=110)
    displayImage(image)
    image = zhangSuenThinning(image)
    displayImage(image)
    saveImage(image, 'i ate a fish today THINNED.jpg')



