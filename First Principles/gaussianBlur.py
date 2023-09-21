import numpy as np
import PIL.Image as Image
import cv2
from scipy.ndimage import convolve


def pixelBlur(start, k, img):
    blurrBuffer = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            if start[0] + i < 0 or start[1] + j < 0:
                continue
            if start[0] + i >= img.shape[0] or start[1] + j >= img.shape[1]:
                continue
            point = img[start[0] + i][start[1] + j]
            point = point * k[i + 2][j + 2]
            blurrBuffer.append(point)
    return sum(blurrBuffer)


def myGaussBlur(img):
    # make a 5x5 gaussian kernel
    kernel = np.array([[1, 4, 7, 4, 1],
                       [4, 16, 26, 16, 4],
                       [7, 26, 41, 26, 7],
                       [4, 16, 26, 16, 4],
                       [1, 4, 7, 4, 1]]) / 273

    canvas = np.zeros((img.shape[0], img.shape[1]), np.float64)

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            canvas[row][col] = pixelBlur([row, col], kernel, img)

    return canvas


def convGaussBlur(img, sigma=1.0):
    x, y = np.mgrid[-5:6, -5:6]
    kernel = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * (1 / (2.0 * np.pi * sigma ** 2))
    return convolve(img, kernel)


if __name__ == "__main__":
    image = Image.open("SoftCat.jpg").convert("L")
    image.show()
    image = np.array(image)
    imageToShow = convGaussBlur(image, sigma=1.9)
    imageToShow = Image.fromarray(imageToShow)
    imageToShow.show()





