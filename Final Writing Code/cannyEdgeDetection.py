from gaussianBlur import convGaussBlur
import numpy as np
import PIL.Image as Image
from scipy.ndimage import convolve
import time


def sobel_filters(img):
    Kx = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]
    Kx = np.array(Kx, np.float32)
    Ky = [[1, 2, 1],
          [0, 0, 0],
          [-1, -2, -1]]
    Ky = np.array(Ky, np.float32)

    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)

    MagnitudeG = np.hypot(Ix, Iy)  # sqrt(Ix**2 + Iy**2)
    MagnitudeG = MagnitudeG / MagnitudeG.max() * 255.0  # normalize to 0-255
    directionTheta = np.arctan2(Iy, Ix)
    return MagnitudeG, directionTheta


def findAngle(angle):
    if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
        return 0
    elif 22.5 <= angle < 67.5:
        return 45
    elif 67.5 <= angle < 112.5:
        return 90
    elif 112.5 <= angle < 157.5:
        return 135


def non_max_suppression(img, D):
    """
    Done to create thin edges instead of thick ones
    img: gradient magnitude
    D: gradient direction
    returns: image but with thin edges
    """
    rows, cols = img.shape
    canvas = np.zeros((rows, cols), dtype=np.int32)
    angle = D * 180. / np.pi  # convert angles into degrees
    angle[angle < 0] += 180

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighborPixels = (0, 0)

            angleVal = findAngle(angle[i, j])

            if angleVal == 0:
                neighborPixels = (img[i, j + 1], img[i, j - 1])
            elif angleVal == 45:
                neighborPixels = (img[i - 1, j + 1], img[i + 1, j - 1])
            elif angleVal == 90:
                neighborPixels = (img[i - 1, j], img[i + 1, j])
            elif angleVal == 135:
                neighborPixels = (img[i - 1, j - 1], img[i + 1, j + 1])

            if img[i, j] >= max(neighborPixels):
                canvas[i, j] = img[i, j]

    return canvas


def threshold(img, lowThresholdRatio=0.09, highThresholdRatio=0.17):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    # highThreshold = 150
    # lowThreshold = 30

    canvas = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint32)

    canvas[img >= highThreshold] = 255
    canvas[(img <= highThreshold) & (img >= lowThreshold)] = 100

    return canvas


def hasStrongNeighbour(img, x, y):
    p1 = img[x + 1, y - 1]
    p2 = img[x + 1, y]
    p3 = img[x + 1, y + 1]
    p4 = img[x, y - 1]
    p5 = img[x, y + 1]
    p6 = img[x - 1, y - 1]
    p7 = img[x - 1, y]
    p8 = img[x - 1, y + 1]
    pixels = np.array([p1, p2, p3, p4, p5, p6, p7, p8])
    if pixels.any() == 255:
        return True
    else:
        return False


def hysteresis(img, weak=100, strong=255):
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if img[i, j] == weak:
                if hasStrongNeighbour(img, i, j):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img


def myCanny(img, sigma=1.4, lowThresholdRatio=0.09, highThresholdRatio=0.17):
    """ REMEMBER TO CONVERT TO GRAYSCALE FIRST USING PIL.Image.convert("F") """
    img = convGaussBlur(img, sigma=sigma)
    img, DirTheta = sobel_filters(img)
    img = non_max_suppression(img, DirTheta)
    img = threshold(img, lowThresholdRatio=lowThresholdRatio, highThresholdRatio=highThresholdRatio)
    img = hysteresis(img)
    return img


if __name__ == "__main__":
    image = Image.open("SoftCat.jpg").convert("F").resize((500, 500))
    image.show()
    image = np.array(image)
    image = convGaussBlur(image, sigma=1.4)
    image, Theta = sobel_filters(image)
    image = non_max_suppression(image, Theta)
    image = threshold(image)
    image = hysteresis(image)
    image = np.array(image)
    image = Image.fromarray(image)
    image.show()




