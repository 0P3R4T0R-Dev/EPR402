import numpy as np
import time

def speedTest():
    # Generate a random single-channel image of shape (33, 33, 1)
    image = np.random.rand(34, 34, 64)
    image2 = np.copy(image)

    # Define the pool size
    pool_size = 2

    start = time.time()
    # Reshape the image to have the shape (17, 2, 17, 2, 1)
    reshaped_image = image.reshape(image.shape[0] // pool_size, pool_size, image.shape[1] // pool_size, pool_size, 64)

    # Apply max pooling by taking the maximum value along the (1, 3) axes
    output_image_complex = reshaped_image.max(axis=(1, 3))

    complexTime = time.time() - start
    print("Time taken complex: %f" % complexTime)


    # Define the pool size
    pool_size = 2

    start = time.time()

    output_image = np.zeros((image2.shape[0] // pool_size, image2.shape[1] // pool_size, 64))

    # Perform max pooling

    for k in range(image2.shape[2]):
        for i in range(0, image2.shape[0], pool_size):
            for j in range(0, image2.shape[1], pool_size):
                # Extract the region to pool
                pool_region = image2[i:i + pool_size, j:j + pool_size, k]

                # Store the maximum value in the output image
                output_image[i//pool_size, j//pool_size, k] = np.max(pool_region)

    simpleTime = time.time() - start
    print("Time taken simple: %f" % simpleTime)


    if np.allclose(output_image, output_image_complex):
        print("The pooling output images match.")

    print("Time difference: %f milliseconds" % ((simpleTime - complexTime)*1000))


def maxPoolingSimple(image, pool_size=2):
    output_image = np.zeros((image.shape[0] // pool_size, image.shape[1] // pool_size, image.shape[2]))
    for k in range(image.shape[2]):
        for i in range(0, image.shape[0]-1, pool_size):
            for j in range(0, image.shape[1]-1, pool_size):
                # Extract the region to pool
                pool_region = image[i:i + pool_size, j:j + pool_size, k]

                # Store the maximum value in the output image
                print(i//pool_size, j//pool_size, k)
                output_image[i // pool_size, j // pool_size, k] = np.max(pool_region)
    return output_image


def maxPooling(image, pool_size=2):
    # Reshape the image to have the shape (17, 2, 17, 2, 1)
    reshaped_image = image.reshape(image.shape[0] // pool_size, pool_size, image.shape[1] // pool_size, pool_size, image.shape[2])

    # Apply max pooling by taking the maximum value along the (1, 3) axes
    output_image_complex = reshaped_image.max(axis=(1, 3))
    return output_image_complex


def maxPoolingFixed(image, pool_size=2):
    if image.shape[0] % pool_size != 0 or image.shape[1] % pool_size != 0:
        blank_row = np.zeros((1, image.shape[1], image.shape[2]))
        blank_col = np.zeros((image.shape[0] + 1, 1, image.shape[2]))
        image = np.concatenate((image, blank_row), axis=0)
        image = np.concatenate((image, blank_col), axis=1)
    pooledResult = maxPooling(image, pool_size)
    return pooledResult[:-1, :-1, :]


if __name__ == "__main__":
    image1 = np.random.rand(33, 33, 5)
    image2 = np.copy(image1)
    output_image_1 = maxPoolingSimple(image1)
    output_image_2 = maxPoolingFixed(image2)
    if np.allclose(output_image_1, output_image_2):
        print("The pooling output images match.")
    print("output_image_1 shape: %s" % str(output_image_1.shape))
    print("output_image_2 shape: %s" % str(output_image_2.shape))


