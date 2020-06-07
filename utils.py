import numpy as np
from skimage import io


def read_image(fname):
    image = io.imread(fname)
    result = np.zeros(shape=[1, 28, 28, 1], dtype=np.float32)
    for yy in range(28):
        for xx in range(28):
            result[0][xx][yy][0] = float(image[xx][yy]) / 255
    return result
