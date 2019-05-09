import cv2
import numpy as np
from PIL import Image


def load_img(path, grayscale=False):
    """
    Load an image.

    # Arguments
        path: Path to image file.
        grayscale: Boolean, whether to load the image as grayscale.

    # Returns
        Image as numpy array.
    """

    img = cv2.imread(path)
    if grayscale:
        if len(img.shape) != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape((img.shape[0], img.shape[1], 1))
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return np.asarray(img, dtype=np.float64)


def random_channel_shift(img, shiftFactor, channel_axis=2):
    '''
    For RGB Only
    '''
    x = img.copy()
    x = np.rollaxis(x, channel_axis, 0)
    intensity = np.random.randint(-255*shiftFactor, 255*shiftFactor)
    channel_idx = np.random.randint(0, 2)
    min_x, max_x = np.min(x[channel_idx]), np.max(x[channel_idx])
    channel = np.clip(x[channel_idx] + intensity, min_x, max_x)
    x[channel_idx] = channel
    x = np.rollaxis(x, 0, channel_axis+1)

    return x.astype(np.float64)

def add_shade(img, weight=0.75):
    rows, cols, _ = img.shape
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5

    shade = np.random.normal(mean, sigma, (img.shape[0], img.shape[1],
                                 1)).astype(np.float64)
    shade = np.concatenate((shade, shade, shade), axis=2)
    shaded_img = cv2.addWeighted(img, weight, 0.25*shade, 0.25, 0)

    return shaded_img

def add_salt_and_pepper(img, ratio=0.2, amount=0.004):
    x = img.copy()
    num_salt = np.ceil(amount*x.size*ratio)
    num_pepper = np.ceil(amount*x.size*(1.0-ratio))

    # Add salt
    coords = [np.random.randint(0, i-1, int(num_salt)) for i in x.shape]
    x[coords[0], coords[1], :] = 1

    # Add pepper
    coords = [np.random.randint(0, i-1, int(num_salt)) for i in x.shape]
    x[coords[0], coords[1], :] = 0

    return x
