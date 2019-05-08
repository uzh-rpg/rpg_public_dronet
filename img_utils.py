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

    return np.asarray(img, dtype=np.uint8)


def random_channel_shift(img, shiftFactor, channel_axis=2):
    '''
    For RGB Only
    '''
    x = np.rollaxis(img, channel_axis, 0)
    intensity = np.random.randint(-255*shiftFactor, 255*shiftFactor)
    channel_idx = np.random.randint(0, 2)
    min_x, max_x = np.min(x[channel_idx]), np.max(x[channel_idx])
    channel = np.clip(x[channel_idx] + intensity, min_x, max_x)
    x[channel_idx] = channel
    x = np.rollaxis(x, 0, channel_axis+1)

    return x.astype(np.float64)


