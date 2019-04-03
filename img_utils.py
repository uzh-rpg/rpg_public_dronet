import cv2
import numpy as np



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

    return np.asarray(img, dtype=np.float32)
