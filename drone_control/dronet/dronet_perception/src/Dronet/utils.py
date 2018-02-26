from cv_bridge import CvBridge, CvBridgeError
from keras.models import model_from_json
import cv2
import numpy as np
import rospy

bridge = CvBridge()

def callback_img(data, target_size, crop_size, rootpath, save_img):
    try:
        image_type = data.encoding
        img = bridge.imgmsg_to_cv2(data, image_type)
    except CvBridgeError, e:
        print e

    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = central_image_crop(img, crop_size[0], crop_size[1])

    if rootpath and save_img:
        temp = rospy.Time.now()
        cv2.imwrite("{}/{}.jpg".format(rootpath, temp), img)

    return np.asarray(img, dtype=np.float32) * np.float32(1.0/255.0)


def central_image_crop(img, crop_width, crop_heigth):
    """
    Crops the input PILLOW image centered in width and starting from the bottom
    in height.
    Arguments:
        crop_width: Width of the crop
        crop_heigth: Height of the crop
    Returns:
        Cropped image
    """
    half_the_width = img.shape[1] / 2
    img = img[(img.shape[0] - crop_heigth): img.shape[0],
              (half_the_width - (crop_width / 2)): (half_the_width + (crop_width / 2))]
    img = img.reshape(img.shape[0], img.shape[1], 1)
    return img

def jsonToModel(json_model_path):
    with open(json_model_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)

    return model
