#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 theomorales <theomorales@Theos-MacBook-Pro.local>
#
# Distributed under terms of the MIT license.

"""
Evaluate the gate detection and localization accuracy
"""

import os
import sys
import utils
import gflags
import cnn_models
import numpy as np
import tensorflow as tf


from tqdm import *
from math import sqrt
from PIL import Image, ImageDraw
from keras import backend as K
from common_flags import FLAGS
from constants import TEST_PHASE

def median_filter(prediction, previous_predictions):
    if len(previous_predictions) < FLAGS.successive_frames:
        return prediction
    window = previous_predictions + [prediction]
    window.sort()
    return window[int(len(window)/2)]

def save_visual_output(img, prediction, index):
    draw = ImageDraw.Draw(img)

    sqrt_win = int(sqrt(FLAGS.nb_windows))
    window_width = FLAGS.img_width / sqrt_win
    window_height = FLAGS.img_height / sqrt_win

    pred_window = prediction

    if pred_window == 0:
        draw.text(((img.width / 2)-30, (img.height/2)-5), "NO GATE", "red")
    else:
        # Draw a red square at the estimated region
        window_idx = pred_window % sqrt_win
        if window_idx == 0:
            window_idx = sqrt_win
        window_x = (window_idx - 1) * window_width
        window_y = window_height * int(pred_window/sqrt_win)
        draw.rectangle([(window_x, window_y),
                       (window_x + window_width, window_y + window_height)],
                       outline="red")
    # Save img
    if not os.path.isdir("visualizations"):
        os.mkdir("visualizations")
    img.save("visualizations/%06d.png" % index)


def _main():

    # Set testing mode (dropout/batch normalization)
    K.set_learning_phase(TEST_PHASE)

    # Input image dimensions
    img_width, img_height = FLAGS.img_width, FLAGS.img_height
    img_channels = 3 if FLAGS.img_mode == "rgb" else 1
    output_dim = FLAGS.nb_windows + 1

    images = []
    path = os.path.join(FLAGS.test_dir, "images")
    print("[*] Loading input images from {}".format(path))
    for file in sorted(os.listdir(path)):
        file = os.path.join(path, file)
        if os.path.isfile(file):
            images.append(file)

    # Load json and create model
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    model = utils.jsonToModel(json_model_path)

    # Load weights
    weights_load_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.weights_fname)
    try:
        model.load_weights(weights_load_path)
        print("Loaded model from {}".format(weights_load_path))
    except Exception as e:
        print(e)

    # Compile model
    model.compile(loss='mse', optimizer='adam')
    graph = tf.get_default_graph()

    if (FLAGS.successive_frames % 2) != 0:
        FLAGS.successive_frames -= 1

    print("[*] Generating {} prediction images...".format(len(images)))
    previous_predictions = []
    n = 0
    step = 10
    with graph.as_default():
        for image in tqdm(images):
            img = Image.open(image)
            np_image = np.array(img).astype(np.float64)
            np_image *= (1./255.)
            np_image = np.expand_dims(np_image, axis=0)
            prediction = np.argmax(model.predict(np_image))
            if FLAGS.filter:
                filtered_pred = median_filter(prediction, previous_predictions)
                if len(previous_predictions) >= FLAGS.successive_frames:
                    del previous_predictions[0]
                save_visual_output(img, filtered_pred, n)
                previous_predictions.append(prediction)
            else:
                save_visual_output(img, prediction, n)
            n += 1


def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
      sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)
