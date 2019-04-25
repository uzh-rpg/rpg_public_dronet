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


from math import sqrt
from PIL import Image, ImageDraw
from keras import backend as K
from common_flags import FLAGS
from constants import TEST_PHASE

def save_visual_output(input_img, prediction, index):
    if FLAGS.img_mode == "rgb":
        img_mode = "RGB"
    else:
        img_mode = "L"
    input_img *= 255.0/input_img.max()
    np_array = np.uint8(input_img)
    img = Image.fromarray(np_array.reshape((np_array.shape[0],
                                           np_array.shape[1])), mode=img_mode)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    sqrt_win = int(sqrt(FLAGS.nb_windows))
    window_width = FLAGS.img_width / sqrt_win
    window_height = FLAGS.img_height / sqrt_win

    pred_window = np.argmax(prediction)

    if pred_window == 0:
        draw.text(((img.width / 2)-30, (img.height/2)-5), "NO GATE", "red")
    else:
        # Draw a red square at the estimated region
        window_idx = pred_window % sqrt_win
        if window_idx == 0:
            window_indx = sqrt_win
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

    # Set testing mode (dropout/batchnormalization)
    K.set_learning_phase(TEST_PHASE)

    # Input image dimensions
    img_width, img_height = FLAGS.img_width, FLAGS.img_height

    # Generate testing data
    test_datagen = utils.DroneDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(FLAGS.test_dir,
                          shuffle=False,
                          color_mode=FLAGS.img_mode,
                          target_size=(FLAGS.img_width, FLAGS.img_height),
                          batch_size = FLAGS.batch_size,
                          max_samples=FLAGS.nb_visualizations)

    # Load json and create model
    # json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    # model = utils.jsonToModel(json_model_path)
    img_channels = 3 if FLAGS.img_mode == "rgb" else 1
    output_dim = FLAGS.nb_windows + 1
    model = cnn_models.resnet8(FLAGS.img_width, FLAGS.img_height, img_channels, output_dim)

    # Load weights
    weights_load_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.weights_fname)
    try:
        model.load_weights(weights_load_path)
        print("Loaded model from {}".format(weights_load_path))
    except Exception as e:
        print(e)


    # Compile model
    model.compile(loss='mse', optimizer='adam')

    # Get predictions and ground truth
    n_samples = test_generator.samples
    nb_batches = int(np.ceil(n_samples / FLAGS.batch_size))
    localization_accuracy = 0

    n = 0
    step = 10
    for i in range(0, nb_batches, step):
        inputs, predictions = utils.compute_predictions(
                model, test_generator, step, verbose = 1)

        for j in range(len(inputs)):
            save_visual_output(inputs[j], predictions[j], n)
            n += 1

    print("[*] Generating {} prediction images...".format(n))

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
