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
import numpy as np


from math import sqrt
from PIL import Image, ImageDraw
from keras import backend as K
from common_flags import FLAGS
from constants import TEST_PHASE


def compute_gate_localization_accuracy(predictions, ground_truth):
    valid = 0
    for i, pred in enumerate(predictions):
        pred_clean = np.zeros(len(pred))
        pred_clean[np.argmax(pred)] = 1.0
        if np.array_equal(pred_clean, ground_truth[i]):
            valid += 1

    return int(valid / len(ground_truth) * 100)

def save_visual_output(input, prediction, ground_truth):
    img = Image.fromarray(np.uint8(input), mode="RGB")
    draw = ImageDraw.Draw(img)

    pred_window = np.argmax(prediction)
    gt_window = np.argmax(ground_truth)

    sqrt_win = int(sqrt(FLAGS.nb_windows))
    window_width = FLAGS.img_width / sqrt_win
    window_height = FLAGS.img_height / sqrt_win

    # Draw a green cross at the ground truth location
    if gt_window != 0:
        window_x = window_width * (gt_window - (sqrt_win * int(gt_window/sqrt_win)) - 1)
        window_y = window_height * int(gt_window/sqrt_win)
        draw.rectangle([(window_x, window_y),
                       (window_x + window_width, window_y + window_height)],
                       outline="green", width=5)

    if pred_window == 0:
        draw.text(((img.width / 2)-30, (img.height/2)-5), "NO GATE", (255, 0, 0, 255))
    else:
        # Draw a red square at the estimated region
        window_x = window_width * (pred_window - (sqrt_win * int(pred_window/sqrt_win)) - 1)
        window_y = window_height * int(pred_window/sqrt_win)
        draw.rectangle([(window_x, window_y),
                       (window_x + window_width, window_y + window_height)],
                       outline="red")
    # Save img
    if not os.path.isdir("visualizations"):
        os.mkdir("visualizations")
    img.save("visualizations/ground_truth_window-{}.png".format(gt_window))


def _main():

    # Set testing mode (dropout/batchnormalization)
    K.set_learning_phase(TEST_PHASE)

    # Input image dimensions
    img_width, img_height = FLAGS.img_width, FLAGS.img_height

    # Generate testing data
    test_datagen = utils.DroneDataGenerator()
    test_generator = test_datagen.flow_from_directory(FLAGS.test_dir,
                          shuffle=True,
                          color_mode=FLAGS.img_mode,
                          target_size=(FLAGS.img_width, FLAGS.img_height),
                          batch_size = FLAGS.batch_size,
                          max_samples=None)

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

    # Get predictions and ground truth
    n_samples = test_generator.samples
    nb_batches = int(np.ceil(n_samples / FLAGS.batch_size))

    inputs, predictions, ground_truth = utils.compute_predictions_and_gt(
            model, test_generator, nb_batches, verbose = 1)

    localization_accuracy = compute_gate_localization_accuracy(predictions,
                                                               ground_truth)

    print("[*] Gate localization accuracy: {}%".format(localization_accuracy))
    print("[*] Generating {} prediction images...".format(FLAGS.nb_visualizations))
    for i in range(FLAGS.nb_visualizations):
        save_visual_output(inputs[i], predictions[i], ground_truth[i])

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
