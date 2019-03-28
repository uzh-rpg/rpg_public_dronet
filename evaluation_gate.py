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
import gflags
import utils
import numpy as np


from common_flags import FLAGS
from keras import backend as K
from constants import TEST_PHASE


def compute_gate_localization_accuracy(predictions, ground_truth):
    valid = 0
    for i, pred in enumerate(predictions):
        if pred == ground_truth[i]:
            valid += 1

    return int(valid / len(ground_truth) * 100)

def _main():

    # Set testing mode (dropout/batchnormalization)
    K.set_learning_phase(TEST_PHASE)

    # Input image dimensions
    img_width, img_height = FLAGS.img_width, FLAGS.img_height

    # Cropped image dimensions
    crop_img_width, crop_img_height = FLAGS.crop_img_width, FLAGS.crop_img_height


    if FLAGS.no_crop:
        crop_size = None
        crop_img_height = img_height
        crop_img_width = img_width
    else:
        crop_size = (crop_img_height, crop_img_width)

    # Generate testing data
    test_datagen = utils.DroneDataGenerator()
    test_generator = test_datagen.flow_from_directory(FLAGS.test_dir,
                          shuffle=False,
                          color_mode=FLAGS.img_mode,
                          target_size=(FLAGS.img_width, FLAGS.img_height),
                                                      crop_size= crop_size,
                          batch_size = FLAGS.batch_size)

    # Load json and create model
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    model = utils.jsonToModel(json_model_path)

    # Load weights
    weights_load_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.weights_fname)
    try:
        model.load_weights(weights_load_path)
        print("Loaded model from {}".format(weights_load_path))
    except:
        print("Impossible to find weight path. Returning untrained model")


    # Compile model
    model.compile(loss='mse', optimizer='adam')

    # Get predictions and ground truth
    n_samples = test_generator.samples
    nb_batches = int(np.ceil(n_samples / FLAGS.batch_size))

    predictions, ground_truth, t = utils.compute_predictions_and_gt(
            model, test_generator, nb_batches, verbose = 1)

    localization_accuracy = compute_gate_localization_accuracy(predictions,
                                                               ground_truth)

    print("[*] Gate localization accuracy: {}%".format(localization_accuracy))

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
