import re
import os
import numpy as np
import tensorflow as tf
import json

from tqdm import *
from math import sqrt
from time import sleep
from keras import backend as K
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import Progbar
from keras.models import model_from_json

import img_utils


def fit_flow_from_directory(config, fit_sample_size, directory, max_samples,
                            target_size=None, color_mode='grayscale',
                            batch_size=32, shuffle=True, seed=None,
                            follow_links=False, nb_windows=25,
                            sample_shape=(255, 340, 1)):
    drone_data_gen = DroneDataGenerator(rescale=1./255)
    fit_drone_data_gen = DroneDataGenerator(**config)
    print("[*] Generating statistically representative samples...")
    batches = drone_data_gen.flow_from_directory(directory, max_samples, target_size,
                                      color_mode, batch_size, shuffle,
                                       seed, follow_links, nb_windows)
    print("[*] Fitting the generator on the samples...")
    for i in tqdm(range(int(batches.samples/batch_size))):
        imgs, labels = next(batches)
        if fit_sample_size < 1:
            index = np.random.choice(imgs.shape[0], int(batch_size*fit_sample_size),
                                    replace=False)
            fit_drone_data_gen.fit(imgs[index])
        else:
            fit_drone_data_gen.fit(imgs[:])
    del drone_data_gen
    del batches
    print("[*] Done!")
    return fit_drone_data_gen.flow_from_directory(directory, max_samples,
                                                  target_size, color_mode,
                                                  batch_size, shuffle, seed,
                                                  follow_links, nb_windows)


class DroneDataGenerator(ImageDataGenerator):
    """
    Generate minibatches of images and labels with real-time augmentation.

    The only function that changes w.r.t. parent class is the flow that
    generates data. This function needed in fact adaptation for different
    directory structure and labels. All the remaining functions remain
    unchanged.

    For an example usage, see the evaluate.py script
    """
    def flow_from_directory(self, directory, max_samples, target_size=None,
            color_mode='grayscale', batch_size=32,
            shuffle=True, seed=None, follow_links=False, nb_windows=25):
        return DroneDirectoryIterator(
                directory, max_samples, self,
                target_size=target_size, color_mode=color_mode,
                batch_size=batch_size, shuffle=shuffle, seed=seed,
                follow_links=follow_links, nb_windows=nb_windows)


class DroneDirectoryIterator(Iterator):
    '''
    Class for managing data loading of images and labels.
    The assumed folder structure is:
        root_folder/
            dataset_1/
                images/
                    img_01.png
                    ...
                annotations.csv
            dataset_2/
                images/
                    img_01.png
                    ...
                annotations.csv
            ...

    # Arguments
       directory: Path to the root directory to read data from.
       image_data_generator: Image Generator.
       target_size: tuple of integers, dimensions to resize input images to.
       crop_size: tuple of integers, dimensions to crop input images.
       color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
       batch_size: The desired batch size
       shuffle: Whether to shuffle data or not
       seed : numpy seed to shuffle data
       follow_links: Bool, whether to follow symbolic links or not

    # TODO: Add functionality to save images to have a look at the augmentation
    '''
    def __init__(self, directory, max_samples, image_data_generator,
            target_size=None, color_mode='grayscale',
            batch_size=32, shuffle=True, seed=None, follow_links=False,
                 nb_windows=25):
        self.samples = 0
        self.max_samples = max_samples
        self.formats = {'png', 'jpg'}
        self.directory = directory
        self.image_data_generator = image_data_generator
        # self.target_size = tuple(target_size)
        self.nb_windows = nb_windows
        self.follow_links = follow_links
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        if self.color_mode == 'rgb':
            self.image_shape = target_size + (3,)
        else:
            self.image_shape = target_size + (1,)

        # Idea = associate each filename with a corresponding steering or label
        self.filenames = []
        self.ground_truth_loc = dict()
        self.gt_coord = dict()
        self.ground_truth_rot = []

        self._walk_dir(directory)

        # Conversion of list into array
        # self.ground_truth_loc = np.array(self.ground_truth_loc, dtype = K.floatx())
        self.ground_truth_rot = np.array(self.ground_truth_rot, dtype = K.floatx())

        assert self.samples > 0, "Empty dataset!"
        super(DroneDirectoryIterator, self).__init__(self.samples,
                batch_size, shuffle, seed)

    def _walk_dir(self, path):
        for root, dirs, files in os.walk(path):
            if "annotations.csv" in files:
                sub_dirs = os.path.relpath(root, path).split('/')
                sub_dirs = ''.join(sub_dirs)
                self._parse_dir(root, sub_dirs)

    def _parse_dir(self, path, sub_dirs):
        annotations_path = os.path.join(path, "annotations.csv")
        images_path = os.path.join(path, "images")
        rot_annotations = []
        with open(annotations_path, 'r') as annotations_file:
            annotations_file.readline() # Skip the header
            for line in annotations_file:
                line = line.split(',')
                frame_no = int(line[0].split('.')[0])
                key = "{}_{}".format(sub_dirs, frame_no)
                gate_center = [int(line[1]), int(line[2])]
                on_screen = (gate_center[0] >= 0 and gate_center[0] <=
                             self.image_shape[0]) and (gate_center[1] >= 0 and
                                                   gate_center[1] <=
                                                   self.image_shape[1])
                self.ground_truth_loc[key] =\
                    self._compute_location_labels(line[1:3], on_screen)
                    # self._compute_location_labels(line[1:3], bool(int(float(line[-1]))))
                self.gt_coord[key] = "{}x{}".format(line[1], line[2])
                rot_annotations.append(line[3])

        if len(self.ground_truth_loc) == 0 or len(rot_annotations) == 0:
            print("[!] Annotations could not be loaded!")
            raise Exception("Annotations not found")

        n = 0
        for filename in sorted(os.listdir(images_path)):
            if self.max_samples and n == self.max_samples:
                break
            is_valid = False
            for extension in self.formats:
                if filename.lower().endswith('.' + extension):
                    is_valid = True
                    break

            if is_valid:
                self.filenames.append(os.path.relpath(os.path.join(images_path,
                                                                   filename),
                                                      self.directory))
                self.samples += 1
                n += 1

    def _compute_location_labels(self, coordinates, visible):
        '''
        Computes the gate location window from the given pixel coordinates, and
        returns a list of binary labels corresponding to the N + 1 windows (+1
        because a special window is defined for the case where the gate is not
        visible).
        '''
        # TODO: Use keras.utils.to_categorical(y, num_classes=None, dtype='float32')
        # which does this automatically!
        sqrt_win = sqrt(self.nb_windows)
        windows_width = [int(i * self.image_shape[0] / sqrt_win)
                         for i in range(1, int(sqrt_win) + 1)]
        windows_height = [int(i * self.image_shape[1] / sqrt_win)
                         for i in range(1, int(sqrt_win) + 1)]
        i, j = 0, 0
        if not visible:
            labels = [0 for i in range(self.nb_windows + 1)]
            labels[0] = 1
            return labels

        for index, window_i in enumerate(windows_width):
            if int(float(coordinates[0])) < window_i:
                i = index + 1 # Start at 1
                break

        for index, window_h in enumerate(windows_height):
            if int(float(coordinates[1])) < window_h:
                j = index + 1 # Start at 1
                break

        labels = [0 for i in range(self.nb_windows + 1)]
        labels[int(i + ((j-1)*sqrt_win))] = 1
        return labels

    def next(self):
        """
        Public function to fetch next batch.

        # Returns
            The next batch of images and labels.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    # TODO: Batch orientation
    def _get_batches_of_transformed_samples(self, index_array) :
        current_batch_size = index_array.shape[0]
        # Image transformation is not under thread lock, so it can be done in
        # parallel
        batch_x = np.zeros((current_batch_size,) + (self.image_shape[1],
                                                    self.image_shape[0],
                                                    self.image_shape[2]),
                dtype=K.floatx())
        batch_localization = np.zeros((current_batch_size, self.nb_windows + 1,),
                dtype=K.floatx())
        batch_orientation = np.zeros((current_batch_size, 2,),
                dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'

        # Build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            x = img_utils.load_img(os.path.join(self.directory, fname),
                                   grayscale=grayscale)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

            # Build batch of localization and orientation data
            # Get rid of the filename and images/ folder
            sub_dirs_str = os.path.split(os.path.split(fname)[0])[0]
            sub_dirs_str = sub_dirs_str.replace('/', '')
            frame_no = int(os.path.split(fname)[-1].split('.')[0])
            key = "{}_{}".format(sub_dirs_str, frame_no)
            # batch_localization[i, 0] = 1.0
            if key in self.ground_truth_loc:
                batch_localization[i, :] = self.ground_truth_loc[key]
            else:
                batch_localization[i, 0] = 0
            batch_orientation[i, 0] = 0.0
            # batch_orientation[i, 1] = self.ground_truth_rot[fname]

        batch_y = batch_localization # TODO: add batch_orientation
        return batch_x, batch_y


def compute_predictions_and_gt(model, generator, steps,
                                     max_q_size=10,
                                     pickle_safe=False, verbose=0):
    """
    Generate predictions and associated ground truth
    for the input samples from a data generator.
    The generator should return the same kind of data as accepted by
    `predict_on_batch`.
    Function adapted from keras `predict_generator`.

    # Arguments
        generator: Generator yielding batches of input samples.
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
        max_q_size: Maximum size for the generator queue.
        pickle_safe: If `True`, use process based threading.
            Note that because
            this implementation relies on multiprocessing,
            you should not pass
            non picklable arguments to the generator
            as they can't be passed
            easily to children processes.
        verbose: verbosity mode, 0 or 1.

    # Returns
        Numpy array(s) of predictions and associated ground truth.

    # Raises
        ValueError: In case the generator yields
            data in an invalid format.
    """
    steps_done = 0
    all_outputs = []
    all_labels = []
    all_inputs = []

    if verbose == 1:
        progbar = Progbar(target=steps)

    while steps_done < steps:
        generator_output = next(generator)

        if isinstance(generator_output, tuple):
            if len(generator_output) == 2:
                x, gt_labels = generator_output
            elif len(generator_output) == 3:
                x, gt_labels, _ = generator_output
            else:
                raise ValueError('output of generator should be '
                                 'a tuple `(x, y, sample_weight)` '
                                 'or `(x, y)`. Found: ' +
                                 str(generator_output))
        else:
            raise ValueError('Output not valid for current evaluation')

        outputs = model.predict_on_batch(x)
        all_outputs += [output for output in outputs]
        all_labels += [label for label in gt_labels]
        all_inputs += [input for input in x]

        steps_done += 1

        if verbose == 1:
            progbar.update(steps_done)

    return all_inputs, all_outputs, all_labels

def compute_predictions(model, generator, steps,
                                     max_q_size=10,
                                     pickle_safe=False, verbose=0):
    """
    # Arguments
        generator: Generator yielding batches of input samples.
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
        max_q_size: Maximum size for the generator queue.
        pickle_safe: If `True`, use process based threading.
            Note that because
            this implementation relies on multiprocessing,
            you should not pass
            non picklable arguments to the generator
            as they can't be passed
            easily to children processes.
        verbose: verbosity mode, 0 or 1.

    # Returns
        Numpy array(s) of predictions and associated ground truth.

    # Raises
        ValueError: In case the generator yields
            data in an invalid format.
    """
    steps_done = 0
    all_outputs = []
    all_inputs = []

    if verbose == 1:
        progbar = Progbar(target=steps)

    while steps_done < steps:
        generator_output = next(generator)

        if isinstance(generator_output, tuple):
            if len(generator_output) == 2:
                x, gt_labels = generator_output
            elif len(generator_output) == 3:
                x, gt_labels, _ = generator_output
            else:
                raise ValueError('output of generator should be '
                                 'a tuple `(x, y, sample_weight)` '
                                 'or `(x, y)`. Found: ' +
                                 str(generator_output))
        else:
            raise ValueError('Output not valid for current evaluation')

        outputs = model.predict_on_batch(x)
        all_outputs += [output for output in outputs]
        all_inputs += [input for input in x]

        steps_done += 1

        if verbose == 1:
            progbar.update(steps_done)

    return all_inputs, all_outputs


def hard_mining_entropy(k, nb_windows):
    """
    Compute binary cross-entropy gate localization evaluation and hard-mining for the current batch.

    # Arguments
        k: Number of samples for hard-mining.

    # Returns
        custom_bin_crossentropy: average binary cross-entropy for the current batch.
    """

    def custom_bin_crossentropy(y_true, y_pred):
        # Parameter t indicates the type of experiment
        # t = y_true[:,0]

        # Number of gate loction samples
        # samples_loc = tf.cast(tf.equal(t,0), tf.int32)
        n_samples_loc = tf.reduce_sum(tf.cast(y_true, tf.int32))

        if n_samples_loc == 0:
            return 0.0
        else:
            # Predicted and real labels
            pred_loc = y_pred
            true_loc = y_true

            # gate loction loss
            l_loc = K.binary_crossentropy(true_loc, pred_loc)
            # Hard mining: use the K biggest losses
            k_min = tf.minimum(k, n_samples_loc) # Returns the minimum between k and n_samples_loc
            l_loc_sum = tf.reduce_sum(l_loc, 1)
            _, indices = tf.nn.top_k(l_loc_sum, k=k_min) # Find the k_min largest entries
            max_l_loc = tf.gather(l_loc, indices) # Match the indices with their values
            hard_l_loc = tf.divide(tf.reduce_sum(max_l_loc, 1), tf.cast(nb_windows, tf.float32))

            return hard_l_loc

    return custom_bin_crossentropy

def hard_mining_categorical_crossentropy(k, nb_windows):

    def custom_categorical_crossentropy(y_true, y_pred):
        # Parameter t indicates the type of experiment
        # t = y_true[:,0]

        # Number of gate loction samples
        # samples_loc = tf.cast(tf.equal(t,0), tf.int32)
        n_samples_loc = tf.reduce_sum(tf.cast(y_true, tf.int32))

        if n_samples_loc == 0:
            return 0.0
        else:
            # Predicted and real labels
            pred_loc = y_pred
            true_loc = y_true

            # gate loction loss
            l_loc = K.categorical_crossentropy(true_loc, pred_loc)
            # Hard mining: use the K biggest losses
            k_min = tf.minimum(k, n_samples_loc) # Returns the minimum between k and n_samples_loc
            # l_loc_sum = tf.reduce_sum(l_loc, 1)
            _, indices = tf.nn.top_k(l_loc, k=k_min) # Find the k_min largest entries
            max_l_loc = tf.gather(l_loc, indices) # Match the indices with their values
            # hard_l_loc = tf.divide(tf.reduce_sum(max_l_loc, 1), tf.cast(nb_windows, tf.float32))

            return max_l_loc

    return custom_categorical_crossentropy



def modelToJson(model, json_model_path):
    """
    Serialize model into json.
    """
    model_json = model.to_json()

    with open(json_model_path,"w") as f:
        f.write(model_json)


def jsonToModel(json_model_path):
    """
    Serialize json into model.
    """
    with open(json_model_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    return model

def write_to_file(dictionary, fname):
    """
    Writes everything is in a dictionary in json model.
    """
    with open(fname, "w") as f:
        json.dump(dictionary,f)
        print("Written file {}".format(fname))
