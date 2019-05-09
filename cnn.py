import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import h5py
import gflags

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.metrics import categorical_accuracy, sparse_categorical_accuracy
from keras import optimizers
from time import time

import logz
import cnn_models
import utils
import log_utils
import keras.backend as K
from common_flags import FLAGS


def getModel(img_width, img_height, img_channels, output_dim, weights_path,
             transfer=False, transfer_from=None, skip_layers=3):
    """
    Initialize model.

    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.
       weights_path: Path to pre-trained model.

    # Returns
       model: A Model instance.
    """
    if FLAGS.restore_model:
        model = utils.jsonToModel(os.path.join(FLAGS.experiment_rootdir,
                                  "model_struct.json"))
    else:
        model = cnn_models.resnet8(img_width, img_height, img_channels, output_dim,
                                  FLAGS.freeze_filters)
    if weights_path:
        try:
            print("Loaded model from {}".format(weights_path))
            f = h5py.File(weights_path)
            model_layers = [layer.name for layer in model.layers]
            layer_dict = dict([(layer.name, layer) for layer in model.layers])
            if transfer and transfer_from is not None:
                print("Transfering weights from {}...".format(transfer_from))
                for i in layer_dict.keys():
                    if i in f:
                        weight_names = f[i].attrs["weight_names"]
                        weights = [f[i][j] for j in weight_names]
                        index = model_layers.index(i)
                        model.layers[index].set_weights(weights)
            else:
                model.load_weights(weights_path)
        except Exception as e:
            print(e)

    return model


def trainModel(train_data_generator, val_data_generator, model, initial_epoch):
    """
    Model training.

    # Arguments
       train_data_generator: Training data generated batch by batch.
       val_data_generator: Validation data generated batch by batch.
       model: Target image channels.
       initial_epoch: Dimension of model output.
    """

    # Initialize loss weights
    model.alpha = tf.Variable(1, trainable=False, name='alpha', dtype=tf.float32)
    # model.beta = tf.Variable(0, trainable=False, name='beta', dtype=tf.float32)

    # Initialize number of samples for hard-mining
    # model.k_mse = tf.Variable(FLAGS.batch_size, trainable=False, name='k_mse', dtype=tf.int32)
    model.k_entropy = tf.Variable(FLAGS.batch_size, trainable=False, name='k_entropy', dtype=tf.int32)

    # optimizer = optimizers.Adam(lr=0.00009, decay=1e-6)
    # optimizer = optimizers.Adam(lr=0.0000008, decay=1e-4)
    optimizer = optimizers.Adam(lr=0.0001, decay=1e-6)


    # Configure training process
    model.compile(loss=['categorical_crossentropy'],
                  loss_weights=[model.alpha],
                  optimizer=optimizer,
                  metrics=[categorical_accuracy])

    # Save model with the lowest validation loss
    weights_path = os.path.join(FLAGS.experiment_rootdir, 'weights_{epoch:03d}.h5')
    writeBestModel = ModelCheckpoint(filepath=weights_path, monitor='val_loss',
                                     save_best_only=True, save_weights_only=True)

    # Save model every 'log_rate' epochs.
    # Save training and validation losses.
    logz.configure_output_dir(FLAGS.experiment_rootdir)
    saveModelAndLoss = log_utils.MyCallback(filepath=FLAGS.experiment_rootdir,
                                            period=FLAGS.log_rate,
                                            batch_size=FLAGS.batch_size)

    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.9,
                                  patience=3, verbose=1, min_delta=0.05)
    # Train model
    steps_per_epoch = int(np.ceil(train_data_generator.samples / FLAGS.batch_size))
    validation_steps = int(np.ceil(val_data_generator.samples / FLAGS.batch_size))

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    model.fit_generator(train_data_generator,
                        epochs=FLAGS.epochs, steps_per_epoch = steps_per_epoch,
                        callbacks=[writeBestModel, saveModelAndLoss,
                                   tensorboard, reduce_lr],
                        shuffle=True,
                        validation_data=val_data_generator,
                        validation_steps = validation_steps,
                        initial_epoch=initial_epoch)


def _main():

    # Create the experiment rootdir if not already there
    if not os.path.exists(FLAGS.experiment_rootdir):
        os.makedirs(FLAGS.experiment_rootdir)

    # Input image dimensions
    img_width, img_height = FLAGS.img_width, FLAGS.img_height

    # Image mode
    if FLAGS.img_mode=='rgb':
        img_channels = 3
    elif FLAGS.img_mode == 'grayscale':
        img_channels = 1
    else:
        raise IOError("Unidentified image mode: use 'grayscale' or 'rgb'")

    # Output dimension
    output_dim = FLAGS.nb_windows + 1
    K.clear_session()
    # Generate training data with no real-time augmentation
    # train_datagen = utils.fit_flow_from_directory(rescale=1./255)
    train_datagen = utils.DroneDataGenerator(rescale=1./255,
                                             channel_shift_range=0.1,
                                            shading_factor=0.75,
                                            salt_and_pepper_factor=0.004)

    config = {
        'featurewise_center': True,
        'featurewise_std_normalization': True
    }
#     train_generator, train_mean, train_std = utils.fit_flow_from_directory(config, 1,
                                                    # FLAGS.train_dir,
                                                    # FLAGS.max_t_samples_per_dataset,
                                                    # shuffle=True,
                                                    # color_mode=FLAGS.img_mode,
                                                    # target_size=(img_width,
                                                                 # img_height),
                                                    # batch_size=FLAGS.batch_size,
#                                                     nb_windows=FLAGS.nb_windows)
    train_generator = train_datagen.flow_from_directory(FLAGS.train_dir,
                                                    FLAGS.max_t_samples_per_dataset,
                                                    shuffle=True,
                                                    color_mode=FLAGS.img_mode,
                                                    target_size=(img_width,
                                                                 img_height),
                                                    batch_size=FLAGS.batch_size,
                                                    nb_windows=FLAGS.nb_windows)

       # Generate validation data with no real-time augmentation
    val_datagen = utils.DroneDataGenerator(rescale=1./255)
    # val_datagen = utils.DroneDataGenerator()

    val_generator = val_datagen.flow_from_directory(FLAGS.val_dir,
                                                    FLAGS.max_v_samples_per_dataset,
                                                    shuffle = True,
                                                    color_mode=FLAGS.img_mode,
                                                    target_size=(img_width,
                                                                 img_height),
                                                    batch_size = FLAGS.batch_size,
                                                   nb_windows =
                                                    FLAGS.nb_windows,
                                                    mean=None,
                                                    std=None)

    if FLAGS.transfer_learning:
        # Model to transfer from
        model_path = os.path.join(FLAGS.model_transfer_fpath)
        # Weights to restore
        weights_path = os.path.join(FLAGS.weights_fpath)
    else:
        model_path = None
    initial_epoch = 0

    if not FLAGS.transfer_learning and not FLAGS.restore_model:
        # In this case weights will start from random
        weights_path = None
    elif FLAGS.restore_model:
        # In this case weigths will start from the specified model
        weights_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.weights_fname)
        initial_epoch = FLAGS.initial_epoch

    # Define model
    model = getModel(img_width, img_height, img_channels,
                        output_dim, weights_path,
                     transfer=FLAGS.transfer_learning,
                     transfer_from=model_path)

    # Serialize model into json
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    utils.modelToJson(model, json_model_path)

    # Train model
    trainModel(train_generator, val_generator, model, initial_epoch)


def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\n%s' % (sys.argv[0], FLAGS))

      sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)
