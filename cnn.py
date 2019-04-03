import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import gflags

from keras.callbacks import ModelCheckpoint, TensorBoard
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
    model = cnn_models.resnet8(img_width, img_height, img_channels, output_dim)
    if weights_path:
        try:
            print("Loaded model from {}".format(weights_path))
            model_layers = [layer.name for layer in model.layers]
            if transfer and transfer_from is not None:
                print("Transfering weights from {} until layer 8...".format(transfer_from))
                original_model = utils.jsonToModel(transfer_from)
                weight_value_tuples = []
                start_at = 2 if img_channels == 3 else 0
                # Skip the last n layers
                for layer in original_model.layers[start_at:-skip_layers]:
                    print("-> Layer {}".format(layer.name))
                    if layer.name in model_layers:
                        target_layer = model.get_layer(name=layer.name)
                        print("--> Target layer: {}".format(target_layer.name))
                        symbolic_weights = target_layer.trainable_weights + target_layer.non_trainable_weights
                        weight_values = layer.get_weights()
                        weight_value_tuples += zip(symbolic_weights, weight_values)
                    else:
                        print("--> [x] No match in target model! Skipping...")

                # Apply to the target model
                K.batch_set_value(weight_value_tuples)
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

    optimizer = optimizers.Adam(lr=0.003, decay=1e-6)

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

    # Train model
    steps_per_epoch = int(np.ceil(train_data_generator.samples / FLAGS.batch_size))
    validation_steps = int(np.ceil(val_data_generator.samples / FLAGS.batch_size))

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    model.fit_generator(train_data_generator,
                        epochs=FLAGS.epochs, steps_per_epoch = steps_per_epoch,
                        callbacks=[writeBestModel, saveModelAndLoss,
                                   tensorboard],
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

    # Generate training data with no real-time augmentation
    train_datagen = utils.DroneDataGenerator(rescale=1./255)

    # Already shuffled ;)
    train_generator = train_datagen.flow_from_directory(FLAGS.train_dir,
                                                        FLAGS.max_t_samples_per_dataset,
                                                        shuffle = False,
                                                        color_mode=FLAGS.img_mode,
                                                        target_size=(img_width,
                                                                     img_height),
                                                        batch_size = FLAGS.batch_size,
                                                       nb_windows = FLAGS.nb_windows)

       # Generate validation data with no real-time augmentation
    val_datagen = utils.DroneDataGenerator(rescale=1./255)

    val_generator = val_datagen.flow_from_directory(FLAGS.val_dir,
                                                    FLAGS.max_v_samples_per_dataset,
                                                    shuffle = False,
                                                    color_mode=FLAGS.img_mode,
                                                    target_size=(img_width,
                                                                 img_height),
                                                    batch_size = FLAGS.batch_size,
                                                   nb_windows = FLAGS.nb_windows)

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
        weights_path = os.path.join(FLAGS.weights_fname)
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
