import gflags



FLAGS = gflags.FLAGS

# Input
gflags.DEFINE_integer('img_width', 640, 'Target Image Width')
gflags.DEFINE_integer('img_height', 480, 'Target Image Height')

gflags.DEFINE_integer('crop_img_width', 200, 'Cropped image widht')
gflags.DEFINE_integer('crop_img_height', 200, 'Cropped image height')
gflags.DEFINE_bool('no_crop', True, 'Whether to crop the input image')

gflags.DEFINE_string('img_mode', "grayscale", 'Load mode for images, either '
                     'rgb or grayscale')

# Training
gflags.DEFINE_integer('batch_size', 32, 'Batch size in training and evaluation')
gflags.DEFINE_integer('epochs', 100, 'Number of epochs for training')
gflags.DEFINE_integer('log_rate', 10, 'Logging rate for full model (epochs)')
gflags.DEFINE_integer('initial_epoch', 0, 'Initial epoch to start training')

# Files
gflags.DEFINE_string('experiment_rootdir', "./model", 'Folder '
                     ' containing all the logs, model weights and results')
gflags.DEFINE_string('train_dir', "./training", 'Folder containing'
                     ' training experiments')
gflags.DEFINE_string('val_dir', "./validation", 'Folder containing'
                     ' validation experiments')
gflags.DEFINE_string('test_dir', "./testing", 'Folder containing'
                     ' testing experiments')

# Model
gflags.DEFINE_bool('restore_model', False, 'Whether to restore a trained'
                   ' model for training')
gflags.DEFINE_string('weights_fname', "model_weights.h5", '(Relative) '
                                          'filename of model weights')
gflags.DEFINE_string('model_transfer_fname', "model_struct.json", '(Relative) '
                                          'filename of model structure to transfer from')
gflags.DEFINE_string('json_model_fname', "model_struct.json",
                          'Model struct json serialization, filename')

gflags.DEFINE_integer('nb_windows', 25, 'Number of regions to segmentate the'
                     ' gate location on the image')
