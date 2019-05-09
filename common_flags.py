import gflags



FLAGS = gflags.FLAGS

# Input
gflags.DEFINE_integer('img_width', 340, 'Target Image Width')
gflags.DEFINE_integer('img_height', 255, 'Target Image Height')
gflags.DEFINE_string('img_mode', "grayscale", 'Load mode for images, either '
                     'rgb or grayscale')

# Training
gflags.DEFINE_integer('batch_size', 32, 'Batch size in training and evaluation')
gflags.DEFINE_integer('epochs', 100, 'Number of epochs for training')
gflags.DEFINE_integer('log_rate', 10, 'Logging rate for full model (epochs)')
gflags.DEFINE_integer('initial_epoch', 0, 'Initial epoch to start training')
gflags.DEFINE_integer('max_t_samples_per_dataset', None, 'Maximum amount of'
                      ' training samples per individual dataset (subfolders inside the'
                      'root dataset dir)')
gflags.DEFINE_integer('max_v_samples_per_dataset', None, 'Maximum amount of'
                      ' validation samples per individual dataset (subfolders inside the'
                      'root dataset dir)')
gflags.DEFINE_integer('nb_visualizations', None, 'Amount of graphically annotated'
                      'images to export (evaluation)')
gflags.DEFINE_bool('freeze_filters', False, 'Wether to freeze the convolution'
                   ' filters during training')

# Files
gflags.DEFINE_string('experiment_rootdir', "./model", 'Folder '
                     ' containing all the logs, model weights and results')
gflags.DEFINE_string('train_dir', "/theos_dataset/training", 'Folder containing'
                     ' training experiments')
gflags.DEFINE_string('val_dir', "/theos_dataset/validation", 'Folder containing'
                     ' validation experiments')
gflags.DEFINE_string('test_dir', "./testing", 'Folder containing'
                     ' testing experiments')

# Model
gflags.DEFINE_bool('restore_model', False, 'Whether to restore a trained'
                   ' model for training')
gflags.DEFINE_string('weights_fname', "model_weights.h5", '(Relative) '
                                          'filename of model weights')
gflags.DEFINE_string('weights_fpath', "model_weights.h5", '(Absolute) '
                                          'filename of model weights to transfer from')
gflags.DEFINE_bool('transfer_learning', False, 'Partially restore a trained'
                   'model\'s weights')
gflags.DEFINE_string('model_transfer_fname', "model_struct.json", '(Relative) '
                                          'filename of model structure to use')
gflags.DEFINE_string('model_transfer_fpath', "model_struct.json", '(Absolute) '
                                          'filename of model structure to transfer from')
gflags.DEFINE_string('json_model_fname', "model_struct.json",
                          'Model struct json serialization, filename')

gflags.DEFINE_integer('nb_windows', 25, 'Number of regions to segmentate the'
                     ' gate location on the image')

# Testing / Visualizing
gflags.DEFINE_integer('successive_frames', 4, 'number of successive frames to'
                      ' use for the prediction filter (backward and forward)')
gflags.DEFINE_integer('max_outliers', 3, 'number of successive frames to'
                      ' use for the prediction filter')
