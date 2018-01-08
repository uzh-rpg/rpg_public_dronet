import gflags
import numpy as np
import os
import sys
import glob
from random import randint
from sklearn import metrics

from keras import backend as K

import utils
from constants import TEST_PHASE
from common_flags import FLAGS


# Functions to evaluate steering prediction

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y].
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary


def compute_explained_variance(predictions, real_values):
    """
    Computes the explained variance of prediction for each
    steering and the average of them
    """
    assert np.all(predictions.shape == real_values.shape)
    ex_variance = explained_variance_1d(predictions, real_values)
    print("EVA = {}".format(ex_variance))
    return ex_variance


def compute_sq_residuals(predictions, real_values):
    assert np.all(predictions.shape == real_values.shape)
    sq_res = np.square(predictions - real_values)
    sr = np.mean(sq_res, axis = -1)
    print("MSE = {}".format(sr))
    return sq_res


def compute_rmse(predictions, real_values):
    assert np.all(predictions.shape == real_values.shape)
    mse = np.mean(np.square(predictions - real_values))
    rmse = np.sqrt(mse)
    print("RMSE = {}".format(rmse))
    return rmse


def compute_highest_regression_errors(predictions, real_values, n_errors=20):
    """
    Compute the indexes with highest error
    """
    assert np.all(predictions.shape == real_values.shape)
    sq_res = np.square(predictions - real_values)
    highest_errors = sq_res.argsort()[-n_errors:][::-1]
    return highest_errors


def random_regression_baseline(real_values):
    mean = np.mean(real_values)
    std = np.std(real_values)
    return np.random.normal(loc=mean, scale=abs(std), size=real_values.shape)


def constant_baseline(real_values):
    mean = np.mean(real_values)
    return mean * np.ones_like(real_values)


def evaluate_regression(predictions, real_values, fname):
    evas = compute_explained_variance(predictions, real_values)
    rmse = compute_rmse(predictions, real_values)
    highest_errors = compute_highest_regression_errors(predictions, real_values,
            n_errors=20)
    dictionary = {"evas": evas.tolist(), "rmse": rmse.tolist(),
                  "highest_errors": highest_errors.tolist()}
    utils.write_to_file(dictionary, fname)


# Functions to evaluate collision

def read_training_labels(file_name):
    labels = []
    try:
        labels = np.loadtxt(file_name, usecols=0)
        labels = np.array(labels)
    except:
        print("File {} failed loading labels".format(file_name)) 
    return labels


def count_samples_per_class(train_dir):
    experiments = glob.glob(train_dir + "/*")
    num_class0 = 0
    num_class1 = 0
    for exp in experiments:
        file_name = os.path.join(exp, "labels.txt")
        try:
            labels = np.loadtxt(file_name, usecols=0)
            num_class1 += np.sum(labels == 1)
            num_class0 += np.sum(labels == 0)
        except:
            print("File {} failed loading labels".format(file_name)) 
            continue
    return np.array([num_class0, num_class1])


def random_classification_baseline(real_values):
    """
    Randomly assigns half of the labels to class 0, and the other half to class 1
    """
    return [randint(0,1) for p in range(real_values.shape[0])]


def weighted_baseline(real_values, samples_per_class):
    """
    Let x be the fraction of instances labeled as 0, and (1-x) the fraction of
    instances labeled as 1, a weighted classifier randomly assigns x% of the
    labels to class 0, and the remaining (1-x)% to class 1.
    """
    weights = samples_per_class/np.sum(samples_per_class)
    return np.random.choice(2, real_values.shape[0], p=weights)


def majority_class_baseline(real_values, samples_per_class):
    """
    Classify all test data as the most common label
    """
    major_class = np.argmax(samples_per_class)
    return [major_class for i in real_values]

            
def compute_highest_classification_errors(predictions, real_values, n_errors=20):
    """
    Compute the indexes with highest error
    """
    assert np.all(predictions.shape == real_values.shape)
    dist = abs(predictions - real_values)
    highest_errors = dist.argsort()[-n_errors:][::-1]
    return highest_errors


def evaluate_classification(pred_prob, pred_labels, real_labels, fname):
    ave_accuracy = metrics.accuracy_score(real_labels, pred_labels)
    print('Average accuracy = ', ave_accuracy)
    precision = metrics.precision_score(real_labels, pred_labels)
    print('Precision = ', precision)
    recall = metrics.precision_score(real_labels, pred_labels)
    print('Recall = ', recall)
    f_score = metrics.f1_score(real_labels, pred_labels)
    print('F1-score = ', f_score)
    highest_errors = compute_highest_classification_errors(pred_prob, real_labels,
            n_errors=20)
    dictionary = {"ave_accuracy": ave_accuracy.tolist(), "precision": precision.tolist(),
                  "recall": recall.tolist(), "f_score": f_score.tolist(),
                  "highest_errors": highest_errors.tolist()}
    utils.write_to_file(dictionary, fname)




def _main():

    # Set testing mode (dropout/batchnormalization)
    K.set_learning_phase(TEST_PHASE)

    # Generate testing data
    test_datagen = utils.DroneDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(FLAGS.test_dir,
                          shuffle=False,
                          color_mode=FLAGS.img_mode,
                          target_size=(FLAGS.img_width, FLAGS.img_height),
                          crop_size=(FLAGS.crop_img_height, FLAGS.crop_img_width),
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

    # Param t. t=1 steering, t=0 collision
    t_mask = t==1


    # ************************* Steering evaluation ***************************
    
    # Predicted and real steerings
    pred_steerings = predictions[t_mask,0]
    real_steerings = ground_truth[t_mask,0]

    # Compute random and constant baselines for steerings
    random_steerings = random_regression_baseline(real_steerings)
    constant_steerings = constant_baseline(real_steerings)

    # Create dictionary with filenames
    dict_fname = {'test_regression.json': pred_steerings,
                  'random_regression.json': random_steerings,
                  'constant_regression.json': constant_steerings}

    # Evaluate predictions: EVA, residuals, and highest errors
    for fname, pred in dict_fname.items():
        abs_fname = os.path.join(FLAGS.experiment_rootdir, fname)
        evaluate_regression(pred, real_steerings, abs_fname)

    # Write predicted and real steerings
    dict_test = {'pred_steerings': pred_steerings.tolist(),
                 'real_steerings': real_steerings.tolist()}
    utils.write_to_file(dict_test,os.path.join(FLAGS.experiment_rootdir,
                                               'predicted_and_real_steerings.json'))



    # *********************** Collision evaluation ****************************
    
    # Predicted probabilities and real labels
    pred_prob = predictions[~t_mask,1]
    pred_labels = np.zeros_like(pred_prob)
    pred_labels[pred_prob >= 0.5] = 1
               
    real_labels = ground_truth[~t_mask,1]

    # Compute random, weighted and majorirty-class baselines for collision
    random_labels = random_classification_baseline(real_labels)

    # Create dictionary with filenames
    dict_fname = {'test_classification.json': pred_labels,
                  'random_classification.json': random_labels}

    # Evaluate predictions: accuracy, precision, recall, F1-score, and highest errors
    for fname, pred in dict_fname.items():
        abs_fname = os.path.join(FLAGS.experiment_rootdir, fname)
        evaluate_classification(pred_prob, pred, real_labels, abs_fname)

    # Write predicted probabilities and real labels
    dict_test = {'pred_probabilities': pred_prob.tolist(),
                 'real_labels': real_labels.tolist()}
    utils.write_to_file(dict_test,os.path.join(FLAGS.experiment_rootdir,
                                               'predicted_and_real_labels.json'))


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
