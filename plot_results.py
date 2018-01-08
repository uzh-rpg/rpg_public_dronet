import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
import gflags
import itertools
from sklearn.metrics import confusion_matrix

from common_flags import FLAGS



def make_and_save_histograms(pred_steerings, real_steerings,
                             img_name = "histograms.png"):
    """
    Plot and save histograms from predicted steerings and real steerings.
    
    # Arguments
        pred_steerings: List of predicted steerings.
        real_steerings: List of real steerings.
        img_name: Name of the png file to save the figure.
    """
    pred_steerings = np.array(pred_steerings)
    real_steerings = np.array(real_steerings)
    max_h = np.maximum(np.max(pred_steerings), np.max(real_steerings))
    min_h = np.minimum(np.min(pred_steerings), np.min(real_steerings))
    bins = np.linspace(min_h, max_h, num=50)
    plt.hist(pred_steerings, bins=bins, alpha=0.5, label='Predicted', color='b')
    plt.hist(real_steerings, bins=bins, alpha=0.5, label='Real', color='r')
    #plt.title('Steering angle')
    plt.legend(fontsize=10)
    plt.savefig(img_name, bbox_inches='tight')
    
    
def plot_confusion_matrix(real_labels, pred_prob, classes,
                          normalize=False,
                          img_name="confusion.png"):
    """
    Plot and save confusion matrix computed from predicted and real labels.
    
        # Arguments
        real_labels: List of real labels.
        pred_prob: List of predicted probabilities.
        normalize: Boolean, whether to apply normalization.
        img_name: Name of the png file to save the figure.
    """
    real_labels = np.array(real_labels)
    
    # Binarize predicted probabilities
    pred_prob = np.array(pred_prob)
    pred_labels = np.zeros_like(pred_prob)
    pred_labels[pred_prob >= 0.5] = 1
    
    cm = confusion_matrix(real_labels, pred_labels)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    #plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation=90)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(img_name)
    
    
def _main():
    
    # Compute histograms from predicted and real steerings
    fname_steer = os.path.join(FLAGS.experiment_rootdir, 'predicted_and_real_steerings.json')
    with open(fname_steer,'r') as f1:
        dict_steerings = json.load(f1)
    make_and_save_histograms(dict_steerings['pred_steerings'], dict_steerings['real_steerings'],
                             os.path.join(FLAGS.experiment_rootdir, "histograms.png"))
    
    # Compute confusion matrix from predicted and real labels
    fname_labels = os.path.join(FLAGS.experiment_rootdir,'predicted_and_real_labels.json')
    with open(fname_labels,'r') as f2:
        dict_labels = json.load(f2)
    plot_confusion_matrix(dict_labels['real_labels'], dict_labels['pred_probabilities'],
                          ['no collision', 'collision'],
                          img_name=os.path.join(FLAGS.experiment_rootdir, "confusion.png"))


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