import logz
import numpy as np

import keras
from keras import backend as K



class MyCallback(keras.callbacks.Callback):
    """
    Customized callback class.
    
    # Arguments
       filepath: Path to save model.
       period: Frequency in epochs with which model is saved.
       batch_size: Number of images per batch.
    """
    
    def __init__(self, filepath, period, batch_size):
        self.filepath = filepath
        self.period = period
        self.batch_size = batch_size
        

    def on_epoch_begin(self, epoch, logs=None):
        
        # Decrease weight for binary cross-entropy loss
        sess = K.get_session()
        self.model.beta.load(np.maximum(0.0, 1.0-np.exp(-1.0/10.0*(epoch-10))), sess)
        self.model.alpha.load(1.0, sess)

        print(self.model.alpha.eval(sess))
        print(self.model.beta.eval(sess))


    def on_epoch_end(self, epoch, logs={}):
        
        # Save training and validation losses
        logz.log_tabular('train_loss', logs.get('loss'))
        logz.log_tabular('val_loss', logs.get('val_loss'))
        logz.dump_tabular()

        # Save model every 'period' epochs
        if (epoch+1) % self.period == 0:
            filename = self.filepath + '/model_weights_' + str(epoch) + '.h5'
            print("Saved model at {}".format(filename))
            self.model.save_weights(filename, overwrite=True)

        # Hard mining
        sess = K.get_session()
        mse_function = self.batch_size-(self.batch_size-10)*(np.maximum(0.0,1.0-np.exp(-1.0/30.0*(epoch-30.0))))
        entropy_function = self.batch_size-(self.batch_size-5)*(np.maximum(0.0,1.0-np.exp(-1.0/30.0*(epoch-30.0))))
        self.model.k_mse.load(int(np.round(mse_function)), sess)
        self.model.k_entropy.load(int(np.round(entropy_function)), sess)

