from __future__ import division

import os

import numpy as np
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.metrics import average_precision_score, mean_squared_error, log_loss

from magpie.nn.config import BATCH_SIZE, NB_EPOCHS
from magpie.nn.input_data import prepare_data
from magpie.nn.models import build_rnn_model, get_model_filename


class CustomLogger(Callback):
    """
    A Keras callback logging additional metrics after every epoch
    """
    def __init__(self, X, y, verbose=True):
        super(CustomLogger, self).__init__()
        self.test_data = (X, y)
        self.aps_list = []
        self.mse_list = []
        self.ll_list = []
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        """ Compute custom metrics at the end of the epoch """
        X_test, y_test = self.test_data
        y_pred = self.model.predict(X_test)

        aps = average_precision_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        ll = log_loss(y_test, y_pred)

        self.aps_list.append(aps)
        self.mse_list.append(mse)
        self.ll_list.append(ll)

        if self.verbose:
            print('Average precision score: {}'.format(aps))
            print('MSE: {}'.format(mse))
            print('Log loss: {}'.format(ll))
            print('')


def compare_results(X_test, y_test, model, i):
    """ Helper function for inspecting the results """
    if i == 0:
        y_pred = model.predict(X_test[:1])
    else:
        y_pred = model.predict(X_test[i - 1:i])
    sorted_indices = np.argsort(-y_pred[0])
    correct_indices = np.where(y_test[i])
    return sorted_indices, correct_indices


def main():
    (X_train, y_train), (X_test, y_test) = prepare_data()
    model = build_rnn_model()

    # Create callbacks
    logger = CustomLogger(X_test, y_test)
    model_filename = get_model_filename('rnn', len(X_train) + len(X_test))
    model_checkpoint = ModelCheckpoint(
        os.path.join(os.environ['HOME'], model_filename),
        save_best_only=True,
    )

    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        nb_epoch=NB_EPOCHS,
        show_accuracy=True,
        validation_data=(X_test, y_test),
        callbacks=[logger, model_checkpoint],
    )

    history.history['aps'] = logger.aps_list
    history.history['ll'] = logger.ll_list
    history.history['mse'] = logger.mse_list

    return history, model

    # accuracy = 1 - hamming_loss(y_test, y_pred)
    # print('Accuracy: {}'.format(accuracy))
    #
    # recall = precision = f1 = 0
    # for i in xrange(samples):
    #     recall += recall_score(y_test[i], y_pred[i])
    #     precision += precision_score(y_test[i], y_pred[i])
    #     f1 += f1_score(y_test[i], y_pred[i])
    #
    # print('Recall: {}'.format(recall / samples))
    # print('Precision: {}'.format(precision / samples))
    # print('F1: {}'.format(f1 / samples))
