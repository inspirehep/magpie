from __future__ import division

import os

import numpy as np
import time
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.metrics import average_precision_score, mean_squared_error, log_loss

from magpie.config import HEP_TEST_PATH, HEP_TRAIN_PATH
from magpie.nn.config import BATCH_SIZE, NB_EPOCHS, LOG_FOLDER
from magpie.nn.input_data import batch_generator, get_train_and_test_data, \
    get_data_from
from magpie.nn.models import get_nn_model


def run_generator(nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE, nn='cnn',
                  nb_worker=1):
    train_batch_generator = batch_generator(HEP_TRAIN_PATH, batch_size)
    X_test, y_test = get_data_from(HEP_TEST_PATH)
    model = get_nn_model(nn)

    # Create callbacks
    logger = CustomLogger(X_test, y_test, nn)
    model_checkpoint = ModelCheckpoint(
        os.path.join(logger.log_dir, 'keras_model'),
        save_best_only=True,
    )

    history = model.fit_generator(
        train_batch_generator,
        len({filename[:-4] for filename in os.listdir(HEP_TRAIN_PATH)}),
        nb_epochs,
        show_accuracy=True,
        validation_data=(X_test, y_test),
        callbacks=[logger, model_checkpoint],
        nb_worker=nb_worker,
    )

    history.history['aps'] = logger.aps_list
    history.history['ll'] = logger.ll_list
    history.history['mse'] = logger.mse_list
    history.history['td'] = logger.td_list

    # Write acc and loss to file
    for metric in ['acc', 'loss']:
        with open(os.path.join(logger.log_dir, metric), 'wb') as f:
            for val in history.history[metric]:
                f.write(str(val) + "\n")

    return history, model


def run(nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE, nn='cnn'):
    (X_train, y_train), (X_test, y_test) = get_train_and_test_data()
    model = get_nn_model(nn)

    # Create callbacks
    logger = CustomLogger(X_test, y_test, nn)
    model_checkpoint = ModelCheckpoint(
        os.path.join(logger.log_dir, 'keras_model'),
        save_best_only=True,
    )

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        nb_epoch=nb_epochs,
        show_accuracy=True,
        validation_data=(X_test, y_test),
        callbacks=[logger, model_checkpoint],
    )

    history.history['aps'] = logger.aps_list
    history.history['ll'] = logger.ll_list
    history.history['mse'] = logger.mse_list
    history.history['td'] = logger.td_list

    # Write acc and loss to file
    for metric in ['acc', 'loss']:
        with open(os.path.join(logger.log_dir, metric), 'wb') as f:
            for val in history.history[metric]:
                f.write(str(val) + "\n")

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


# def run_in_minibatches(nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE, nn='cnn'):
#     model = get_nn_model(nn)
#
#     for epoch in xrange(nb_epochs):
#         train_batch_generator = get_batch_generator(train_dir, batch_size=batch_size)
#         test_batch_generator = get_batch_generator(test_dir, batch_size=batch_size)
#         train_loss, train_acc = [], []
#         for X, y in train_batch_generator:
#             loss, acc = model.train_on_batch(X, y, accuracy=True)
#             train_loss.append(loss)
#             train_acc.append(acc)
#         train_loss, train_acc = np.mean(train_loss), np.mean(train_acc)
#
#         test_loss, test_acc = [], []
#         for X, y in test_batch_generator:
#             loss, acc = model.test_on_batch(X, y, accuracy=True)
#             test_loss.append(loss)
#             test_acc.append(acc)
#         test_loss, test_acc = np.mean(test_loss), np.mean(test_acc)
#
#         X_test, y_test = self.test_data
#         y_pred = self.model.predict(X_test)
#
#         aps = average_precision_score(y_test, y_pred)
#         mse = mean_squared_error(y_test, y_pred)
#         ll = log_loss(y_test, y_pred)
#         td = compute_threshold_distance(y_test, y_pred)


def compare_results(X_test, y_test, model, i):
    """ Helper function for inspecting the results """
    if i == 0:
        y_pred = model.predict(X_test[:1])
    else:
        y_pred = model.predict(X_test[i - 1:i])
    sorted_indices = np.argsort(-y_pred[0])
    correct_indices = np.where(y_test[i])[0]
    return sorted_indices, correct_indices


def compute_threshold_distance(y_tests, y_preds):
    """
    Compute the threshold distance error between two output vectors.
    :param y_preds: matrix with predicted float vectors for each sample
    :param y_tests: matrix with ground truth output vectors for each sample

    :return: float with the score
    """
    assert len(y_tests) == len(y_preds)

    distances = []
    for i in xrange(len(y_preds)):
        y_pred, y_test = y_preds[i], y_tests[i]
        sorted_indices = np.argsort(-y_pred)
        correct_indices = np.where(y_test)[0]

        for i in correct_indices:
            position = np.where(sorted_indices == i)[0][0]
            distance = max(0, position - len(correct_indices) + 1)
            distances.append(distance)

    return np.mean(distances)


class CustomLogger(Callback):
    """
    A Keras callback logging additional metrics after every epoch
    """
    def __init__(self, X, y, nn_type, verbose=True):
        super(CustomLogger, self).__init__()
        self.test_data = (X, y)
        self.aps_list = []
        self.mse_list = []
        self.ll_list = []
        self.td_list = []
        self.verbose = verbose
        self.nn_type = nn_type
        self.log_dir = self.create_log_dir()

    def create_log_dir(self):
        """ Create a directory where all the logs would be stored  """
        dir_name = 'run_{}_{}'.format(self.nn_type, time.strftime('%d%m%H%M%S'))
        log_dir = os.path.join(LOG_FOLDER, dir_name)
        os.mkdir(log_dir)
        return log_dir

    def log_to_file(self, filename, value):
        """ Write a value to the file """
        with open(os.path.join(self.log_dir, filename), 'a') as f:
            f.write(str(value) + "\n")

    def on_train_begin(self, *args, **kwargs):
        """ Create a config file and write down the run parameters """
        with open(os.path.join(self.log_dir, 'config'), 'wb') as f:
            f.write("Model parameters:\n")
            f.write(str(self.params) + "\n\n")
            f.write("Model YAML:\n")
            f.write(self.model.to_yaml())

    def on_epoch_end(self, epoch, logs=None):
        """ Compute custom metrics at the end of the epoch """
        X_test, y_test = self.test_data
        y_pred = self.model.predict(X_test)

        aps = average_precision_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        ll = log_loss(y_test, y_pred)
        td = compute_threshold_distance(y_test, y_pred)
        val_acc = logs.get('val_acc', -1)
        val_loss = logs.get('val_loss', -1)

        self.aps_list.append(aps)
        self.mse_list.append(mse)
        self.ll_list.append(ll)
        self.td_list.append(td)

        log_dictionary = {
            'aps': aps,
            'mse': mse,
            'll': ll,
            'td': td,
            'val_acc': val_acc,
            'val_loss': val_loss
        }

        for metric_name, metric_value in log_dictionary.iteritems():
            self.log_to_file(metric_name, metric_value)

        if self.verbose:
            print('Average precision score: {}'.format(aps))
            print('MSE: {}'.format(mse))
            print('Threshold distance: {}'.format(td))
            print('Log loss: {}'.format(ll))
            print('')
