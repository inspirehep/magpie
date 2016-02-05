from __future__ import division

import os

import numpy as np
import time
from keras.callbacks import Callback, ModelCheckpoint
from magpie.config import HEP_TEST_PATH, HEP_TRAIN_PATH, NB_EPOCHS, BATCH_SIZE
from magpie.evaluation.rank_metrics import mean_reciprocal_rank, r_precision, \
    precision_at_k, ndcg_at_k, mean_average_precision
from magpie.nn.config import LOG_FOLDER
from magpie.nn.input_data import get_train_and_test_data, get_data_from,\
    FilenameIterator, iterate_over_batches
from magpie.nn.models import get_nn_model


def run_generator(nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE, nn='cnn',
                  nb_worker=1):
    filename_it = FilenameIterator(HEP_TRAIN_PATH, batch_size)
    train_batch_generator = iterate_over_batches(filename_it, nn=nn)
    X_test, y_test = get_data_from(HEP_TEST_PATH, nn=nn)
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

    history.history['map'] = logger.map_list
    history.history['ndcg'] = logger.ndcg_list
    history.history['mrr'] = logger.mrr_list
    history.history['r_prec'] = logger.r_prec_list
    history.history['precision@3'] = logger.p_at_3_list
    history.history['precision@5'] = logger.p_at_5_list

    # Write acc and loss to file
    for metric in ['acc', 'loss']:
        with open(os.path.join(logger.log_dir, metric), 'wb') as f:
            for val in history.history[metric]:
                f.write(str(val) + "\n")

    return history, model


def run(nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE, nn='cnn'):
    (X_train, y_train), (X_test, y_test) = get_train_and_test_data(nn=nn)
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

    history.history['map'] = logger.map_list
    history.history['ndcg'] = logger.ndcg_list
    history.history['mrr'] = logger.mrr_list
    history.history['r_prec'] = logger.r_prec_list
    history.history['precision@3'] = logger.p_at_3_list
    history.history['precision@5'] = logger.p_at_5_list

    # Write acc and loss to file
    for metric in ['acc', 'loss']:
        with open(os.path.join(logger.log_dir, metric), 'wb') as f:
            for val in history.history[metric]:
                f.write(str(val) + "\n")

    return history, model


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
        self.map_list = []
        self.ndcg_list = []
        self.mrr_list = []
        self.r_prec_list = []
        self.p_at_3_list = []
        self.p_at_5_list = []
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

        # map = average_precision_score(y_test, y_pred, average='samples')
        # auc = roc_auc_score(y_test, y_pred, average='samples')

        y_pred = np.fliplr(y_pred.argsort())
        for i in xrange(len(y_test)):
            y_pred[i] = y_test[i][y_pred[i]]

        map = mean_average_precision(y_pred)
        mrr = mean_reciprocal_rank(y_pred)
        ndcg = np.mean([ndcg_at_k(row, len(row)) for row in y_pred])
        r_prec = np.mean([r_precision(row) for row in y_pred])
        p_at_3 = np.mean([precision_at_k(row, 3) for row in y_pred])
        p_at_5 = np.mean([precision_at_k(row, 5) for row in y_pred])
        val_acc = logs.get('val_acc', -1)
        val_loss = logs.get('val_loss', -1)

        self.map_list.append(map)
        self.mrr_list.append(mrr)
        self.ndcg_list.append(ndcg)
        self.r_prec_list.append(r_prec)
        self.p_at_3_list.append(p_at_3)
        self.p_at_5_list.append(p_at_5)

        log_dictionary = {
            'map': map,
            'mrr': mrr,
            'ndcg': ndcg,
            'r_prec': r_prec,
            'precision@3': p_at_3,
            'precision@5': p_at_5,
            'val_acc': val_acc,
            'val_loss': val_loss
        }

        for metric_name, metric_value in log_dictionary.iteritems():
            self.log_to_file(metric_name, metric_value)

        if self.verbose:
            print('Mean Average Precision: {}'.format(map))
            print('NDCG: {}'.format(ndcg))
            print('Mean Reciprocal Rank: {}'.format(mrr))
            print('R Precision: {}'.format(r_prec))
            print('Precision@3: {}'.format(p_at_3))
            print('Precision@5: {}'.format(p_at_5))
            print('')
