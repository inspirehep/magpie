from __future__ import division

import os

import numpy as np
import time

from keras.callbacks import Callback, ModelCheckpoint
from magpie.config import HEP_TEST_PATH, HEP_TRAIN_PATH, NB_EPOCHS, BATCH_SIZE, \
    EMBEDDING_SIZE
from magpie.evaluation.rank_metrics import mean_reciprocal_rank, r_precision, \
    precision_at_k, ndcg_at_k, mean_average_precision
from magpie.misc.labels import get_labels
from magpie.nn.config import LOG_FOLDER, SAMPLE_LENGTH
from magpie.nn.input_data import get_data_for_model
from magpie.nn.models import get_nn_model


def batch_train(train_dir=HEP_TRAIN_PATH, test_dir=HEP_TEST_PATH,
                nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE, nn='berger_cnn',
                nb_worker=1, verbose=1):
    """
    Train a NN model out-of-core with given parameters.
    :param train_dir: path to the directory with training files
    :param test_dir: path to the directory with testing files
    :param nb_epochs: number of epochs
    :param batch_size: size of one batch
    :param nn: nn type, for supported ones look at `get_nn_model()`
    :param nb_worker: number of workers to read the data
    :param verbose: verbosity flag

    :return: tuple containing a history object and a trained keras model
    """
    model = get_nn_model(nn)
    train_generator, (x_test, y_test) = get_data_for_model(
        model,
        as_generator=True,
        batch_size=batch_size,
        train_dir=train_dir,
        test_dir=test_dir,
    )

    # Create callbacks
    logger = CustomLogger(x_test, y_test, nn)
    model_checkpoint = ModelCheckpoint(
        os.path.join(logger.log_dir, 'keras_model'),
        save_best_only=True,
    )

    history = model.fit_generator(
        train_generator,
        len({filename[:-4] for filename in os.listdir(train_dir)}),
        nb_epochs,
        show_accuracy=True,
        validation_data=(x_test, y_test),
        callbacks=[logger, model_checkpoint],
        nb_worker=nb_worker,
        verbose=verbose,
    )

    finish_logging(logger, history)

    return history, model


def train(train_dir=HEP_TRAIN_PATH, test_dir=HEP_TEST_PATH, nb_epochs=NB_EPOCHS,
          batch_size=BATCH_SIZE, nn='berger_cnn', verbose=1):
    """
    Train a NN model with given parameters, all in memory
    :param train_dir: path to the directory with training files
    :param test_dir: path to the directory with testing files
    :param nb_epochs: number of epochs
    :param batch_size: size of one batch
    :param nn: nn type, for supported ones look at `get_nn_model()`
    :param verbose: verbosity flag

    :return: tuple containing a history object and a trained keras model
    """
    model = get_nn_model(nn)
    (x_train, y_train), (x_test, y_test) = get_data_for_model(
        model,
        as_generator=False,
        train_dir=train_dir,
        test_dir=test_dir,
    )

    # Create callbacks
    logger = CustomLogger(x_test, y_test, nn)
    model_checkpoint = ModelCheckpoint(
        os.path.join(logger.log_dir, 'keras_model'),
        save_best_only=True,
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        nb_epoch=nb_epochs,
        show_accuracy=True,
        validation_data=(x_test, y_test),
        callbacks=[logger, model_checkpoint],
        verbose=verbose,
    )

    finish_logging(logger, history)

    return history, model


def extract(doc, model, **kwargs):
    """
    Use a given trained NN model to extract keywords from a document
    :param doc: Document object
    :param model: keras Model object
    :param kwargs: should contain elements: 'word2vec_model' and 'scaler'

    :return: list of tuples of the form [('kw1', 0.85), ('kw2', 0.6) ...]
    """
    word2vec_model = kwargs['word2vec_model']
    scaler = kwargs['scaler']

    words = doc.get_all_words()[:SAMPLE_LENGTH]
    x_matrix = np.zeros((1, SAMPLE_LENGTH, EMBEDDING_SIZE))

    for i, w in enumerate(words):
        if w in word2vec_model:
            word_vector = word2vec_model[w].reshape(1, -1)
            x_matrix[doc.doc_id][i] = scaler.transform(word_vector, copy=True)[0]

    x = [x_matrix] * len(model.input) if type(model.input) == list else [x_matrix]

    # Predict
    y_predicted = model.predict(x)

    zipped = zip(get_labels(), y_predicted[0])

    return sorted(zipped, key=lambda elem: elem[1], reverse=True)


def finish_logging(logger, history):
    """ Save the rest of the logs after finishing optimisation. """
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
        dir_name = '{}_{}'.format(self.nn_type, time.strftime('%d%m%H%M%S'))
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
        x_test, y_test = self.test_data
        y_pred = self.model.predict(x_test)

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
