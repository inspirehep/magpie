from __future__ import unicode_literals, division

import os
import threading

import numpy as np
from gensim.models import Word2Vec

from magpie.base.document import Document
from magpie.config import BATCH_SIZE, SAMPLE_LENGTH
from magpie.utils import get_answers_for_doc, load_from_disk


def get_data_for_model(train_dir, labels, test_dir=None, nn_model=None,
                       as_generator=False, batch_size=BATCH_SIZE,
                       word2vec_model=None, scaler=None):
    """
    Get data in the form of matrices or generators for both train and test sets.
    :param train_dir: directory with train files
    :param labels: an iterable of predefined labels (controlled vocabulary)
    :param test_dir: directory with test files
    :param nn_model: Keras model of the NN
    :param as_generator: flag whether to return a generator or in-memory matrix
    :param batch_size: integer, size of the batch
    :param word2vec_model: trained w2v gensim model
    :param scaler: scaling object for X matrix normalisation e.g. StandardScaler

    :return: tuple with 2 elements for train and test data. Each element can be
    either a pair of matrices (X, y) or their generator
    """

    kwargs = dict(
        label_indices={lab: i for i, lab in enumerate(labels)},
        word2vec_model=word2vec_model,
        scaler=scaler,
        nn_model=nn_model,
    )

    if as_generator:
        filename_it = FilenameIterator(train_dir, batch_size)
        train_data = iterate_over_batches(filename_it, **kwargs)
    else:
        train_files = {filename[:-4] for filename in os.listdir(train_dir)}
        train_data = build_x_and_y(train_files, train_dir, **kwargs)

    test_data = None
    if test_dir:
        test_files = {filename[:-4] for filename in os.listdir(test_dir)}
        test_data = build_x_and_y(test_files, test_dir, **kwargs)

    return train_data, test_data


def build_x_and_y(filenames, file_directory, **kwargs):
    """
    Given file names and their directory, build (X, y) data matrices
    :param filenames: iterable of strings showing file ids (no extension)
    :param file_directory: path to a directory where those files lie
    :param kwargs: additional necessary data for matrix building e.g. scaler

    :return: a tuple (X, y)
    """
    label_indices = kwargs['label_indices']
    word2vec_model = kwargs['word2vec_model']
    scaler = kwargs['scaler']
    nn_model = kwargs['nn_model']

    x_matrix = np.zeros((len(filenames), SAMPLE_LENGTH, word2vec_model.vector_size))
    y_matrix = np.zeros((len(filenames), len(label_indices)), dtype=np.bool_)

    for doc_id, fname in enumerate(filenames):
        doc = Document(doc_id, os.path.join(file_directory, fname + '.txt'))
        words = doc.get_all_words()[:SAMPLE_LENGTH]

        for i, w in enumerate(words):
            if w in word2vec_model:
                word_vector = word2vec_model[w].reshape(1, -1)
                x_matrix[doc_id][i] = scaler.transform(word_vector, copy=True)[0]

        labels = get_answers_for_doc(
            fname + '.txt',
            file_directory,
            filtered_by=set(label_indices.keys()),
        )

        for lab in labels:
            index = label_indices[lab]
            y_matrix[doc_id][index] = True

    if nn_model and type(nn_model.input) == list:
        return [x_matrix] * len(nn_model.input), y_matrix
    else:
        return [x_matrix], y_matrix


def iterate_over_batches(filename_it, **kwargs):
    """
    Iterate infinitely over a given filename iterator
    :param filename_it: FilenameIterator object
    :param kwargs: additional necessary data for matrix building e.g. scaler
    :return: yields tuples (X, y) when called
    """
    while True:
        files = filename_it.next()
        yield build_x_and_y(files, filename_it.dirname, **kwargs)


class FilenameIterator(object):
    """ A threadsafe iterator yielding a fixed number of filenames from a given
     folder and looping forever. Can be used for external memory training. """
    def __init__(self, dirname, batch_size):
        self.dirname = dirname
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.files = list({filename[:-4] for filename in os.listdir(dirname)})
        self.i = 0

    def __iter__(self):
        return self

    def next(self):
        with self.lock:

            if self.i == len(self.files):
                self.i = 0

            batch = self.files[self.i:self.i + self.batch_size]
            if len(batch) < self.batch_size:
                self.i = 0
            else:
                self.i += self.batch_size

            return batch
