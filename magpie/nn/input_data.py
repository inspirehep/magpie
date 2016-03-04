import os
import threading

import numpy as np
from gensim.models import Word2Vec

from magpie.base.document import Document
from magpie.config import HEP_TRAIN_PATH, HEP_TEST_PATH, BATCH_SIZE, \
    WORD2VEC_MODELPATH, NO_OF_LABELS, EMBEDDING_SIZE
from magpie.misc.labels import get_labels
from magpie.nn.config import SAMPLE_LENGTH
from magpie.utils import get_answers_for_doc, get_scaler


def get_data_for_model(
        nn_model,
        as_generator=False,
        batch_size=BATCH_SIZE,
        train_dir=HEP_TRAIN_PATH,
        test_dir=HEP_TEST_PATH,
):
    """
    Get data in the form of matrices or generators for both train and test sets.
    :param nn_model: Keras model of the NN
    :param as_generator: flag whether to return a generator or in-memory matrix
    :param batch_size: integer, size of the batch
    :param train_dir: directory with train files
    :param test_dir: directory with test files

    :return: tuple with 2 elements for train and test data. Each element can be
    either a pair of matrices (X, y) or their generator
    """
    kwargs = dict(
        label_indices={lab: i for i, lab in enumerate(get_labels())},
        word2vec_model=Word2Vec.load(WORD2VEC_MODELPATH),
        scaler=get_scaler(),
        nn_model=nn_model,
    )

    if as_generator:
        filename_it = FilenameIterator(train_dir, batch_size)
        train_data = iterate_over_batches(filename_it, **kwargs)
    else:
        train_files = {filename[:-4] for filename in os.listdir(train_dir)}
        train_data = build_x_and_y(train_files, train_dir, **kwargs)

    test_files = {filename[:-4] for filename in os.listdir(test_dir)}
    x_test, y_test = build_x_and_y(test_files, test_dir, **kwargs)

    return train_data, (x_test, y_test)


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

    x_matrix = np.zeros((len(filenames), SAMPLE_LENGTH, EMBEDDING_SIZE))
    y_matrix = np.zeros((len(filenames), NO_OF_LABELS), dtype=np.bool_)

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

    if type(nn_model.input) == list:
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
            batch = self.files[self.i:self.i + self.batch_size]
            if len(batch) < self.batch_size:
                self.i = self.batch_size - len(batch)
                batch += self.files[:self.i]
            else:
                self.i += self.batch_size

            return batch
