import os
import threading

from gensim.models import Word2Vec
import numpy as np

from magpie.base.document import Document
from magpie.config import HEP_TRAIN_PATH, HEP_TEST_PATH, BATCH_SIZE, \
    WORD2VEC_MODELPATH
from magpie.feature_extraction import WORD2VEC_LENGTH
from magpie.nn.config import OUTPUT_UNITS, SAMPLE_LENGTH
from magpie.nn.considered_keywords import get_n_most_popular_keywords
from magpie.utils import get_documents, get_all_answers, get_answers_for_doc


def get_train_and_test_data(train_dir=HEP_TRAIN_PATH, test_dir=HEP_TEST_PATH):
    """
    Fetch and preprocess the training and testing data from given directories
    :param train_dir: directory with the training files
    :param test_dir: directory with the validation files

    :return: nested tuple with numpy matrices
    """
    x_train, y_train = get_data_from(train_dir)
    x_test, y_test = get_data_from(test_dir)

    return (x_train, y_train), (x_test, y_test)


def get_data_from(data_dir):
    """
    Build X and Y matrices from a given directory. Make sure their order matches
    :param data_dir: directory with data files

    :return: tuple with two numpy (multidimensional) arrays
    """
    x = sorted(build_x(data_dir), key=lambda a: a[0])
    y = sorted(build_y(data_dir), key=lambda a: a[0])
    x_agg, y_agg = [], []
    for (fx, mx), (fy, my) in zip(x, y):
        if fx == fy:
            x_agg.append(mx)
            y_agg.append(my)
    return np.array(x_agg), np.array(y_agg)


def batch_generator(dirname, batch_size=BATCH_SIZE):
    """
    NOT THREADSAFE.
    Generator producing batches of a fixed size from a directory
    and looping forever. Can be used if data doesn't fit in memory.
    :param dirname: directory with input files
    :param batch_size: size of the batch

    :return: generator yielding numpy arrays
    """
    word2vec_model = Word2Vec.load(WORD2VEC_MODELPATH)
    keywords = get_n_most_popular_keywords(OUTPUT_UNITS)
    keyword_indices = {kw: i for i, kw in enumerate(keywords)}
    docs = get_documents()

    while True:
        x_matrix = np.zeros((batch_size, SAMPLE_LENGTH, WORD2VEC_LENGTH))
        y_matrix = np.zeros((batch_size, OUTPUT_UNITS), dtype=np.bool_)
        for sample in xrange(batch_size):
            try:
                doc = docs.next()
            except StopIteration:
                docs = get_documents()
                doc = docs.next()

            words = doc.get_all_words()[:SAMPLE_LENGTH]

            for i, w in enumerate(words):
                if w in word2vec_model:
                    x_matrix[sample][i] = word2vec_model[w]

            answers = get_answers_for_doc(doc.filename, dirname)
            for kw in answers:
                if kw in keyword_indices:
                    index = keyword_indices[kw]
                    y_matrix[sample][index] = True

        yield [x_matrix], y_matrix


def iterate_over_batches(filename_it):
    """
    In theory threadsafe. Yield data batches using a threadsafe BatchIterator
    :param filename_it: BatchIterator object

    :return: generator yielding numpy arrays
    """
    word2vec_model = Word2Vec.load(WORD2VEC_MODELPATH)
    dirname = filename_it.dirname

    keywords = get_n_most_popular_keywords(OUTPUT_UNITS)
    keyword_indices = {kw: i for i, kw in enumerate(keywords)}

    while True:
        filenames = filename_it.next()
        x_matrix = np.zeros((len(filenames), SAMPLE_LENGTH, WORD2VEC_LENGTH))
        y_matrix = np.zeros((len(filenames), OUTPUT_UNITS), dtype=np.bool_)
        for doc_id, fname in enumerate(filenames):
            doc = Document(doc_id, os.path.join(dirname, fname + '.txt'))
            words = doc.get_all_words()[:SAMPLE_LENGTH]

            for i, w in enumerate(words):
                if w in word2vec_model:
                    x_matrix[doc_id][i] = word2vec_model[w]

            answers = get_answers_for_doc(fname + '.txt', dirname)
            for kw in answers:
                if kw in keyword_indices:
                    index = keyword_indices[kw]
                    y_matrix[doc_id][index] = True

        yield [x_matrix], y_matrix


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


def build_x(data_dir):
    """
    Build the X matrix from files from a given directory.
    Use Word2Vec for embedding
    :param data_dir: directory with files

    :return: list with 2D matrices for each sample. Each of them represents one
    abstract and has a shape of (SAMPLE_LENGTH, WORD2VEC_LENGTH)
    """
    word2vec_model = Word2Vec.load(WORD2VEC_MODELPATH)
    doc_tuples = []
    for doc in get_documents(data_dir=data_dir):

        words = doc.get_all_words()[:SAMPLE_LENGTH]
        matrix = np.zeros((SAMPLE_LENGTH, WORD2VEC_LENGTH))

        for i, w in enumerate(words):
            if w in word2vec_model:
                matrix[i] = word2vec_model[w]

        doc_tuples.append((doc.filename[:-4], matrix))

    return doc_tuples


def build_y(data_dir):
    """
    Build the y matrix. It's a list of output vectors of length OUTPUT_UNITS
    :param data_dir: directory with files

    :return: list of bool numpy arrays with ones on specific indices
    """
    ans_dict = get_all_answers(data_dir)
    keywords = get_n_most_popular_keywords(OUTPUT_UNITS)
    keyword_indices = {kw: i for i, kw in enumerate(keywords)}

    for file_id, kw_set in ans_dict.items():
        ans_vector = np.zeros(OUTPUT_UNITS, dtype=np.bool_)
        for kw in kw_set:
            if kw in keyword_indices:
                index = keyword_indices[kw]
                ans_vector[index] = True
        ans_dict[file_id] = ans_vector

    return [(k, v) for k, v in ans_dict.iteritems()]
