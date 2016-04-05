import numpy as np

from magpie.config import EMBEDDING_SIZE

FEATURE_VECTOR = {
    # Candidate features
    'tf_mean': 'float64',
    'tf_sum': 'float64',
    'tf_min': 'float64',
    'tf_max': 'float64',
    'idf_mean': 'float64',
    'idf_sum': 'float64',
    'idf_min': 'float64',
    'idf_max': 'float64',
    'tfidf': 'float64',
    'first_occurrence_mean': 'float64',
    'first_occurrence_min': 'float64',
    'first_occurrence_max': 'float64',
    'last_occurrence_mean': 'float64',
    'last_occurrence_min': 'float64',
    'last_occurrence_max': 'float64',
    'spread_means': 'float64',
    'spread_minmax': 'float64',
    'no_of_words': 'uint8',
    'no_of_letters': 'uint16',
    'hops_from_anchor': 'uint16',
    # 'word2vec': 'float64', # N dimensional

    # Document features
    'total_words_in_doc': 'uint32',
    'unique_words_in_doc': 'uint32',
}


def preallocate_feature_matrix(n_samples):
    """
    Create an empty feature matrix represented as a dictionary of arrays
    :param n_samples: number of samples/rows in the matrix

    :return: dictionary of numpy arrays
    """
    X = {k: np.zeros(n_samples, dtype=v)
         for k, v in FEATURE_VECTOR.iteritems()}
    X['word2vec'] = np.zeros((n_samples, EMBEDDING_SIZE), dtype='float32')

    return X
