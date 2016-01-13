import numpy as np

FEATURE_VECTOR = {
    # Candidate features
    'tf': 'float64',
    'idf': 'float64',
    'tfidf': 'float64',
    'first_occurrence': 'float64',
    'last_occurrence': 'float64',
    'spread': 'float64',
    'no_of_words': 'uint8',
    'no_of_letters': 'uint16',
    'hops_from_anchor': 'uint16',
    # 'word2vec': 'float64', # N dimensional

    # Document features
    'total_words_in_doc': 'uint32',
    'unique_words_in_doc': 'uint32',
}

WORD2VEC_LENGTH = 100


def preallocate_feature_matrix(n_samples):
    """
    Create an empty feature matrix represented as a dictionary of arrays
    :param n_samples: number of samples/rows in the matrix

    :return: dictionary of numpy arrays
    """
    X = {k: np.zeros(n_samples, dtype=v)
         for k, v in FEATURE_VECTOR.iteritems()}
    X['word2vec'] = np.zeros((n_samples, WORD2VEC_LENGTH), dtype='float32')

    return X
