from __future__ import division

import pandas as pd

from magpie.base.word2vec import compute_word2vec_for_phrase
from magpie.misc.stemmer import stem


def extract_keyword_features(kw_candidates, X, inv_index, model):
    """
    Extract and return a matrix with keyword features only.
    :param kw_candidates: an iterable containing KeywordTokens
    :param X: preallocated pandas matrix
    :param inv_index: InvertedIndex object for a given document
    :param model: LearningModel object

    :return: None, operates in place on the X matrix
    """
    for i in xrange(len(kw_candidates)):
        parsed_form = kw_candidates[i].get_parsed_form()
        keyphrase = tokenize_keyword(parsed_form)

        # TF & IDF
        tf_vec = inv_index.get_phrase_frequency(keyphrase)
        idf_vec = model.get_global_index().get_phrase_idf(keyphrase)

        X['tf_sum'][i] = sum(tf_vec)
        X['tf_mean'][i] = X['tf_sum'][i] / len(tf_vec)
        X['tf_max'][i] = max(tf_vec)
        X['tf_min'][i] = min(tf_vec)
        # X['tf_var'][i] = np.var(tf_vec)

        X['idf_sum'][i] = sum(idf_vec)
        X['idf_mean'][i] = X['idf_sum'][i] / len(idf_vec)
        X['idf_max'][i] = max(idf_vec)
        X['idf_min'][i] = min(idf_vec)
        # X['idf_var'][i] = np.var(idf_vec)

        X['tfidf'][i] = X['tf_mean'][i] * X['idf_mean'][i]

        # Occurrences and spread
        first_occurrence_vec = inv_index.get_first_phrase_occurrence(keyphrase)
        last_occurrence_vec = inv_index.get_last_phrase_occurrence(keyphrase)

        X['first_occurrence_mean'][i] = sum(first_occurrence_vec) / len(first_occurrence_vec)
        X['first_occurrence_min'][i] = min(first_occurrence_vec)
        X['first_occurrence_max'][i] = max(first_occurrence_vec)

        X['last_occurrence_mean'][i] = sum(last_occurrence_vec) / len(last_occurrence_vec)
        X['last_occurrence_min'][i] = min(last_occurrence_vec)
        X['last_occurrence_max'][i] = max(last_occurrence_vec)

        X['spread_means'][i] = max(0, X['last_occurrence_mean'][i] - X['first_occurrence_mean'][i])
        X['spread_minmax'][i] = max(0, X['last_occurrence_max'][i] - X['first_occurrence_min'][i])

        # Others
        X['no_of_words'][i] = len(keyphrase)
        X['no_of_letters'][i] = len(parsed_form)

        X['hops_from_anchor'][i] = kw_candidates[i].hops_from_anchor
        X['word2vec'][i] = compute_word2vec_for_phrase(parsed_form, model.word2vec)


def tokenize_keyword(kw_parsed):
    """
    Preprocess a keyword for feature computing. Split a parsed label into words
    and stem each one.
    :param kw_parsed: parsed form of a KeywordToken object

    :return: list of strings/unicodes
    """
    return [stem(w) for w in kw_parsed.split()]


def rebuild_feature_matrix(matrix):
    """
    Create a pandas object from the matrix dictionary and combine word2vec
    embeddings with other features
    :param matrix: dictionary of arrays, structure used for building X

    :return: pandas object with restructured information from matrix
    """
    w2v = matrix['word2vec']
    del matrix['word2vec']

    m1, m2 = pd.DataFrame(matrix), pd.DataFrame(w2v)
    return pd.concat([m1, m2], axis=1)
