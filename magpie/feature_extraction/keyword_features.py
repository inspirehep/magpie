import pandas as pd

from magpie.base.word2vec import compute_word2vec_for_phrase
from magpie.utils.stemmer import stem


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

        # TF, IDF etc
        tf = inv_index.get_term_frequency(keyphrase)
        idf = model.get_global_index().get_term_idf(keyphrase)

        # Occurrences
        first_occurrence = inv_index.get_first_phrase_occurrence(keyphrase)
        last_occurrence = inv_index.get_last_phrase_occurrence(keyphrase)

        X['tf'][i] = tf
        X['idf'][i] = idf
        X['tfidf'][i] = tf * idf
        X['first_occurrence'][i] = first_occurrence
        X['last_occurrence'][i] = last_occurrence
        X['spread'][i] = max(0, last_occurrence - first_occurrence)
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
