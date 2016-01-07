import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def extract_keyword_features(kw_candidates, X, inv_index, global_freqs):
    """
    Extract and return a matrix with keyword features only.
    :param kw_candidates: an iterable containing KeywordTokens
    :param X: preallocated pandas matrix
    :param inv_index: InvertedIndex object for a given document
    :param global_freqs: GlobalFrequencyIndex object with a corpus word count

    :return: None, operates in place on the X matrix
    """
    for i in xrange(len(kw_candidates)):

        parsed_label = kw_candidates[i].get_parsed_form()

        # TF, IDF etc
        tf = inv_index.get_term_frequency(parsed_label)
        idf = global_freqs.get_term_idf(parsed_label)

        # Occurrences
        first_occurrence = inv_index.get_first_phrase_occurrence(parsed_label)
        last_occurrence = inv_index.get_last_phrase_occurrence(parsed_label)

        X['tf'][i] = tf
        X['idf'][i] = idf
        X['tfidf'][i] = tf * idf
        X['first_occurrence'][i] = first_occurrence
        X['last_occurrence'][i] = last_occurrence
        X['spread'][i] = max(0, last_occurrence - first_occurrence)
        X['no_of_words'][i] = len(parsed_label.split())
        X['no_of_letters'][i] = len(parsed_label)

