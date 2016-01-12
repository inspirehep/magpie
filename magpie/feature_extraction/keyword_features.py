from magpie.utils.stemmer import stem


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
        keyphrase = tokenize_keyword(kw_candidates[i])

        # TF, IDF etc
        tf = inv_index.get_term_frequency(keyphrase)
        idf = global_freqs.get_term_idf(keyphrase)

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
        X['no_of_letters'][i] = len(kw_candidates[i].get_parsed_form())
        X['hops_from_anchor'][i] = kw_candidates[i].hops_from_anchor


def tokenize_keyword(kw_token):
    """
    Preprocess a keyword token for feature computing. Extract a parsed label,
    split into words and stem each one.
    :param kw_token: KeywordToken object

    :return: list of strings/unicodes
    """
    words = kw_token.get_parsed_form().split()
    return [stem(w) for w in words]
