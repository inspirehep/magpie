import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def extract_keyword_features(kw_candidates, inv_index, global_freqs):
    """
    Extract and return a matrix with keyword features only.
    :param kw_candidates: an iterable containing KeywordTokens
    :param inv_index: InvertedIndex object for a given document
    :param global_freqs: GlobalFrequencyIndex object with a corpus word count

    :return: pandas DataFrame with keyword features
    """
    if not kw_candidates:
        return pd.DataFrame([])

    samples = []
    for kw in kw_candidates:
        feature_vector = build_feature_vector(kw, inv_index, global_freqs)
        samples.append((kw.get_parsed_form(), feature_vector))

    # TODO might be faster to convert dicts directly to the DataFrame
    dv = DictVectorizer()
    matrix = dv.fit_transform([s[1] for s in samples])

    df = pd.DataFrame(matrix.toarray(), columns=dv.get_feature_names())
    # df['kw'] = [s[0] for s in samples]
    return df


def build_feature_vector(kw, inv_index, global_freqs):
    """
    Build a feature vector for a given keyword
    :param kw: KeywordToken object
    :param inv_index: InvertedIndex object for a given document
    :param global_freqs: GlobalFrequencyIndex object with a corpus word count

    :return: dictionary with computed features
    """
    parsed_label = kw.get_parsed_form()

    # TF, IDF etc
    tf = inv_index.get_term_frequency(parsed_label)
    idf = global_freqs.get_term_idf(kw.get_parsed_form())

    return {
        # 'kw': parsed_label,
        'doc_occurrences': inv_index.get_term_occurrences(parsed_label),
        'corpus_occurences': global_freqs.get_term_occurrences(parsed_label),
        'tf': tf,
        'idf': idf,  # in how many docs they occur
        'tfidf': tf * idf,
        # count how close together in the occurrences are the kw terms
        # 'no_of_words': len(parsed_label.split()),
        # 'no_of_letters': len(parsed_label),
    }
