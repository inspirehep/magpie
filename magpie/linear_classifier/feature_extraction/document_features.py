def extract_document_features(inv_index, X):
    """
    Extract and add to the feature vector the property of the document itself.
    The same for all the candidates from the same document.
    :param inv_index: inverted index of a document
    :param X: feature matrix

    :return: None, works in place on X
    """
    for i in xrange(len(X['tf_mean'])):
        X['total_words_in_doc'][i] = inv_index.get_total_number_of_words()
        X['unique_words_in_doc'][i] = inv_index.get_number_of_unique_words()
