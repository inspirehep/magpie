from __future__ import division

import collections
import math

from magpie.misc.stemmer import stem


class GlobalFrequencyIndex(object):
    """
    Holds the word count (bag of words) for the whole corpus.
    Enables to calculate IDF and word occurrences.
    """
    def __init__(self, documents):
        self.index = collections.defaultdict(set)
        self.total_docs = len(documents)

        contents = [(d.doc_id, d.get_meaningful_words())
                    for d in documents]

        # Build the index
        for doc_id, words in contents:
            for w in words:
                self.index[stem(w)].add(doc_id)

    # def get_term_occurrences(self, term):
    #     words = term.split()
    #     scores = [self._get_word_occurrences(w) for w in words]
    #
    #     # TODO another function could do here
    #     return sum(scores)
    #
    # def _get_word_occurrences(self, word):
    #     stemmed = stem(word)
    #     word_id = self.vectorizer.vocabulary_.get(stemmed)
    #     if word_id:
    #         return self.X[:, word_id].sum()
    #     else:
    #         return 0

    def get_phrase_idf(self, phrase):
        """
        Compute idf values for a list of words
        :param phrase: list of unicodes
        :return: list of floats: idfs
        """
        return [self.get_word_idf(w) for w in phrase]

    def get_word_idf(self, word):
        """
        Compute idf for a given word
        :param word: unicode
        :return: float: idf value
        """
        return math.log(self.total_docs / (1 + len(self.index[word])))
        # word_id = self.vectorizer.vocabulary_.get(stemmed)
        # if word_id:
        #     return self.transformer.idf_[word_id]
        # else:
        #     # This word is not in the index
        #     return 1
