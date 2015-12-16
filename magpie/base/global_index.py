import collections
import math

from magpie.utils.stemmer import stem


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

    def get_term_idf(self, term):
        words = term.split()
        scores = [self._get_word_idf(w) for w in words]

        # TODO another function could do here
        return sum(scores)

    def _get_word_idf(self, word):
        stemmed = stem(word)
        return math.log(self.total_docs / (1 + len(self.index[stemmed])))
        # word_id = self.vectorizer.vocabulary_.get(stemmed)
        # if word_id:
        #     return self.transformer.idf_[word_id]
        # else:
        #     # This word is not in the index
        #     return 1
