from __future__ import division
from collections import defaultdict

import numpy as np

from magpie.utils.stemmer import stem


class InvertedIndex(object):
    """ Creates and keeps an inverted index for a given document. """
    def __init__(self, document):
        self.index = defaultdict(list)

        words = document.get_meaningful_words()
        self.word_count = len(words)
        self._build_index(words)

    def _build_index(self, words):
        for position, word in enumerate(words):
            stemmed_word = stem(word)
            self.add_occurrence(stemmed_word, position)

    def get_number_of_unique_words(self):
        return len(self.index)

    def get_total_number_of_words(self):
        return self.word_count

    def add_occurrence(self, word, position):
        self.index[word].append(position)

    def get_first_phrase_occurrence(self, keyphrase):
        # TODO maybe a better function would do here
        return np.mean([self._get_first_term_occurrence(term)
                        for term in keyphrase])

    def _get_first_term_occurrence(self, term):
        if term not in self.index:
            return 1
        else:
            return min(self.index[term]) / self.word_count

    def get_last_phrase_occurrence(self, keyphrase):
        # TODO maybe a better function would do here
        return np.mean([self._get_last_term_occurrence(term)
                        for term in keyphrase])

    def _get_last_term_occurrence(self, term):
        if term not in self.index:
            return 0
        else:
            return max(self.index[term]) / self.word_count

    def get_term_occurrences(self, term):
        words = term.split()
        word_scores = [self._get_word_occurrences(w) for w in words]

        # TODO maybe a better function than sum would do here
        return np.mean(word_scores)

    def _get_word_occurrences(self, word):
        return len(self.index.get(stem(word), []))

    def get_term_frequency(self, keyphrase):
        # TODO maybe a better function than sum would do here
        return np.mean([self._get_word_frequency(w) for w in keyphrase])

    def _get_word_frequency(self, word):
        term_count = len(self.index.get(word, []))
        return term_count / self.word_count


