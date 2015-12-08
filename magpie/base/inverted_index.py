from __future__ import division
from collections import defaultdict
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

    def get_first_occurrence(self, word):
        stemmed = stem(word)
        if stemmed not in self.index:
            return None
        else:
            return min(self.index[stemmed])

    def get_term_occurrences(self, term):
        words = term.split()
        word_scores = [self._get_word_occurrences(w) for w in words]

        # TODO maybe a better function than sum would do here
        return sum(word_scores)

    def _get_word_occurrences(self, word):
        return len(self.index.get(stem(word), []))

    def get_term_frequency(self, term):
        words = term.split()
        word_scores = [self._get_word_frequency(w) for w in words]

        # TODO maybe a better function than sum would do here
        return sum(word_scores)

    def _get_word_frequency(self, word):
        term_count = len(self.index.get(stem(word), []))
        return term_count / self.word_count


