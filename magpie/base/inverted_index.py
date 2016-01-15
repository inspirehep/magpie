from __future__ import division
from collections import defaultdict

from magpie.misc.stemmer import stem


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
        """ Get the number of unique words in the document """
        return len(self.index)

    def get_total_number_of_words(self):
        """ Get the total number of words in the document """
        return self.word_count

    def add_occurrence(self, word, position):
        """
        Add a word occurrence to the index
        :param word: unicode, stemmed word
        :param position: integer
        """
        self.index[word].append(position)

    def get_first_phrase_occurrence(self, phrase):
        """
        Get first occurrences for a list of words
        :param phrase: list of unicodes
        :return: list of floats
        """
        return [self.get_first_word_occurrence(w) for w in phrase]

    def get_first_word_occurrence(self, term):
        """
        Get a normalized first occurrence value for a given word
        :param term: unicode
        :return: float between 0 and 1
        """
        if term not in self.index:
            return 1
        else:
            return min(self.index[term]) / self.word_count

    def get_last_phrase_occurrence(self, phrase):
        """
        Get last occurrences for a list of words
        :param phrase: list of unicodes
        :return: list of floats
        """
        return [self.get_last_word_occurrence(w) for w in phrase]

    def get_last_word_occurrence(self, term):
        """
        Get a normalized last occurrence value for a given word
        :param term: unicode
        :return: float between 0 and 1
        """
        if term not in self.index:
            return 0
        else:
            return max(self.index[term]) / self.word_count

    def get_phrase_occurrences(self, phrase):
        """
        Get the number of occurrences in the document for a list of words
        :param phrase: list of unicodes
        :return: list of ints
        """
        return [self.get_word_occurrences(w) for w in phrase]

    def get_word_occurrences(self, word):
        """
        Get the number of occurrences of a given word in the document
        :param word: unicode
        :return: int
        """
        return len(self.index.get(word, []))

    def get_phrase_frequency(self, phrase):
        """
        Get the document frequencies for a list of words
        :param phrase: list of unicodes
        :return: list of floats between 0 and 1
        """
        return [self.get_word_frequency(w) for w in phrase]

    def get_word_frequency(self, word):
        """
        Get the word frequency in the document
        :param word: unicode
        :return: float
        """
        return self.get_word_occurrences(word) / self.word_count


