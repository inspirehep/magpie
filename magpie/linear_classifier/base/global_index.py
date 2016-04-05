from __future__ import division, unicode_literals

import collections
import math

import time

from magpie.misc.stemmer import stem
from magpie.utils import get_documents


def build_global_frequency_index(trainset_dir, verbose=True):
    """
    Build the GlobalFrequencyIndex object from the files in a given directory
    :param trainset_dir: path to the directory with files for training
    :return: GlobalFrequencyIndex object
    """
    tick = time.clock()

    global_index = GlobalFrequencyIndex()
    for doc in get_documents(trainset_dir):
        global_index.add_document(doc)

    if verbose:
        print("Global index built in : {0:.2f}s".format(time.clock() - tick))

    return global_index


class GlobalFrequencyIndex(object):
    """
    Holds the word count (bag of words) for the whole corpus.
    Enables to calculate IDF and word occurrences.
    """
    def __init__(self, docs=None):
        self.index = collections.defaultdict(set)
        self.total_docs = 0
        documents = docs or []
        for doc in documents:
            self.add_document(doc)
            self.total_docs += 1

    def add_document(self, doc):
        """
        Add the contents of a document to the index
        :param doc: Document object
        """
        for w in doc.get_meaningful_words():
            self.index[stem(w)].add(doc.doc_id)
        self.total_docs += 1

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
