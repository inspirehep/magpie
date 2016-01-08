import os

import nltk

from nltk.tokenize import WordPunctTokenizer, sent_tokenize, word_tokenize
from magpie.utils.stopwords import STOPWORDS, PUNCTUATION

nltk.download('punkt')  # make sure it's downloaded before using


class Document(object):
    """ Class representing a document that the keywords are extracted from """
    def __init__(self, doc_id, filepath):
        if not os.path.exists(filepath):
            raise ValueError("The file " + filepath + " doesn't exist")

        self.doc_id = doc_id
        self.filepath = filepath
        self.filename = os.path.basename(filepath)

        with open(filepath, 'r') as f:
            self.text = f.read().decode('utf-8')

        self.wordset = self.compute_wordset()

    def __str__(self):
        return self.text

    def compute_wordset(self):
        tokens = WordPunctTokenizer().tokenize(self.text)
        lowercase = map(unicode.lower, tokens)
        return set(lowercase) - {',', '.', '!', ';', ':', '-', '', None}

    def get_all_words(self):
        return word_tokenize(self.text)

    def get_meaningful_words(self):
        return [w.lower() for w in word_tokenize(self.text)
                if w.lower() not in STOPWORDS and w not in PUNCTUATION]

    def read_sentences(self):
        lines = self.text.split('\n')
        return [sentence for inner_list in lines
                for sentence in sent_tokenize(inner_list)]

