import os

import nltk

from nltk.tokenize import WordPunctTokenizer, sent_tokenize, word_tokenize
from magpie.misc.stopwords import STOPWORDS, PUNCTUATION

nltk.download('punkt', quiet=True)  # make sure it's downloaded before using


class Document(object):
    """ Class representing a document that the keywords are extracted from """
    def __init__(self, doc_id, filepath, text=None):
        self.doc_id = doc_id

        if text:
            self.text = text
            self.filename = None
            self.filepath = None
        else:  # is a path to a file
            if not os.path.exists(filepath):
                raise ValueError("The file " + filepath + " doesn't exist")

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
        """ Return all words tokenized, in lowercase and without punctuation """
        return [w.lower() for w in word_tokenize(self.text)
                if w not in PUNCTUATION]

    def get_meaningful_words(self):
        """ Return only non-stopwords, tokenized, in lowercase and without
        punctuation """
        return [w for w in self.get_all_words() if w not in STOPWORDS]

    def read_sentences(self):
        lines = self.text.split('\n')
        raw = [sentence for inner_list in lines
               for sentence in sent_tokenize(inner_list)]
        return [[w.lower() for w in word_tokenize(s) if w not in PUNCTUATION]
                for s in raw]
