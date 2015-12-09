from nltk import word_tokenize

from magpie.utils.stemmer import stem


class GlobalFrequencyIndex(object):
    """
    Holds the word count (bag of words) for the whole corpus.
    Enables to calculate IDF and word occurences.
    """
    def __init__(self, documents):
        from sklearn.feature_extraction.text import CountVectorizer,\
            TfidfTransformer

        contents = [d.text for d in documents]

        # Create a custom tokenizing function
        def tokenizer(doc):
            return [stem(w.lower()) for w in word_tokenize(doc)]

        self.vectorizer = CountVectorizer(tokenizer=tokenizer)
        self.X = self.vectorizer.fit_transform(contents)

        self.transformer = TfidfTransformer()
        self.tfidf = self.transformer.fit_transform(self.X)

    def get_term_occurrences(self, term):
        words = term.split()
        scores = [self._get_word_occurrences(w) for w in words]

        # TODO another function could do here
        return sum(scores)

    def _get_word_occurrences(self, word):
        stemmed = stem(word)
        word_id = self.vectorizer.vocabulary_.get(stemmed)
        if word_id:
            return self.X[:, word_id].sum()
        else:
            return 0

    def get_term_idf(self, term):
        words = term.split()
        scores = [self._get_word_idf(w) for w in words]

        # TODO another function could do here
        return sum(scores)

    def _get_word_idf(self, word):
        stemmed = stem(word)
        word_id = self.vectorizer.vocabulary_.get(stemmed)
        if word_id:
            return self.transformer.idf_[word_id]
        else:
            # This word is not in the index
            return 1
