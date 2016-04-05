from nltk import PorterStemmer, SnowballStemmer, LancasterStemmer

STEMMER_TYPE = 'Porter'


def stem(word):
    """ Use a chosen NLTK stemmer to stem a word. """
    return _stemmer.stem(word)


def _create_stemmer(stemmer_type):
    """ Initialize a stemmer """
    return {
        'Porter': PorterStemmer(),
        'Snowball': SnowballStemmer('english'),
        'Lancaster': LancasterStemmer(),
    }[stemmer_type]


_stemmer = _create_stemmer(STEMMER_TYPE)
