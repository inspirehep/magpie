from nltk import PorterStemmer, SnowballStemmer, LancasterStemmer

STEMMER_TYPE = 'Porter'


def stem(word):
    """ Use a chosen NLTK stemmer to stem a word. """
    stemmer = {
        'Porter': PorterStemmer(),
        'Snowball': SnowballStemmer('english'),
        'Lancaster': LancasterStemmer(),
    }[STEMMER_TYPE]

    return stemmer.stem(word)
