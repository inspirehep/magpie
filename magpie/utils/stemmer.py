from nltk import PorterStemmer, SnowballStemmer, LancasterStemmer

STEMMER_TYPE = 'Porter'


def stem(word):
    stemmer = {
        'Porter': PorterStemmer(),
        'Snowball': SnowballStemmer('english'),
        'Lancaster': LancasterStemmer(),
    }[STEMMER_TYPE]

    return stemmer.stem(word)
