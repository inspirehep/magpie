import numpy as np
from gensim.models import Word2Vec

from magpie.feature_extraction import WORD2VEC_LENGTH


def train_word2vec(docs):
    """
    Builds word embeddings from documents and return a model
    :param docs: list of Document objects

    :return: trained gensim object with word embeddings
    """
    doc_sentences = map(lambda d: d.read_sentences(), docs)
    all_sentences = reduce(lambda d1, d2: d1 + d2, doc_sentences)

    # Set values for various parameters
    num_features = WORD2VEC_LENGTH    # Word vector dimensionality
    min_word_count = 5   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 5           # Context window size

    # Initialize and train the model
    model = Word2Vec(
        all_sentences,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context,
    )

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    return model


def compute_word2vec_for_phrase(phrase, model):
    """
    Compute (add) word embedding for a multiword phrase using a given model
    :param phrase: unicode, parsed label of a keyphrase
    :param model: gensim word2vec object

    :return: numpy array
    """
    result = np.zeros(model.vector_size, dtype='float32')
    for word in phrase.split():
        if word in model:
            result += model[word]

    return result
