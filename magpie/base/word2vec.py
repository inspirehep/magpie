import os
from itertools import chain

import numpy as np
import time
from gensim.models import Word2Vec

from magpie.base.document import Document
from magpie.feature_extraction import WORD2VEC_LENGTH


def get_word2vec_model(model_path, train_dir, verbose=True):
    """
    Build or read from disk a word2vec trained model
    :param model_path: path to the trained model, None or 'retrain'
    :param train_dir: path to the training set

    :return: gensim Word2Vec object
    """
    tick = time.clock()

    if not model_path:
        res = out_of_core_train(train_dir)
    else:
        res = Word2Vec.load(model_path)

    if verbose:
        print("Getting word2vec model: {0:.2f}s".format(time.clock() - tick))

    return res


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


# def create_sentence_iterator(doc_directory):
#     """
#     Return an iterator over tokenized sentences from different files.
#     :param doc_directory: directory with the files
#     :return: iterator over sentences (lists of strings)
#     """
#     from magpie import api
#     doc_gen = api.get_documents(data_dir=doc_directory)
#     sentence_gen = (d.read_sentences() for d in doc_gen)
#     return chain.from_iterable(sentence_gen)


def out_of_core_train(doc_directory):
    """
    Train the Word2Vec object iteratively, loading stuff to memory one by one.
    :param doc_directory: directory with the documents

    :return: Word2Vec object
    """
    class SentenceIterator(object):
        def __init__(self, dirname):
            self.dirname = dirname

        def __iter__(self):
            files = {filename[:-4] for filename in os.listdir(self.dirname)}
            for doc_id, fname in enumerate(files):
                d = Document(doc_id, os.path.join(self.dirname, fname + '.txt'))
                for sentence in d.read_sentences():
                    yield sentence

    # Set values for various parameters
    num_features = WORD2VEC_LENGTH    # Word vector dimensionality
    min_word_count = 5   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 5           # Context window size

    # Initialize and train the model
    model = Word2Vec(
        SentenceIterator(doc_directory),
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context,
    )

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    return model
