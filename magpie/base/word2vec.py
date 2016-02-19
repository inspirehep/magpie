import os

import numpy as np
import time
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler

from magpie.base.document import Document
from magpie.config import HEP_TRAIN_PATH, WORD2VEC_MODELPATH, SCALER_PATH
from magpie.feature_extraction import EMBEDDING_SIZE
from magpie.misc.utils import save_to_disk
from magpie.utils import get_documents


def get_word2vec_model(model_path, train_dir, verbose=True):
    """
    Build or read from disk a word2vec trained model
    :param model_path: path to the trained model, None or 'retrain'
    :param train_dir: path to the training set

    :return: gensim Word2Vec object
    """
    tick = time.clock()

    if model_path:
        res = Word2Vec.load(model_path)
    else:
        res = out_of_core_train(train_dir)

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
    num_features = EMBEDDING_SIZE    # Word vector dimensionality
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


def out_of_core_x_normalisation(data_dir=HEP_TRAIN_PATH, batch_size=1024,
                                persist=False):
    """ Get all the word2vec vectors in a 2D matrix and fit the scaler on it.
     This scaler can be used afterwards for normalizing feature matrices. """
    doc_generator = get_documents(data_dir=data_dir)
    word2vec_model = Word2Vec.load(WORD2VEC_MODELPATH)
    scaler = StandardScaler(copy=False)

    no_more_samples = False
    while not no_more_samples:
        batch = []
        for i in xrange(batch_size):
            try:
                batch.append(doc_generator.next())
            except StopIteration:
                no_more_samples = True
                break

        vectors = []
        for doc in batch:
            for word in doc.get_all_words():
                if word in word2vec_model:
                    vectors.append(word2vec_model[word])

        matrix = np.array(vectors)
        print "Matrix shape: {}".format(matrix.shape)

        scaler.partial_fit(matrix)

    if persist:
        save_to_disk(SCALER_PATH, scaler)

    return scaler


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
    num_features = EMBEDDING_SIZE    # Word vector dimensionality
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
