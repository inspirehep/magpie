from __future__ import print_function, unicode_literals
import os
import six
import numpy as np

from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler

from magpie.base.document import Document
from magpie.config import WORD2VEC_MODELPATH, EMBEDDING_SIZE
from magpie.utils import get_documents, save_to_disk


def train_word2vec_in_memory(docs, vec_dim=EMBEDDING_SIZE):
    """
    Builds word embeddings from documents and return a model
    :param docs: list of Document objects
    :param vec_dim: the dimensionality of the vector that's being built

    :return: trained gensim object with word embeddings
    """
    doc_sentences = map(lambda d: d.read_sentences(), docs)
    all_sentences = reduce(lambda d1, d2: d1 + d2, doc_sentences)

    # Set values for various parameters
    num_features = vec_dim  # Word vector dimensionality
    min_word_count = 5      # Minimum word count
    num_workers = 4         # Number of threads to run in parallel
    context = 5             # Context window size

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


def fit_scaler(data_dir, word2vec_model=WORD2VEC_MODELPATH, batch_size=1024,
               persist_to_path=None):
    """ Get all the word2vec vectors in a 2D matrix and fit the scaler on it.
     This scaler can be used afterwards for normalizing feature matrices. """
    if type(word2vec_model) == str:
        word2vec_model = Word2Vec.load(word2vec_model)

    doc_generator = get_documents(data_dir)
    scaler = StandardScaler(copy=False)

    no_more_samples = False
    while not no_more_samples:
        batch = []
        for i in range(batch_size):
            try:
                batch.append(six.next(doc_generator))
            except StopIteration:
                no_more_samples = True
                break

        vectors = []
        for doc in batch:
            for word in doc.get_all_words():
                if word in word2vec_model:
                    vectors.append(word2vec_model[word])

        matrix = np.array(vectors)
        print("Fitted to {} vectors".format(matrix.shape[0]))

        scaler.partial_fit(matrix)

    if persist_to_path:
        save_to_disk(persist_to_path, scaler)

    return scaler


def train_word2vec(doc_directory, vec_dim=EMBEDDING_SIZE):
    """
    Train the Word2Vec object iteratively, loading stuff to memory one by one.
    :param doc_directory: directory with the documents
    :param vec_dim: the dimensionality of the vector that's being built

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
    num_features = vec_dim  # Word vector dimensionality
    min_word_count = 5      # Minimum word count
    num_workers = 4         # Number of threads to run in parallel
    context = 5             # Context window size

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
