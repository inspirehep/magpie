from gensim.models import Word2Vec
from theano.gradient import np

from magpie.config import HEP_TRAIN_PATH, HEP_TEST_PATH
from magpie.feature_extraction import WORD2VEC_LENGTH
from magpie.nn.config import OUTPUT_UNITS, SAMPLE_LENGTH, WORD2VEC_MODELPATH
from magpie.nn.considered_keywords import get_n_most_popular_keywords
from magpie.utils import get_documents, get_all_answers


def prepare_data(train_dir=HEP_TRAIN_PATH, test_dir=HEP_TEST_PATH):
    """
    Fetch and preprocess the training and testing data from given directories
    :param train_dir: directory with the training files
    :param test_dir: directory with the validation files

    :return: nested tuple with numpy matrices
    """
    x_train, y_train = build_and_sort(train_dir)
    x_test, y_test = build_and_sort(test_dir)

    return (x_train, y_train), (x_test, y_test)


def build_and_sort(data_dir):
    """
    Build X and Y matrices from a given directory. Make sure their order matches
    :param data_dir: directory with data files

    :return: tuple with two numpy (multidimensional) arrays
    """
    x = sorted(build_x(data_dir), key=lambda a: a[0])
    y = sorted(build_y(data_dir), key=lambda a: a[0])
    x_agg, y_agg = [], []
    for (fx, mx), (fy, my) in zip(x, y):
        if fx == fy:
            x_agg.append(mx)
            y_agg.append(my)
    return np.array(x_agg), np.array(y_agg)


def build_x(data_dir):
    """
    Build the X matrix from files from a given directory.
    Use Word2Vec for embedding
    :param data_dir: directory with files

    :return: list with 2D matrices for each sample. Each of them represents one
    abstract and has a shape of (SAMPLE_LENGTH, WORD2VEC_LENGTH)
    """
    word2vec_model = Word2Vec.load(WORD2VEC_MODELPATH)
    doc_tuples = []
    for doc in get_documents(data_dir=data_dir):

        words = doc.get_all_words()[:SAMPLE_LENGTH]
        matrix = np.zeros((SAMPLE_LENGTH, WORD2VEC_LENGTH), dtype=np.float32)

        for i, w in enumerate(words):
            if w in word2vec_model:
                matrix[i] = word2vec_model[w]

        doc_tuples.append((doc.filename[:-4], matrix))

    return doc_tuples


def build_y(data_dir):
    """
    Build the y matrix. It's a list of output vectors of length OUTPUT_UNITS
    :param data_dir: directory with files

    :return: list of bool numpy arrays with ones on specific indices
    """
    ans_dict = get_all_answers(data_dir)
    keywords = get_n_most_popular_keywords(OUTPUT_UNITS)
    keyword_indices = {kw: i for i, kw in enumerate(keywords)}

    for file_id, kw_set in ans_dict.items():
        ans_vector = np.zeros(OUTPUT_UNITS, dtype=np.bool_)
        for kw in kw_set:
            if kw in keyword_indices:
                index = keyword_indices[kw]
                ans_vector[index] = True
        ans_dict[file_id] = ans_vector

    return [(k, v) for k, v in ans_dict.iteritems()]
