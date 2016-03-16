import os
from collections import defaultdict

from flask import Flask, request, jsonify
from gensim.models import Word2Vec

from magpie.api import extract_from_text
from magpie.config import DATA_DIR
from magpie.nn.models import berger_cnn
from magpie.utils import get_scaler

app = Flask('magpie')

word2vec_models = defaultdict(dict)
scalers = defaultdict(dict)
nn_models = dict()

supported_corpora = ['hep-keywords', 'hep-categories']


def get_cached_model(corpus):
    """ Get the cached Keras NN model or rebuild it if missed. """
    global nn_models

    if corpus not in nn_models:
        nn_models[corpus] = build_model_for_corpus(corpus)

    return nn_models[corpus]


def get_cached_scaler(model, corpus):
    """ Get the cached NN scaler or reload it if missed. """
    global scalers

    input_layer = model.input[0] if type(model.input) == list else model.input
    _, _, embedding_dim = input_layer.get_shape()
    embedding_size = embedding_dim.value

    if embedding_size not in scalers[corpus]:
        scaler_path = os.path.join(DATA_DIR, corpus, 'scalers',
                                   'scaler_nn_{0}.pickle'.format(embedding_size))
        s = get_scaler(path=scaler_path)
        scalers[corpus][embedding_size] = s

    return scalers[corpus][embedding_size]


def get_cached_w2v_model(model, corpus):
    """ Get the cached word2vec model or reload it if missed. """
    global word2vec_models

    input_layer = model.input[0] if type(model.input) == list else model.input
    _, _, embedding_dim = input_layer.get_shape()
    embedding_size = embedding_dim.value

    if embedding_size not in word2vec_models[corpus]:
        w2v_path = os.path.join(DATA_DIR, corpus, 'w2v_models',
                                'word2vec{0}_model.gensim'.format(embedding_size))
        w = Word2Vec.load(w2v_path)
        word2vec_models[corpus][embedding_size] = w

    return word2vec_models[corpus][embedding_size]


def build_model_for_corpus(corpus):
    """ Build an appropriate Keras NN model depending on the corpus """
    model = None
    if corpus == 'hep-keywords':
        model = berger_cnn(embedding_size=100, output_length=1000)
    elif corpus == 'hep-categories':
        model = berger_cnn(embedding_size=50, output_length=14)

    path = os.path.join(DATA_DIR, corpus, 'berger_cnn.trained')
    model.load_weights(path)
    return model


@app.route('/')
@app.route('/hello')
def hello():
    """ Function for testing purposes """
    return 'Hello World!'


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    """
    Takes a following JSON as input:
    {
        'text': 'my abstract'       # the text to be fed to the model
        'corpus': 'hep-keywords'    # corpus to work on. Currently supported are
                                    # in the supported_corpora variable
    }

    :return:
    {
        'status_code': 200      # 200, 400, 403 etc
        'labels': []            # list of two-element tuples each with a label
                                # and its confidence value e.g. [('jan', 0.95)]
    }
    """
    if request.method == 'GET':
        return "GET method is not supported for this URI, use POST"

    json = request.json
    if not json or 'text' not in json:
        return jsonify({'status_code': 400, 'keywords': []})

    corpus = json.get('corpus', supported_corpora[0])
    if corpus not in supported_corpora:
        return jsonify({'status_code': 404, 'keywords': [],
                        'info': 'Corpus ' + corpus + ' is not available'})

    model = get_cached_model(corpus)
    scaler = get_cached_scaler(model, corpus)
    word2vec_model = get_cached_w2v_model(model, corpus)

    kwargs = dict(word2vec_model=word2vec_model, scaler=scaler)
    labels = extract_from_text(json['text'], model, **kwargs)

    return jsonify({
        'status_code': 200,
        'labels': labels,
    })


@app.route("/word2vec", methods=['GET', 'POST'])
def word2vec():
    """
    Takes a following JSON as input:
    {
        'corpus': 'hep-keywords'        # corpus to work on. Currently supported
                                        # are in the supported_corpora variable
        'positive': ['cern', 'geneva']  # words to add
        'negative': ['heidelberg']      # words to subtract

        'vector_size': 100              # (optional) size of the vector
    }

    :return:
    {
        'status_code': 200      # 200, 400, 403 etc
        'similar_words': []     # list of the form [('w1', 0.99), ('w2', 0.67)]
    }
    """
    if request.method == 'GET':
        return "GET method is not supported for this URI, use POST"

    json = request.json
    if not json or not ('positive' in json or 'negative' in json) or 'corpus' not in json:
        return jsonify({'status_code': 400, 'similar_words': []})

    positive, negative = json.get('positive', []), json.get('negative', [])
    corpus = json['corpus']
    for word in positive + negative:
        if word not in word2vec_models[corpus]:
            return jsonify({'status_code': 404, 'similar_words': None,
                            'info': '{0} does not have a representation in the '
                                    '{1} corpus'.format(word, corpus)})

    return jsonify({
        'status_code': 200,
        'vector': word2vec_models[corpus].most_similar(positive=positive,
                                                       negative=negative)
    })


if __name__ == "__main__":
    app.run(port=5051)
