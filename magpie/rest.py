from collections import defaultdict

from flask import Flask, request, jsonify
from gensim.models import Word2Vec

from magpie.api import extract_from_text
from magpie.config import NN_TRAINED_MODEL, WORD2VEC_MODELPATH
from magpie.nn.models import get_nn_model
from magpie.utils import get_scaler

app = Flask('magpie')
word2vec_model = Word2Vec.load(WORD2VEC_MODELPATH)
scaler = get_scaler()
nn_models = defaultdict(dict)

supported_models = ['berger_cnn']
supported_domains = ['hep']


def get_cached_model(nn_model, domain):
    """ Get the cached Keras NN model or reload it if missed. """
    global nn_models

    if nn_model not in nn_models or domain not in nn_models[nn_model]:
        m = get_nn_model(nn_model)
        path = get_model_weights_path(domain)
        m.load_weights(path)
        nn_models[nn_model][domain] = m

    return nn_models[nn_model][domain]


def get_model_weights_path(domain):
    """ Return the path to the trained model for a given domain. """
    if domain == 'hep':
        return NN_TRAINED_MODEL
    else:
        raise ValueError("Incorrect domain")


@app.route('/')
@app.route('/hello')
def hello():
    """ Function for testing purposes """
    return 'Hello World!'


@app.route("/extract", methods=['GET', 'POST'])
def extract():
    """
    Takes a following JSON as input:
    {
        'text': 'my abstract'   # the text to be fed to the model

        'domain': 'hep'         # (optional) currently supported: 'hep'
        'model': 'berger_cnn'   # (optional) the model to use for prediction
                                # currently supported: 'berger_cnn'
    }

    :return:
    {
        'status_code': 200      # 200, 400, 403 etc
        'keywords': []          # list of two-element tuples each with a keyword
                                # and its confidence value e.g. [('jan', 0.95)]
    }
    """
    if request.method == 'GET':
        return "GET method is not supported for this URI, use POST"

    json = request.json
    if not json or 'text' not in json:
        return jsonify({'status_code': 400, 'keywords': []})

    domain = json.get('domain', supported_domains[0])
    if domain not in supported_domains:
        return jsonify({'status_code': 404, 'keywords': [],
                        'info': 'Domain ' + domain + ' is not available'})

    model_name = json.get('model', supported_models[0])
    if model_name not in supported_models:
        return jsonify({'status_code': 404, 'keywords': [],
                        'info': 'Model ' + model_name + ' is not available'})

    model = get_cached_model(model_name, domain)

    kwargs = dict(word2vec_model=word2vec_model, scaler=scaler)
    keywords = extract_from_text(json['text'], model, **kwargs)

    return jsonify({
        'status_code': 200,
        'keywords': keywords
    })


@app.route("/word2vec", methods=['GET', 'POST'])
def word2vec():
    """
    Takes a following JSON as input:
    {
        'domain': 'hep'                 # currently supported: 'hep'
        'positive': ['cern', 'geneva']  # words to add
        'negative': ['heidelberg']      # words to subtract
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
    if not json or not ('positive' in json or 'negative' in json) or 'domain' not in json:
        return jsonify({'status_code': 400, 'similar_words': []})

    positive, negative = json.get('positive', []), json.get('negative', [])
    for word in positive + negative:
        if word not in word2vec_model:
            return jsonify({'status_code': 404, 'similar_words': None,
                            'info': word + ' does not have a representation'})

    return jsonify({
        'status_code': 200,
        'vector': word2vec_model.most_similar(positive=positive, negative=negative)
    })

if __name__ == "__main__":
    app.run(port=5051)
