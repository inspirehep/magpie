from flask import Flask, request, jsonify
from gensim.models import Word2Vec

from magpie.api import extract_from_text
from magpie.config import NN_TRAINED_MODEL, WORD2VEC_MODELPATH
from magpie.utils import get_scaler

app = Flask('magpie')
word2vec_model = Word2Vec.load(WORD2VEC_MODELPATH)
scaler = get_scaler()


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
        'domain': 'hep'         # currently supported: 'hep'
        'text': 'my abstract'   # the text to be fed to the model

        'model': 'berger_cnn'   # (optional) the model to use for prediction
                                # currently supported: 'berger_cnn', ?
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
    if not json or 'text' not in json or 'domain' not in json:
        return jsonify({'status_code': 400, 'keywords': []})

    kwargs = dict(word2vec_model=word2vec_model, scaler=scaler)
    keywords = extract_from_text(json['text'], NN_TRAINED_MODEL, **kwargs)

    return jsonify({
        'status_code': 200,
        'keywords': keywords
    })


@app.route("/word2vec", methods=['GET', 'POST'])
def word2vec():
    """
    Takes a following JSON as input:
    {
        'domain': 'hep' # currently supported: 'hep'
        'word': 'cern'  # the word to be checked
    }

    :return:
    {
        'status_code': 200      # 200, 400, 403 etc
        'vector': []            # word2vec vector
    }
    """
    if request.method == 'GET':
        return "GET method is not supported for this URI, use POST"

    json = request.json
    if not json or 'word' not in json or 'domain' not in json:
        return jsonify({'status_code': 400, 'vector': None})

    if json['word'] in word2vec_model:
        return jsonify({
            'status_code': 200,
            'vector': word2vec_model[json['word']].tolist()
        })
    else:
        return jsonify({'status_code': 404, 'vector': None})

if __name__ == "__main__":
    app.run()
