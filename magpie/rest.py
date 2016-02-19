from flask import Flask, request, jsonify

from magpie.api import extract_from_text
from magpie.config import NN_TRAINED_MODEL

app = Flask('magpie')


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

    keywords = extract_from_text(json['text'], model_path=NN_TRAINED_MODEL)

    return jsonify({
        'status_code': 200,
        'keywords': keywords
    })


if __name__ == "__main__":
    app.run()
