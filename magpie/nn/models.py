from keras.layers.convolutional import MaxPooling1D, Convolution1D
from keras.layers.core import Flatten, Dropout, Dense, Merge
from keras.layers.recurrent import GRU
from keras.models import Sequential

from magpie.config import CONSIDERED_KEYWORDS
from magpie.feature_extraction import EMBEDDING_SIZE
from magpie.nn.config import SAMPLE_LENGTH


def get_nn_model(nn_model):
    if nn_model == 'berger_cnn':
        return berger_cnn()
    elif nn_model == 'berger_rnn':
        return berger_rnn()
    elif nn_model == 'berger_cnn_rnn':
        return berger_cnn_rnn()
    else:
        raise ValueError("Unknown NN type: {}".format(nn_model))


def berger_cnn():
    """ Create and return a keras model of a CNN """
    NB_FILTER = 100
    NGRAM_LENGTHS = [1, 2, 3, 4]

    conv_layers = []
    for ngram_length in NGRAM_LENGTHS:
        ngram_layer = Sequential()
        ngram_layer.add(Convolution1D(
            NB_FILTER,
            ngram_length,
            input_dim=EMBEDDING_SIZE,
            input_length=SAMPLE_LENGTH,
            init='lecun_uniform',
            activation='tanh',
        ))
        pool_length = SAMPLE_LENGTH - ngram_length + 1
        ngram_layer.add(MaxPooling1D(pool_length=pool_length))
        conv_layers.append(ngram_layer)

    model = Sequential()
    model.add(Merge(conv_layers, mode='concat'))

    model.add(Dropout(0.5))
    model.add(Flatten())

    # We add a vanilla hidden layer:
    # model.add(Dense(250, activation='relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(CONSIDERED_KEYWORDS, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        class_mode='binary',
    )

    return model


def berger_cnn_rnn():
    """ Create and return a keras model of a CNN with a GRU layer. """
    NB_FILTER = 100
    NGRAM_LENGTHS = [1, 2, 3, 4]
    HIDDEN_LAYER_SIZE = 200

    conv_layers = []
    for ngram_length in NGRAM_LENGTHS:
        ngram_layer = Sequential()
        ngram_layer.add(Convolution1D(
            NB_FILTER,
            ngram_length,
            input_dim=EMBEDDING_SIZE,
            input_length=SAMPLE_LENGTH,
            init='lecun_uniform',
            activation='tanh',
        ))
        pool_length = SAMPLE_LENGTH - ngram_length + 1
        ngram_layer.add(MaxPooling1D(pool_length=pool_length))
        conv_layers.append(ngram_layer)

    model = Sequential()
    model.add(Merge(conv_layers, mode='concat'))

    model.add(Dropout(0.5))

    model.add(GRU(
        HIDDEN_LAYER_SIZE,
        init='glorot_uniform',
        inner_init='normal',
    ))
    model.add(Dropout(0.5))

    model.add(Dense(CONSIDERED_KEYWORDS, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        class_mode='binary',
    )

    return model


def berger_rnn():
    """ Create and return a keras model of a RNN """
    HIDDEN_LAYER_SIZE = 200

    model = Sequential()

    model.add(GRU(
        HIDDEN_LAYER_SIZE,
        input_dim=EMBEDDING_SIZE,
        input_length=SAMPLE_LENGTH,
        init='glorot_uniform',
        inner_init='normal',
    ))
    model.add(Dropout(0.5))

    # We add a vanilla hidden layer:
    # model.add(Dense(250, activation='relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(CONSIDERED_KEYWORDS, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        class_mode='binary',
    )

    return model
