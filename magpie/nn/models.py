from keras.layers.convolutional import MaxPooling1D, Convolution1D
from keras.layers.core import Flatten, Dropout, Dense, Merge
from keras.layers.recurrent import GRU
from keras.models import Sequential

from magpie.feature_extraction import WORD2VEC_LENGTH
from magpie.nn.config import OUTPUT_UNITS, SAMPLE_LENGTH

# Convolutional parameters
NB_FILTER = 100
NGRAM_LENGTHS = [1, 2, 3, 4]

# Recurrent parameters
HIDDEN_LAYER_SIZE = 200


def get_nn_model(nn_model):
    if nn_model == 'cnn':
        return build_cnn_model()
    elif nn_model == 'rnn':
        return build_rnn_model()
    else:
        raise ValueError("Unknown NN type: {}".format(nn_model))


def build_cnn_model():
    """
    Create and return a keras model of a CNN

    :return: keras model
    """
    conv_layers = []
    for ngram_length in NGRAM_LENGTHS:
        ngram_layer = Sequential()
        ngram_layer.add(Convolution1D(
            NB_FILTER,
            ngram_length,
            input_dim=WORD2VEC_LENGTH,
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

    model.add(Dense(OUTPUT_UNITS, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adadelta',
        class_mode='binary',
    )

    return model


def build_rnn_model():
    """
    Create and return a keras model of a RNN

    :return: keras model
    """
    model = Sequential()

    model.add(GRU(
        HIDDEN_LAYER_SIZE,
        input_dim=WORD2VEC_LENGTH,
        input_length=SAMPLE_LENGTH,
        init='glorot_uniform',
        inner_init='normal',
    ))
    model.add(Dropout(0.5))

    # We add a vanilla hidden layer:
    # model.add(Dense(250, activation='relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(OUTPUT_UNITS, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        class_mode='binary',
    )

    return model
