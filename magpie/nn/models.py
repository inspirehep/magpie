from keras.layers.convolutional import MaxPooling1D, Convolution1D
from keras.layers.core import Flatten, Dropout, Dense, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU
from keras.models import Sequential

from magpie.config import EMBEDDING_SIZE, SAMPLE_LENGTH

DEFAULT_LABELS = 1000


def get_nn_model(nn_model, embedding=EMBEDDING_SIZE, output_length=DEFAULT_LABELS):
    if nn_model == 'cnn':
        return cnn(embedding_size=embedding, output_length=output_length)
    elif nn_model == 'rnn':
        return rnn(embedding_size=embedding, output_length=output_length)
    elif nn_model == 'crnn':
        return crnn(embedding_size=embedding, output_length=output_length)
    else:
        raise ValueError("Unknown NN type: {}".format(nn_model))


def cnn(embedding_size=EMBEDDING_SIZE, output_length=DEFAULT_LABELS):
    """ Create and return a keras model of a CNN """
    NB_FILTER = 256
    NGRAM_LENGTHS = [1, 2, 3, 4, 5]

    conv_layers = []
    for ngram_length in NGRAM_LENGTHS:
        ngram_layer = Sequential()
        ngram_layer.add(Convolution1D(
            NB_FILTER,
            ngram_length,
            input_dim=embedding_size,
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

    model.add(Dense(output_length, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )

    return model


def crnn(embedding_size=EMBEDDING_SIZE, output_length=DEFAULT_LABELS):
    """ Create and return a keras model of a CNN with a GRU layer. """
    # Works only with customized Keras and TensorFlow
    from keras.layers import AsymmetricZeroPadding1D
    NB_FILTER = 256
    NGRAM_LENGTHS = [1, 2, 3, 4, 5]
    HIDDEN_LAYER_SIZE = 512

    model = Sequential()

    conv_layers = []
    for ngram_length in NGRAM_LENGTHS:
        ngram_layer = Sequential()
        ngram_layer.add(Convolution1D(
            NB_FILTER,
            ngram_length,
            input_dim=embedding_size,
            input_length=SAMPLE_LENGTH,
            init='lecun_uniform',
            activation='tanh',
        ))
        ngram_layer.add(AsymmetricZeroPadding1D(right_padding=ngram_length - 1))
        conv_layers.append(ngram_layer)

    model.add(Merge(conv_layers, mode='concat'))
    model.add(Dropout(0.3))

    model.add(GRU(
        HIDDEN_LAYER_SIZE,
        init='glorot_uniform',
        inner_init='normal',
        activation='tanh',
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(output_length, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )

    return model


def rnn(embedding_size=EMBEDDING_SIZE, output_length=DEFAULT_LABELS):
    """ Create and return a keras model of a RNN """
    HIDDEN_LAYER_SIZE = 256

    model = Sequential()

    model.add(GRU(
        HIDDEN_LAYER_SIZE,
        input_dim=embedding_size,
        input_length=SAMPLE_LENGTH,
        init='glorot_uniform',
        inner_init='normal',
        activation='relu',
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Dense(output_length, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )

    return model
