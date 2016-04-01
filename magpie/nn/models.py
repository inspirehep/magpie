from keras.layers.embeddings import Embedding
from keras.layers.convolutional import MaxPooling1D, Convolution1D
from keras.layers.core import Flatten, Dropout, Dense, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU
from keras.models import Sequential, Graph

from magpie.config import NO_OF_LABELS, EMBEDDING_SIZE
from magpie.nn.config import SAMPLE_LENGTH


def get_nn_model(nn_model):
    if nn_model == 'berger_cnn':
        return berger_cnn()
    elif nn_model == 'berger_rnn':
        return berger_rnn()
    elif nn_model == 'crnn':
        return crnn()
    elif nn_model == 'cnn_embedding':
        return cnn_embedding()
    else:
        raise ValueError("Unknown NN type: {}".format(nn_model))


def berger_cnn(embedding_size=EMBEDDING_SIZE, output_length=NO_OF_LABELS):
    """ Create and return a keras model of a CNN """
    NB_FILTER = 100
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
        class_mode='binary',
    )

    return model


def crnn(embedding_size=EMBEDDING_SIZE, output_length=NO_OF_LABELS):
    """ Create and return a keras model of a CNN with a GRU layer. """
    from keras.layers import AsymmetricZeroPadding1D
    NB_FILTER = 100
    NGRAM_LENGTH = 3
    NGRAM_LENGTHS = [1, 2, 3, 4, 5]
    HIDDEN_LAYER_SIZE = 200

    model = Sequential()
    # model.add(Convolution1D(
    #     NB_FILTER,
    #     NGRAM_LENGTH,
    #     input_dim=EMBEDDING_SIZE,
    #     input_length=SAMPLE_LENGTH,
    #     init='lecun_uniform',
    #     activation='tanh',
    # ))
    # model.add(Dropout(0.5))

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
    model.add(Dropout(0.5))

    model.add(GRU(
        HIDDEN_LAYER_SIZE,
        init='glorot_uniform',
        inner_init='normal',
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Dense(output_length, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        class_mode='binary',
    )

    return model


def berger_rnn():
    """ Create and return a keras model of a RNN """
    HIDDEN_LAYER_SIZE = 256

    model = Sequential()

    model.add(GRU(
        HIDDEN_LAYER_SIZE,
        input_dim=EMBEDDING_SIZE,
        input_length=SAMPLE_LENGTH,
        init='glorot_uniform',
        inner_init='normal',
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    # We add a vanilla hidden layer:
    # model.add(Dense(250, activation='relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(NO_OF_LABELS, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        class_mode='binary',
    )

    return model


def rnn_embedding():
    pass


def cnn_embedding():
    """ Create and return a keras model of a CNN with the embedding layer """
    NB_FILTER = 100
    NGRAM_LENGTHS = [1, 2, 3, 4]
    VOCAB_SIZE = 10000

    graph = Graph()
    graph.add_input(name='input', input_shape=(SAMPLE_LENGTH, ), dtype='int')

    graph.add_node(Embedding(
        VOCAB_SIZE,  # integers in range 0...9999
        EMBEDDING_SIZE,
        input_length=SAMPLE_LENGTH,
    ), name='embedding', input='input')

    for ngram_length in NGRAM_LENGTHS:
        conv_name = 'convolution' + str(ngram_length)
        pool_name = 'pooling' + str(ngram_length)
        graph.add_node(Convolution1D(
            NB_FILTER,
            ngram_length,
            input_dim=EMBEDDING_SIZE,
            input_length=SAMPLE_LENGTH,
            init='lecun_uniform',
            activation='tanh',
        ), input='embedding', name=conv_name)
        pool_length = SAMPLE_LENGTH - ngram_length + 1
        graph.add_node(MaxPooling1D(pool_length=pool_length),
                       input=conv_name, name=pool_name)

    graph.add_node(Dropout(0.5), name='dropout', merge_mode='concat',
                   inputs=['pooling' + str(n) for n in NGRAM_LENGTHS])
    graph.add_node(Flatten(), name='flatten', input='dropout')

    graph.add_node(Dense(NO_OF_LABELS, activation='sigmoid'),
                   name='dense', input='flatten')

    graph.add_output(name='output', input='dense')

    graph.compile(optimizer='adam', loss={'output': 'binary_crossentropy'})

    return graph
