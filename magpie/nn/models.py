from keras.layers import Input, Dense, GRU, Dropout, BatchNormalization, \
                         MaxPooling1D, Conv1D, Flatten, Concatenate
from keras.models import Model
import keras.backend as K

from magpie.config import SAMPLE_LENGTH

def top_custom_categorical_accuracy(k=1):
    def top_k_categorical_accuracy(y_true, y_pred):
        return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)
    return top_k_categorical_accuracy

def get_nn_model(nn_model, embedding, output_length, k=1):
    if nn_model == 'cnn':
        return cnn(embedding_size=embedding, output_length=output_length, k=k)
    elif nn_model == 'rnn':
        return rnn(embedding_size=embedding, output_length=output_length, k=k)
    else:
        raise ValueError("Unknown NN type: {}".format(nn_model))


def cnn(embedding_size, output_length, k):
    """ Create and return a keras model of a CNN """

    NB_FILTER = 256
    NGRAM_LENGTHS = [1, 2, 3, 4, 5]

    conv_layers, inputs = [], []

    for ngram_length in NGRAM_LENGTHS:
        current_input = Input(shape=(SAMPLE_LENGTH, embedding_size))
        inputs.append(current_input)

        convolution = Conv1D(
            NB_FILTER,
            ngram_length,
            kernel_initializer='lecun_uniform',
            activation='tanh',
        )(current_input)

        pool_size = SAMPLE_LENGTH - ngram_length + 1
        pooling = MaxPooling1D(pool_size=pool_size)(convolution)
        conv_layers.append(pooling)

    merged = Concatenate()(conv_layers)
    dropout = Dropout(0.5)(merged)
    flattened = Flatten()(dropout)
    outputs = Dense(output_length, activation='sigmoid')(flattened)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[top_custom_categorical_accuracy(k)],
    )

    return model


def rnn(embedding_size, output_length, k):
    """ Create and return a keras model of a RNN """
    HIDDEN_LAYER_SIZE = 256

    inputs = Input(shape=(SAMPLE_LENGTH, embedding_size))

    gru = GRU(
        HIDDEN_LAYER_SIZE,
        input_shape=(SAMPLE_LENGTH, embedding_size),
        kernel_initializer="glorot_uniform",
        recurrent_initializer='normal',
        activation='relu',
    )(inputs)

    batch_normalization = BatchNormalization()(gru)
    dropout = Dropout(0.1)(batch_normalization)
    outputs = Dense(output_length, activation='sigmoid')(dropout)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[top_custom_categorical_accuracy(k)],
    )

    return model
