from __future__ import print_function

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


def keras_model():
    import os

    from keras.models import Sequential
    from keras.layers.core import Dense
    from keras.layers.core import Dropout
    from keras.layers.recurrent import GRU

    from magpie.config import HEP_TEST_PATH, HEP_TRAIN_PATH, CONSIDERED_KEYWORDS
    from magpie.feature_extraction import EMBEDDING_SIZE
    from magpie.nn.config import SAMPLE_LENGTH
    from magpie.nn.input_data import get_data_for_model

    NB_EPOCHS = 2

    model = Sequential()
    model.add(GRU(
        {{choice([256, 512])}},
        input_dim=EMBEDDING_SIZE,
        input_length=SAMPLE_LENGTH,
        init='glorot_uniform',
        inner_init='normal',
    ))
    model.add(Dropout(0.1))

    # We add a vanilla hidden layer:
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(CONSIDERED_KEYWORDS, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        class_mode='binary',
    )

    print("Model compiled")

    train_generator, (x_test, y_test) = get_data_for_model(
        model,
        as_generator=True,
        batch_size=64,
        train_dir=HEP_TRAIN_PATH,
        test_dir=HEP_TEST_PATH,
    )

    print("Data loaded")

    model.fit_generator(
        train_generator,
        len({filename[:-4] for filename in os.listdir(HEP_TRAIN_PATH)}),
        NB_EPOCHS,
        verbose=2,
    )

    score = model.evaluate(x_test, y_test)
    return {'loss': score, 'status': STATUS_OK}

if __name__ == '__main__':
    best_run = optim.minimize(keras_model,
                              algo=tpe.suggest,
                              max_evals=10,
                              trials=Trials())
    print(best_run)
