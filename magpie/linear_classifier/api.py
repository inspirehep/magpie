from __future__ import division, unicode_literals, print_function

import sys
import time

import numpy as np
from gensim.models import Word2Vec

from magpie.config import BATCH_SIZE, NB_EPOCHS, WORD2VEC_MODELPATH
from magpie.evaluation.standard_evaluation import build_y_true, \
    calculate_basic_metrics
from magpie.linear_classifier.base.build_matrices import build_train_matrices, build_test_matrices
from magpie.linear_classifier.base.global_index import build_global_frequency_index
from magpie.linear_classifier.base.model import LearningModel
from magpie.linear_classifier.config import ONTOLOGY_PATH, MODEL_PATH
from magpie.linear_classifier.labels import get_keywords
from magpie.linear_classifier.utils import get_ontology
from magpie.utils import get_documents, save_to_disk, load_from_disk


def test(
    testset_path,
    ontology=ONTOLOGY_PATH,
    model=MODEL_PATH,
    recreate_ontology=False,
    verbose=True,
):
    """
    Test the trained model on a set under a given path.
    :param testset_path: path to the directory with the test set
    :param ontology: path to the ontology
    :param model: path where the model is pickled
    :param recreate_ontology: boolean flag whether to recreate the ontology
    :param verbose: whether to print computation times

    :return tuple of three floats (precision, recall, f1_score)
    """
    if type(model) in [str, unicode]:
        model = load_from_disk(model)

    if type(ontology) in [str, unicode]:
        ontology = get_ontology(path=ontology, recreate=recreate_ontology)

    keywords = get_keywords()
    keyword_indices = {kw: i for i, kw in enumerate(keywords)}

    all_metrics = calculate_basic_metrics([range(5)]).keys()
    metrics_agg = {m: [] for m in all_metrics}

    for doc in get_documents(testset_path, as_generator=True):
        x, answers, kw_vector = build_test_matrices(
            [doc],
            model,
            testset_path,
            ontology,
        )

        y_true = build_y_true(answers, keyword_indices, doc.doc_id)

        # Predict
        ranking = model.scale_and_predict(x.as_matrix())

        y_pred = y_true[0][ranking[::-1]]

        metrics = calculate_basic_metrics([y_pred])

        for k, v in metrics.iteritems():
            metrics_agg[k].append(v)

    return {k: np.mean(v) for k, v in metrics_agg.iteritems()}


# def batch_test(
#     testset_path=HEP_TEST_PATH,
#     batch_size=BATCH_SIZE,
#     ontology=ONTOLOGY_PATH,
#     model=MODEL_PATH,
#     recreate_ontology=False,
#     verbose=True,
# ):
#     """
#     Test the trained model on a set under a given path.
#     :param testset_path: path to the directory with the test set
#     :param batch_size: size of the testing batch
#     :param ontology: path to the ontology
#     :param model: path where the model is pickled
#     :param recreate_ontology: boolean flag whether to recreate the ontology
#     :param verbose: whether to print computation times
#
#     :return tuple of three floats (precision, recall, f1_score)
#     """
#     if type(model) in [str, unicode]:
#         model = load_from_disk(model)
#
#     if type(ontology) in [str, unicode]:
#         ontology = get_ontology(path=ontology, recreate=recreate_ontology)
#
#     doc_generator = get_documents(testset_path, as_generator=True)
#     start_time = time.clock()
#
#     all_metrics = calculate_basic_metrics([range(5)]).keys()
#     metrics_agg = {m: [] for m in all_metrics}
#
#     if verbose:
#         print("Batches:", end=' ')
#
#     no_more_samples = False
#     batch_number = 0
#     while not no_more_samples:
#         batch_number += 1
#
#         batch = []
#         for i in xrange(batch_size):
#             try:
#                 batch.append(doc_generator.next())
#             except StopIteration:
#                 no_more_samples = True
#                 break
#
#         if not batch:
#             break
#
#         X, answers, kw_vector = build_test_matrices(
#             batch,
#             model,
#             testset_path,
#             ontology,
#         )
#
#         # Predict
#         y_pred = model.scale_and_predict_confidence(X)
#
#         # Evaluate the results
#         metrics = evaluate_results(
#             y_pred,
#             kw_vector,
#             answers,
#         )
#         for k, v in metrics.iteritems():
#             metrics_agg[k].append(v)
#
#         if verbose:
#             sys.stdout.write(b'.')
#             sys.stdout.flush()
#
#     if verbose:
#         print()
#         print("Testing finished in: {0:.2f}s".format(time.clock() - start_time))
#
#     return {k: np.mean(v) for k, v in metrics_agg.iteritems()}


def train(
    trainset_dir,
    word2vec_path=WORD2VEC_MODELPATH,
    ontology_path=ONTOLOGY_PATH,
    model_path=MODEL_PATH,
    recreate_ontology=False,
    verbose=True,
):
    """
    Train and save the model on a given dataset
    :param trainset_dir: path to the directory with the training set
    :param word2vec_path: path to the gensim word2vec model
    :param ontology_path: path to the ontology file
    :param model_path: path where the model should be pickled
    :param recreate_ontology: boolean flag whether to recreate the ontology
    :param verbose: whether to print computation times

    :return None if everything goes fine, error otherwise
    """
    ontology = get_ontology(path=ontology_path, recreate=recreate_ontology)
    docs = get_documents(trainset_dir)

    global_index = build_global_frequency_index(trainset_dir, verbose=verbose)
    word2vec_model = Word2Vec.load(word2vec_path)
    model = LearningModel(global_index, word2vec_model)

    tick = time.clock()

    x, y = build_train_matrices(docs, model, trainset_dir, ontology)

    if verbose:
        print("Matrices built in: {0:.2f}s".format(time.clock() - tick))
    t1 = time.clock()

    if verbose:
        print("X size: {}".format(x.shape))

    # Normalize features
    x = model.maybe_fit_and_scale(x)

    # Train the model
    model.fit_classifier(x, y)

    if verbose:
        print("Fitting the model: {0:.2f}s".format(time.clock() - t1))

    # Pickle the model
    save_to_disk(model_path, model, overwrite=True)


def batch_train(
    trainset_dir,
    testset_dir,
    nb_epochs=NB_EPOCHS,
    batch_size=BATCH_SIZE,
    ontology_path=ONTOLOGY_PATH,
    model_path=MODEL_PATH,
    recreate_ontology=False,
    word2vec_path=WORD2VEC_MODELPATH,
    verbose=True,
):
    """
    Train and save the model on a given dataset
    :param trainset_dir: path to the directory with the training set
    :param testset_dir: path to the directory with the test set
    :param nb_epochs: number of passes over the training set
    :param batch_size: the size of a single batch
    :param ontology_path: path to the ontology file
    :param model_path: path to the pickled LearningModel object
    :param word2vec_path: path to the gensim word2vec model
    :param recreate_ontology: boolean flag whether to recreate the ontology
    :param verbose: whether to print computation times

    :return None if everything goes fine, error otherwise
    """
    ontology = get_ontology(path=ontology_path, recreate=recreate_ontology, verbose=False)

    global_index = build_global_frequency_index(trainset_dir, verbose=False)
    word2vec_model = Word2Vec.load(word2vec_path)
    model = LearningModel(global_index, word2vec_model)
    previous_best = -1

    for epoch in xrange(nb_epochs):
        doc_generator = get_documents(
            trainset_dir,
            as_generator=True,
            shuffle=True,
        )
        epoch_start = time.clock()

        if verbose:
            print("Epoch {}".format(epoch + 1), end=' ')

        no_more_samples = False
        batch_number = 0
        while not no_more_samples:
            batch_number += 1

            batch = []
            for i in xrange(batch_size):
                try:
                    batch.append(doc_generator.next())
                except StopIteration:
                    no_more_samples = True
                    break

            if not batch:
                break

            x, y = build_train_matrices(batch, model, trainset_dir, ontology)

            # Normalize features
            x = model.maybe_fit_and_scale(x)

            # Train the model
            model.partial_fit_classifier(x, y)

            if verbose:
                sys.stdout.write(b'.')
                sys.stdout.flush()

        if verbose:
            print(" {0:.2f}s".format(time.clock() - epoch_start))

        metrics = test(
            testset_dir,
            model=model,
            ontology=ontology,
            verbose=False
        )

        for k, v in metrics.iteritems():
            print("{0}: {1}".format(k, v))

        if metrics['map'] > previous_best:
            previous_best = metrics['map']
            save_to_disk(model_path, model, overwrite=True)
