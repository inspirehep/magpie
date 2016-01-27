from __future__ import division, unicode_literals, print_function

import os
import time

import numpy as np
import pandas as pd
import sys

from magpie.base.document import Document
from magpie.base.global_index import build_global_frequency_index
from magpie.base.inverted_index import InvertedIndex
from magpie.base.model import LearningModel
from magpie.base.word2vec import get_word2vec_model
from magpie.candidates import generate_keyword_candidates
from magpie.candidates.utils import add_gt_answers_to_candidates_set
from magpie.config import MODEL_PATH, HEP_TRAIN_PATH, HEP_ONTOLOGY, \
    HEP_TEST_PATH, BATCH_SIZE, NB_EPOCHS, WORD2VEC_MODELPATH
from magpie.evaluation.standard_evaluation import evaluate_results
from magpie.evaluation.utils import remove_unguessable_answers
from magpie.feature_extraction import preallocate_feature_matrix
from magpie.feature_extraction.document_features import \
    extract_document_features
from magpie.feature_extraction.keyword_features import extract_keyword_features, \
    rebuild_feature_matrix
from magpie.misc.utils import save_to_disk, load_from_disk
from magpie.utils import get_ontology, get_answers_for_doc, get_documents


def extract(
    path_to_file,
    ontology_path=HEP_ONTOLOGY,
    model_path=MODEL_PATH,
    recreate_ontology=False,
    verbose=False,
):
    """
    Extract keywords from a given file
    :param path_to_file: unicode with the filepath
    :param ontology_path: unicode with the ontology path
    :param model_path: unicode with the trained model path
    :param recreate_ontology: boolean flag whether to recreate the ontology
    :param verbose: whether to print additional info

    :return: set of predicted keywords
    """
    doc = Document(0, path_to_file)
    ontology = get_ontology(path=ontology_path, recreate=recreate_ontology)
    inv_index = InvertedIndex(doc)

    # Load the model
    model = load_from_disk(model_path)

    # Generate keyword candidates
    kw_candidates = list(generate_keyword_candidates(doc, ontology))

    X = preallocate_feature_matrix(len(kw_candidates))
    # Extract features for keywords
    extract_keyword_features(
        kw_candidates,
        X,
        inv_index,
        model,
    )

    # Extract document features
    extract_document_features(inv_index, X)

    X = rebuild_feature_matrix(X)

    # Predict
    y_predicted = model.scale_and_predict(X)

    kw_predicted = []
    for bit, kw in zip(y_predicted, kw_candidates):
        if bit == 1:
            kw_predicted.append(kw)

    # Print results
    if verbose:
        print("Document content:")
        print(doc)

        print("Predicted keywords:")
        for kw in kw_predicted:
            print(u"\t" + unicode(kw.get_canonical_form()))
        print()

        answers = get_answers_for_doc(doc.filename, os.path.dirname(doc.filepath))
        answers = remove_unguessable_answers(answers, ontology)

        candidates = {kw.get_canonical_form() for kw in kw_candidates}
        print("Ground truth keywords:")
        for kw in answers:
            in_candidates = "(in candidates)" if kw in candidates else ""
            print("\t" + kw.ljust(30, ' ') + in_candidates)
        print()

        y = []
        for kw in kw_candidates:
            y.append(1 if kw.get_canonical_form() in answers else 0)

        X['name'] = [kw.get_canonical_form() for kw in kw_candidates]
        X['predicted'] = y_predicted
        X['ground truth'] = y

        pd.set_option('expand_frame_repr', False)
        X = X[['name', 'predicted', 'ground truth', 'tf', 'idf', 'tfidf',
               'first_occurrence', 'last_occurrence', 'spread',
               'hops_from_anchor', 'no_of_letters', 'no_of_words']]
        print(X[(X['ground truth'] == 1) | (X['predicted'])])

    return {kw.get_canonical_form() for kw in kw_predicted}


def test(
    testset_path=HEP_TEST_PATH,
    ontology_path=HEP_ONTOLOGY,
    model_path=MODEL_PATH,
    recreate_ontology=False,
    verbose=True,
):
    """
    Test the trained model on a set under a given path.
    :param testset_path: path to the directory with the test set
    :param ontology_path: path to the ontology
    :param model_path: path where the model is pickled
    :param recreate_ontology: boolean flag whether to recreate the ontology
    :param verbose: whether to print computation times

    :return tuple of four floats (precision, recall, f1_score, accuracy)
    """
    ontology = get_ontology(path=ontology_path, recreate=recreate_ontology)

    # Load the model
    model = load_from_disk(model_path)

    feature_matrices = []
    kw_vector = []
    answers = dict()

    cand_gen_time = feature_ext_time = 0

    for doc in get_documents(testset_path):
        inv_index = InvertedIndex(doc)
        candidates_start = time.clock()

        # Generate keyword candidates
        kw_candidates = list(generate_keyword_candidates(doc, ontology))

        candidates_end = time.clock()

        # Preallocate the feature matrix
        X = preallocate_feature_matrix(len(kw_candidates))

        # Extract features for keywords
        extract_keyword_features(
            kw_candidates,
            X,
            inv_index,
            model,
        )

        # Extract document features
        extract_document_features(inv_index, X)

        features_end = time.clock()

        # Get ground truth answers
        answers[doc.doc_id] = get_answers_for_doc(doc.filename, testset_path)

        X = rebuild_feature_matrix(X)
        feature_matrices.append(X)

        kw_vector.extend([(doc.doc_id, kw) for kw in kw_candidates])

        cand_gen_time += candidates_end - candidates_start
        feature_ext_time += features_end - candidates_end

    # Merge feature matrices from different documents
    X = pd.concat(feature_matrices)

    if verbose:
        print("Candidate generation: {0:.2f}s".format(cand_gen_time))
        print("Feature extraction: {0:.2f}s".format(feature_ext_time))

    features_time = time.clock()

    # Predict
    y_predicted = model.scale_and_predict(X)

    if verbose:
        print("Prediction time: {0:.2f}s".format(time.clock() - features_time))

    # Remove ground truth answers that are not in the ontology
    for doc_id, kw_set in answers.items():
        answers[doc_id] = remove_unguessable_answers(kw_set, ontology)

    # Evaluate the results
    precision, recall, accuracy = evaluate_results(
        y_predicted,
        kw_vector,
        answers,
    )

    f1_score = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1_score, accuracy


def batch_test(
    testset_path=HEP_TEST_PATH,
    batch_size=BATCH_SIZE,
    ontology_path=HEP_ONTOLOGY,
    model_path=MODEL_PATH,
    recreate_ontology=False,
    verbose=True,
):
    ontology = get_ontology(path=ontology_path, recreate=recreate_ontology)
    doc_generator = get_documents(testset_path)
    start_time = time.clock()

    # Load the model
    model = load_from_disk(model_path)

    precision_list = []
    recall_list = []
    accuracy_list = []

    no_more_samples = False
    batch_number = 0
    if verbose:
        print("Batches:", end=' ')
    while not no_more_samples:
        batch_number += 1

        batch = []
        for i in xrange(batch_size):
            try:
                batch.append(doc_generator.next())
            except StopIteration:
                no_more_samples = True
                break

        feature_matrices = []
        kw_vector = []
        answers = dict()

        cand_gen_time = feature_ext_time = 0

        for doc in batch:
            inv_index = InvertedIndex(doc)
            candidates_start = time.clock()

            # Generate keyword candidates
            kw_candidates = list(generate_keyword_candidates(doc, ontology))

            candidates_end = time.clock()

            # Preallocate the feature matrix
            X = preallocate_feature_matrix(len(kw_candidates))

            # Extract features for keywords
            extract_keyword_features(
                kw_candidates,
                X,
                inv_index,
                model,
            )

            # Extract document features
            extract_document_features(inv_index, X)

            features_end = time.clock()

            # Get ground truth answers
            answers[doc.doc_id] = get_answers_for_doc(doc.filename, testset_path)

            X = rebuild_feature_matrix(X)
            feature_matrices.append(X)

            kw_vector.extend([(doc.doc_id, kw) for kw in kw_candidates])

            cand_gen_time += candidates_end - candidates_start
            feature_ext_time += features_end - candidates_end

        # Merge feature matrices from different documents
        X = pd.concat(feature_matrices)

        # Predict
        y_predicted = model.scale_and_predict(X)

        # Remove ground truth answers that are not in the ontology
        for doc_id, kw_set in answers.items():
            answers[doc_id] = remove_unguessable_answers(kw_set, ontology)

        # Evaluate the results
        precision, recall, accuracy = evaluate_results(
            y_predicted,
            kw_vector,
            answers,
        )

        precision_list.append(precision)
        recall_list.append(recall)
        accuracy_list.append(accuracy)

        if verbose:
            sys.stdout.write(b'.')
            sys.stdout.flush()

    if verbose:
        print()
        print("Testing finished in: {0:.2f}s".format(time.clock() - start_time))

    precision, recall = np.mean(precision_list), np.mean(recall_list)
    f1_score = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1_score, np.mean(accuracy_list)


def train(
    trainset_dir=HEP_TRAIN_PATH,
    word2vec_path=WORD2VEC_MODELPATH,
    ontology_path=HEP_ONTOLOGY,
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
    docs = get_documents(trainset_dir, as_generator=False)

    global_index = build_global_frequency_index(trainset_dir, verbose=verbose)
    word2vec_model = get_word2vec_model(word2vec_path, trainset_dir, verbose=verbose)
    model = LearningModel(global_index, word2vec_model)

    output_vectors = []
    feature_matrices = []

    cand_gen_time = feature_ext_time = 0

    for doc in docs:
        inv_index = InvertedIndex(doc)
        candidates_start = time.clock()

        # Generate keyword candidates
        kw_candidates = list(generate_keyword_candidates(doc, ontology))

        # Get ground truth answers
        doc_answers = get_answers_for_doc(doc.filename, trainset_dir)

        # If an answer was not generated, add it anyway
        add_gt_answers_to_candidates_set(kw_candidates, doc_answers, ontology)

        candidates_end = time.clock()

        # Preallocate the feature matrix
        X = preallocate_feature_matrix(len(kw_candidates))

        # Extract features for keywords
        extract_keyword_features(
            kw_candidates,
            X,
            inv_index,
            model,
        )

        # Extract document features
        extract_document_features(inv_index, X)

        X = rebuild_feature_matrix(X)
        feature_matrices.append(X)

        features_end = time.clock()

        # Create the output vector
        output_vector = np.zeros(len(kw_candidates), dtype=np.bool_)
        for i, kw in enumerate(kw_candidates):
            if kw.get_canonical_form() in doc_answers:
                output_vector[i] = True

        # feature_matrices.append(feature_matrix)
        output_vectors.append(output_vector)

        cand_gen_time += candidates_end - candidates_start
        feature_ext_time += features_end - candidates_end

    # Merge the pandas
    X = pd.concat(feature_matrices)

    # Cast the output vector to numpy
    y = np.concatenate(output_vectors)

    if verbose:
        print("Candidate generation: {0:.2f}s".format(cand_gen_time))
        print("Feature extraction: {0:.2f}s".format(feature_ext_time))
    t1 = time.clock()

    if verbose:
        print("X size: {}".format(X.shape))

    # Normalize features
    X = model.maybe_fit_and_scale(X)

    # Train the model
    model.fit_classifier(X, y)

    if verbose:
        print("Fitting the model: {0:.2f}s".format(time.clock() - t1))

    # Pickle the model
    save_to_disk(model_path, model, overwrite=True)


def batch_train(
    trainset_dir=HEP_TRAIN_PATH,
    nb_epochs=NB_EPOCHS,
    batch_size=BATCH_SIZE,
    ontology_path=HEP_ONTOLOGY,
    model_path=MODEL_PATH,
    recreate_ontology=False,
    word2vec_path=WORD2VEC_MODELPATH,
    verbose=True,
):
    """
    Train and save the model on a given dataset
    :param trainset_dir: path to the directory with the training set
    :param nb_epochs: number of passes over the training set
    :param ontology_path: path to the ontology file
    :param model_path: path to the pickled LearningModel object
    :param word2vec_path: path to the gensim word2vec model
    :param recreate_ontology: boolean flag whether to recreate the ontology
    :param verbose: whether to print computation times

    :return None if everything goes fine, error otherwise
    """
    ontology = get_ontology(path=ontology_path, recreate=recreate_ontology)

    global_index = build_global_frequency_index(trainset_dir, verbose=verbose)
    word2vec_model = get_word2vec_model(word2vec_path, trainset_dir, verbose=verbose)
    model = LearningModel(global_index, word2vec_model)

    for epoch in xrange(nb_epochs):
        doc_generator = get_documents(
            data_dir=trainset_dir,
            as_generator=True,
            shuffle=True,
        )
        samples_seen = 0
        epoch_start = time.clock()

        no_more_samples = False
        batch_number = 0
        if verbose:
            print("Batches:", end=' ')
        while not no_more_samples:
            # batch_start = time.clock()
            batch_number += 1

            output_vectors = []
            feature_matrices = []
            batch = []
            for i in xrange(batch_size):
                try:
                    batch.append(doc_generator.next())
                except StopIteration:
                    no_more_samples = True
                    break

            # TODO from here
            cand_gen_time = feature_ext_time = 0
            for doc in batch:
                inv_index = InvertedIndex(doc)
                candidates_start = time.clock()

                # Generate keyword candidates
                kw_candidates = list(generate_keyword_candidates(doc, ontology))

                # Get ground truth answers
                doc_answers = get_answers_for_doc(doc.filename, trainset_dir)

                # If an answer was not generated, add it anyway
                add_gt_answers_to_candidates_set(kw_candidates, doc_answers, ontology)

                candidates_end = time.clock()

                # Preallocate the feature matrix
                X = preallocate_feature_matrix(len(kw_candidates))

                # Extract features for keywords
                extract_keyword_features(
                    kw_candidates,
                    X,
                    inv_index,
                    model,
                )

                # Extract document features
                extract_document_features(inv_index, X)

                X = rebuild_feature_matrix(X)
                feature_matrices.append(X)

                features_end = time.clock()

                # Create the output vector
                output_vector = np.zeros(len(kw_candidates), dtype=np.bool_)
                for i, kw in enumerate(kw_candidates):
                    if kw.get_canonical_form() in doc_answers:
                        output_vector[i] = True

                output_vectors.append(output_vector)

                cand_gen_time += candidates_end - candidates_start
                feature_ext_time += features_end - candidates_end

            # Merge the pandas
            X = pd.concat(feature_matrices)

            # Cast the output vector to numpy
            y = np.concatenate(output_vectors)

            # TODO TO HERE - should be extracted

            # if verbose:
            #     print("Candidate generation: {0:.2f}s".format(cand_gen_time))
            #     print("Feature extraction: {0:.2f}s".format(feature_ext_time))
            #
            # if verbose:
            #     print("X size: {}".format(X.shape))
            samples_seen += len(X)

            # Normalize features
            X = model.maybe_fit_and_scale(X)

            # Train the model
            model.partial_fit_classifier(X, y)

            if verbose:
                # print("Batch {0} computed in: {1:.2f}s\n"
                #       .format(batch_number, time.clock() - batch_start))
                sys.stdout.write(b'.')
                sys.stdout.flush()

        if verbose:
            print()
            print("Epoch {0} finished in: {1:.2f}s"
                  .format(epoch + 1, time.clock() - epoch_start))
            print("Samples seen: {0}".format(samples_seen))

    # Pickle the model
    save_to_disk(model_path, model, overwrite=True)


if __name__ == '__main__':
    train()
