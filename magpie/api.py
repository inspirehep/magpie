from __future__ import division

import os
import time

import numpy as np
import pandas as pd

from magpie.base.document import Document
from magpie.base.global_index import GlobalFrequencyIndex
from magpie.base.inverted_index import InvertedIndex
from magpie.base.model import LearningModel
from magpie.base.ontology import OntologyFactory
from magpie.candidates import generate_keyword_candidates
from magpie.candidates.utils import add_gt_answers_to_candidates_set
from magpie.config import MODEL_PATH, HEP_TRAIN_PATH, HEP_ONTOLOGY, \
    HEP_TEST_PATH
from magpie.evaluation.standard_evaluation import evaluate_results
from magpie.evaluation.utils import remove_unguessable_answers
from magpie.feature_extraction.document_features import \
    extract_document_features
from magpie.feature_extraction.keyword_features import extract_keyword_features
from magpie.utils.utils import save_to_disk, load_from_disk

__all__ = ['extract', 'train', 'test']


def get_ontology(path=HEP_ONTOLOGY, recreate=False):
    """ Load or create the ontology from a given path """
    return OntologyFactory(path, recreate=recreate)


def get_documents(data_dir=HEP_TRAIN_PATH, as_generator=True):
    """ Extract documents from *.txt files in a given directory """
    files = {filename[:-4] for filename in os.listdir(data_dir)}
    generator = (Document(doc_id, os.path.join(data_dir, f + '.txt'))
                 for doc_id, f in enumerate(files))
    return generator if as_generator else list(generator)


def get_all_answers(data_dir):
    """ Extract ground truth answers from *.key files in a given directory """
    answers = dict()

    files = {filename[:-4] for filename in os.listdir(data_dir)}
    for f in files:
        with open(os.path.join(data_dir, f + '.key'), 'rb') as answer_file:
            answers[f] = {unicode(line.rstrip('\n')) for line in answer_file}

    return answers


def get_answers_for_doc(doc_name, data_dir):
    """
    Read ground_truth answers from a .key file corresponding to the doc_name
    :param doc_name: the name of the document, should end with .txt
    :param data_dir: directory in which the documents and answer files are

    :return: set of unicodes containing answers for this particular document
    """
    filename = os.path.join(data_dir, doc_name[:-4] + '.key')

    if not os.path.exists(filename):
        raise ValueError("Answer file " + filename + " does not exist")

    with open(filename, 'rb') as f:
        answers = {unicode(line.rstrip('\n')) for line in f}

    return answers


def extract(
    path_to_file,
    ontology_path=HEP_ONTOLOGY,
    model_path=MODEL_PATH,
    show_answers=False,
    recreate_ontology=False
):
    """
    Extract keywords from a given file
    :param path_to_file: unicode with the filepath
    :param ontology_path: unicode with the ontology path
    :param model_path: unicode with the trained model path
    :param recreate_ontology: boolean flag whether to recreate the ontology
    :return:
    """
    doc = Document(0, path_to_file)
    ontology = get_ontology(path=ontology_path, recreate=recreate_ontology)
    inv_index = InvertedIndex(doc)

    # Load the model
    model = load_from_disk(model_path)

    # Generate keyword candidates
    kw_candidates = list(generate_keyword_candidates(doc, ontology))

    # Extract features for keywords
    kw_features = extract_keyword_features(
        kw_candidates,
        inv_index,
        model.get_global_index()
    )

    # Extract document features
    doc_features = extract_document_features(inv_index, len(kw_candidates))

    # Merge matrices
    X = pd.concat([kw_features, doc_features], axis=1)

    # Predict
    y_predicted = model.scale_and_predict(X)

    kw_predicted = []
    for bit, kw in zip(y_predicted, kw_candidates):
        if bit == 1:
            kw_predicted.append(kw)

    # Print results
    print(u"Document content:")
    print doc

    print(u"Predicted keywords:")
    for kw in kw_predicted:
        print(u"\t" + unicode(kw.get_canonical_form()))
    print

    if show_answers:
        answers = get_answers_for_doc(doc.filename, os.path.dirname(doc.filepath))

        answer_dict = dict(placeholder=answers)
        remove_unguessable_answers(answer_dict, ontology)
        answers = answer_dict['placeholder']

        candidates = {kw.get_canonical_form() for kw in kw_candidates}
        print(u"Ground truth keywords:")
        for kw in answers:
            in_candidates = u"(in candidates)" if kw in candidates else u""
            print(u"\t" + kw.ljust(30, u' ') + in_candidates)
        print

        y = []
        for kw in kw_candidates:
            y.append(1 if kw.get_canonical_form() in answers else 0)

        X['name'] = [kw.get_canonical_form() for kw in kw_candidates]
        X['predicted'] = y_predicted
        X['ground truth'] = y

        pd.set_option('expand_frame_repr', False)
        X = X[['name', 'predicted', 'ground truth', 'tf', 'idf', 'tfidf',
               'first_occurrence', 'last_occurrence', 'spread', 'no_of_letters',
               'no_of_words']]
        print X[(X['ground truth'] == 1) | (X['predicted'])]


def test(
    testset_path=HEP_TEST_PATH,
    ontology_path=HEP_ONTOLOGY,
    model_path=MODEL_PATH,
    recreate_ontology=False
):
    """
    Test the trained model on a set under a given path
    :param testset_dir: path to the directory with the test set
    :param recreate_ontology: boolean flag whether to recreate the ontology
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

        # Extract features for keywords
        kw_features = extract_keyword_features(
            kw_candidates,
            inv_index,
            model.get_global_index()
        )

        # Extract document features
        doc_features = extract_document_features(inv_index, len(kw_candidates))

        # Merge matrices
        feature_matrix = pd.concat([kw_features, doc_features], axis=1)

        features_end = time.clock()

        # Get ground truth answers
        answers[doc.doc_id] = get_answers_for_doc(doc.filename, testset_path)

        feature_matrices.append(feature_matrix)
        kw_vector.extend([(doc.doc_id, kw) for kw in kw_candidates])

        cand_gen_time += candidates_end - candidates_start
        feature_ext_time += features_end - candidates_end

    # Merge feature matrices from different documents
    X = pd.concat(feature_matrices)

    features_time = time.clock()
    print(u"Candidate generation: {0:.2f}s".format(cand_gen_time))
    print(u"Feature extraction: {0:.2f}s".format(feature_ext_time))

    # Predict
    y_predicted = model.scale_and_predict(X)

    predict_time = time.clock()
    print(u"Prediction time: {0:.2f}s".format(predict_time - features_time))

    # Remove ground truth answers that are not in the ontology
    remove_unguessable_answers(answers, ontology)

    # Evaluate the results
    precision, recall, accuracy = evaluate_results(
        y_predicted,
        kw_vector,
        answers
    )

    evaluation_time = time.clock()
    print(u"Evaluation time: {0:.2f}s".format(evaluation_time - predict_time))

    f1_score = (2 * precision * recall) / (precision + recall)
    print
    print(u"Precision: {0:.2f}%".format(precision * 100))
    print(u"Recall: {0:.2f}%".format(recall * 100))
    print(u"F1-score: {0:.2f}%".format(f1_score * 100))
    print(u"Accuracy: {0:.2f}%".format(accuracy * 100))


def train(
    trainset_dir=HEP_TRAIN_PATH,
    ontology_path=HEP_ONTOLOGY,
    model_path=MODEL_PATH,
    recreate_ontology=False
):
    """
    Train and save the model on a given dataset
    :param trainset_dir: path to the directory with the training set
    :param recreate_ontology: boolean flag whether to recreate the ontology
    """
    ontology = get_ontology(path=ontology_path, recreate=recreate_ontology)
    docs = get_documents(trainset_dir, as_generator=False)

    global_freqs = GlobalFrequencyIndex(docs)
    X = pd.DataFrame()
    output_vectors = []

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

        # Extract features for keywords
        kw_features = extract_keyword_features(
            kw_candidates,
            inv_index,
            global_freqs
        )

        # Extract document features
        doc_features = extract_document_features(inv_index, len(kw_candidates))

        # Merge matrices
        feature_matrix = pd.concat([kw_features, doc_features], axis=1)
        X = pd.concat([X, feature_matrix])

        features_end = time.clock()

        # Create the output vector
        # TODO this vector is very sparse, we can make it more memory efficient
        output_vector = []
        for kw in kw_candidates:
            if kw.get_canonical_form() in doc_answers:
                output_vector.append(1)  # True
            else:
                output_vector.append(0)  # False

        # feature_matrices.append(feature_matrix)
        output_vectors.extend(output_vector)

        cand_gen_time += candidates_end - candidates_start
        feature_ext_time += features_end - candidates_end

    # Cast the output vector to scipy
    y = np.array(output_vectors)

    print(u"Candidate generation: {0:.2f}s".format(cand_gen_time))
    print(u"Feature extraction: {0:.2f}s".format(feature_ext_time))
    fitting_time = time.clock()

    # Normalize features
    model = LearningModel(global_frequencies=global_freqs)
    x_scaled = model.fit_and_scale(X)

    # Train the model
    model.fit_classifier(x_scaled, y)

    pickle_time = time.clock()
    print(u"Fitting the model: {0:.2f}s".format(pickle_time - fitting_time))

    # Pickle the model
    save_to_disk(model_path, model, overwrite=True)

    end_time = time.clock()
    print(u"Pickling the model: {0:.2f}s".format(end_time - pickle_time))


def calculate_recall_for_kw_candidates(data_dir=HEP_TRAIN_PATH,
                                       recreate_ontology=False):
    """
    Generate keyword candidates for files in a given directory
    and compute their recall in reference to ground truth answers
    :param data_dir: directory with .txt and .key files
    :param recreate_ontology: boolean flag for recreating the ontology
    """
    average_recall = 0
    total_kw_number = 0

    ontology = get_ontology(recreate=recreate_ontology)
    docs = get_documents(data_dir)
    total_docs = 0

    start_time = time.clock()
    for doc in docs:
        kw_candidates = {kw.get_canonical_form() for kw
                         in generate_keyword_candidates(doc, ontology)}

        answers = get_answers_for_doc(doc.filename, data_dir)

        # print(document.get_meaningful_words())

        # print(u"Candidates:")
        # for kw in sorted(kw_candidates):
        #     print(u"\t" + unicode(kw))
        # print
        #
        # print(u"Answers:")
        # for kw in sorted(answers):
        #     print(u"\t" + unicode(kw))
        # print
        #
        # print(u"Conjunction:")
        # for kw in sorted(kw_candidates & answers):
        #     print(u"\t" + unicode(kw))
        # print

        recall = len(kw_candidates & answers) / (len(answers))
        print
        print(u"Paper: " + doc.filename)
        print(u"Candidates: " + str(len(kw_candidates)))
        print(u"Recall: " + unicode(recall * 100) + u"%")

        average_recall += recall
        total_kw_number += len(kw_candidates)
        total_docs += 1

    average_recall /= total_docs

    print
    print(u"Total # of keywords: " + str(total_kw_number))
    print(u"Averaged recall: " + unicode(average_recall * 100) + u"%")
    end_time = time.clock()
    print(u"Time elapsed: " + str(end_time - start_time))
