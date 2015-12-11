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
from magpie.config import ONTOLOGY_DIR, ROOT_DIR, MODEL_PATH, HEP_TRAIN_PATH, \
    HEP_ONTOLOGY, HEP_TEST_PATH
from magpie.evaluation.standard_evaluation import evaluate_results
from magpie.feature_extraction.document_features import \
    extract_document_features
from magpie.feature_extraction.keyword_features import extract_keyword_features
from magpie.utils.utils import save_to_disk, load_from_disk


def get_ontology(path=HEP_ONTOLOGY, recreate=False):
    """ Load or create the ontology from a given path """
    return OntologyFactory(path, recreate=recreate)


def get_documents(data_dir=HEP_TRAIN_PATH):
    """ Extract documents from *.txt files in a given directory """
    files = {filename[:-4] for filename in os.listdir(data_dir)}
    return [Document(doc_id, os.path.join(data_dir, f + '.txt'))
            for doc_id, f in enumerate(files)]


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


def extract(path_to_file, recreate_ontology=False):
    pass


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
    docs = get_documents(testset_path)

    global_freqs = GlobalFrequencyIndex(docs)
    feature_matrices = []
    kw_vector = []
    answers = dict()

    cand_gen_time = feature_ext_time = 0

    for doc in docs:
        inv_index = InvertedIndex(doc)
        candidates_start = time.clock()

        # Generate keyword candidates
        kw_candidates = list(generate_keyword_candidates(doc, ontology))

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
    print(u"Candidate generation: " + str(cand_gen_time) + u"s")
    print(u"Feature extraction: " + str(feature_ext_time) + u"s")

    # Load the model
    model = load_from_disk(model_path)

    # Predict
    y_predicted = model.scale_and_predict(X)

    predict_time = time.clock()
    print(u"Prediction time: " + str(predict_time - features_time) + u"s")

    # Evaluate the results
    precision, recall, accuracy = evaluate_results(
        y_predicted,
        kw_vector,
        answers
    )

    evaluation_time = time.clock()
    print(u"Evaluation time: " + str(evaluation_time - predict_time) + u"s")

    f1_score = (2 * precision * recall) / (precision + recall)
    print
    print(u"Precision: " + str(precision * 100) + u"%")
    print(u"Recall: " + str(recall * 100) + u"%")
    print(u"F1-score: " + unicode(f1_score * 100) + u"%")
    print(u"Accuracy: " + str(accuracy * 100) + u"%")


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
    docs = get_documents(trainset_dir)

    global_freqs = GlobalFrequencyIndex(docs)
    feature_matrices = []
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

        features_end = time.clock()

        # Create the output vector
        output_vector = []
        for kw in kw_candidates:
            if kw.get_canonical_form() in doc_answers:
                output_vector.append(1)  # True
            else:
                output_vector.append(0)  # False

        feature_matrices.append(feature_matrix)
        output_vectors.append(output_vector)

        cand_gen_time += candidates_end - candidates_start
        feature_ext_time += features_end - candidates_end

    # Merge feature matrices and output vectors from different documents
    X = pd.concat(feature_matrices)
    y = np.array([x for inner in output_vectors for x in inner])  # flatten

    print(u"Candidate generation: " + str(cand_gen_time) + u"s")
    print(u"Feature extraction: " + str(feature_ext_time) + u"s")
    fitting_time = time.clock()

    # Normalize features
    model = LearningModel()
    x_scaled = model.fit_and_scale(X)

    # Train the model
    model.fit_classifier(x_scaled, y)

    pickle_time = time.clock()
    print(u"Fitting the model: " + str(pickle_time - fitting_time) + u"s")

    # Pickle the model
    save_to_disk(MODEL_PATH, model, overwrite=True)

    end_time = time.clock()
    print(u"Pickling the model: " + str(end_time - pickle_time) + u"s")


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
        print(u"Recall: " + unicode(recall))

        average_recall += recall
        total_kw_number += len(kw_candidates)

    average_recall /= len(docs)

    print
    print(u"Total # of keywords: " + str(total_kw_number))
    print(u"Averaged recall: " + unicode(average_recall))
    end_time = time.clock()
    print(u"Time elapsed: " + str(end_time - start_time))

if __name__ == '__main__':
    test()
