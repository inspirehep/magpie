from __future__ import division

import os
import time
import logging

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

from magpie.base.document import Document
from magpie.base.global_index import GlobalFrequencyIndex
from magpie.base.inverted_index import InvertedIndex
from magpie.base.ontology import OntologyFactory
from magpie.candidates import generate_keyword_candidates
from magpie.config import ONTOLOGY_DIR, ROOT_DIR
from magpie.feature_extraction.document_features import \
    extract_document_features
from magpie.feature_extraction.keyword_features import extract_keyword_features

HEP_ONTOLOGY = os.path.join(ONTOLOGY_DIR, 'HEPontCore.rdf')


def get_ontology(recreate=False):
    return OntologyFactory(HEP_ONTOLOGY, recreate=recreate)


def get_documents():
    hep_data_path = os.path.join(ROOT_DIR, 'data', 'hep')
    files = {fname[:-4] for fname in os.listdir(hep_data_path)}
    return [Document(doc_id, os.path.join(hep_data_path, f + '.txt'))
            for doc_id, f in enumerate(files)]


def get_answers():
    hep_data_path = os.path.join(ROOT_DIR, 'data', 'hep')
    answers = dict()

    files = {fname[:-4] for fname in os.listdir(hep_data_path)}
    for f in files:
        with open(os.path.join(hep_data_path, f + '.key'), 'rb') as answer_file:
            answers[f] = {line.rstrip('\n') for line in answer_file}

    return answers


def train(recreate_ontology=False):
    ontology = get_ontology(recreate=recreate_ontology)
    docs = get_documents()
    answers = get_answers()

    global_freqs = GlobalFrequencyIndex(docs)
    feature_matrices = []
    output_vectors = []

    start_time = time.clock()

    for doc in docs:
        inv_index = InvertedIndex(doc)

        # Generate keyword candidates
        kw_candidates = [kw for kw in
                         generate_keyword_candidates(doc, ontology)]

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

        # Get ground truth answers
        doc_answers = answers.get(doc.filename[:-4])
        if not doc_answers:
            logging.error(
                "File {0} containing answers to the file {1} was not found"
                .format(doc.filename[-4]) + '.key', doc.filename
            )
            continue

        # Create the output vector
        output_vector = []
        for kw in kw_candidates:
            if kw.get_canonical_form() in doc_answers:
                output_vector.append(1)  # True
            else:
                output_vector.append(0)  # False

        feature_matrices.append(feature_matrix)
        output_vectors.append(output_vector)

    # Merge feature matrices and output vectors from different documents
    X = pd.concat(feature_matrices)
    y = np.array([x for inner in output_vectors for x in inner])  # flatten

    # Normalize features
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the model
    classifier = RandomForestClassifier()
    classifier = classifier.fit(X_scaled, y)

    print
    end_time = time.clock()
    print(u"Time elapsed: " + str(end_time - start_time))


def calculate_recall_for_kw_candidates(recreate_ontology=False):
    average_recall = 0
    total_kw_number = 0
    ontology = OntologyFactory(HEP_ONTOLOGY, recreate=recreate_ontology)
    hep_data_path = os.path.join(ROOT_DIR, 'data', 'hep')
    files = {fname[:-4] for fname in os.listdir(hep_data_path)}
    start_time = time.clock()
    for f in files:
        document = Document(os.path.join(hep_data_path, f + '.txt'))
        kw_candidates = {kw.get_canonical_form() for kw
                         in generate_keyword_candidates(document, ontology)}
        with open(os.path.join(hep_data_path, f + '.key'), 'rb') as answer_file:
            answers = {line.rstrip('\n') for line in answer_file}

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
        print(u"Paper: " + f)
        print(u"Candidates: " + str(len(kw_candidates)))
        print(u"Recall: " + unicode(recall))

        average_recall += recall
        total_kw_number += len(kw_candidates)

    average_recall /= len(files)

    print
    print(u"Total # of keywords: " + str(total_kw_number))
    print(u"Averaged recall: " + unicode(average_recall))
    end_time = time.clock()
    print(u"Time elapsed: " + str(end_time - start_time))

