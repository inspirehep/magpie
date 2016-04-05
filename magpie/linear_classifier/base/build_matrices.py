import numpy as np
import pandas as pd
from magpie.linear_classifier.candidates import generate_keyword_candidates
from magpie.linear_classifier.feature_extraction import preallocate_feature_matrix
from magpie.linear_classifier.feature_extraction.document_features import \
    extract_document_features

from magpie.linear_classifier.base.inverted_index import InvertedIndex
from magpie.linear_classifier.candidates.utils import \
    add_gt_answers_to_candidates_set
from magpie.linear_classifier.feature_extraction.keyword_features import \
    extract_keyword_features, rebuild_feature_matrix
from magpie.misc.labels import get_keywords
from magpie.utils import get_answers_for_doc


def build_test_matrices(docs, model, file_dir, ontology):
    """
    Build the X feature matrix and answers & kw_vector variables, needed for
    evaluating the predictions.
    :param docs: documents to process. Either list or generator of Document obj
    :param model: LearningModel object
    :param file_dir: directory where the answer files are located
    :param ontology: Ontology object
    :return: X numpy array, answers dictionary and kw_vector tuple list
    """
    considered_keywords = set(get_keywords())
    feature_matrices = []
    kw_vector = []
    answers = dict()

    for doc in docs:
        inv_index = InvertedIndex(doc)

        # Generate keyword candidates
        kw_candidates = list(generate_keyword_candidates(doc, ontology))

        X = build_feature_matrix(kw_candidates, inv_index, model)
        feature_matrices.append(X)

        # Get ground truth answers
        answers[doc.doc_id] = get_answers_for_doc(
            doc.filename,
            file_dir,
            filtered_by=considered_keywords,
        )

        kw_vector.extend([(doc.doc_id, kw.get_canonical_form())
                          for kw in kw_candidates])

    # Merge feature matrices from different documents
    X = pd.concat(feature_matrices)

    return X, answers, kw_vector


def build_train_matrices(docs, model, file_dir, ontology):
    """
    Build X matrix and y vector from the input data
    :param docs: documents to process. Either list of generator of Document obj
    :param model: LearningModel object
    :param file_dir: directory where the answer files are located
    :param ontology: Ontology object

    :return: X and y numpy arrays
    """
    considered_keywords = set(get_keywords())
    feature_matrices = []
    output_vectors = []

    for doc in docs:
        inv_index = InvertedIndex(doc)

        # Generate keyword candidates
        kw_candidates = list(generate_keyword_candidates(doc, ontology))

        # Get ground truth answers
        doc_answers = get_answers_for_doc(
            doc.filename,
            file_dir,
            filtered_by=considered_keywords,
        )

        # If an answer was not generated, add it anyway
        add_gt_answers_to_candidates_set(kw_candidates, doc_answers, ontology)

        # Create the output vector
        output_vector = np.zeros((len(kw_candidates), 2), dtype=np.int16)
        for i, kw in enumerate(kw_candidates):
            if kw.get_canonical_form() in doc_answers:
                output_vector[i][0] = True
            output_vector[i][1] = doc.doc_id

        output_vectors.append(output_vector)

        X = build_feature_matrix(kw_candidates, inv_index, model)
        feature_matrices.append(X)

    # Merge the pandas
    X = pd.concat(feature_matrices)

    # Cast the output vector to numpy
    y = np.concatenate(output_vectors)

    return X, y


def build_feature_matrix(candidates, inv_index, model):
    """
    Extract keyword & document features and build a feature matrix
    :param candidates: list of KeywordToken candidates
    :param inv_index: InvertedIndex object
    :param model: LearningModel object

    :return: pandas DataFrame
    """
    x_matrix = preallocate_feature_matrix(len(candidates))

    extract_keyword_features(
        candidates,
        x_matrix,
        inv_index,
        model,
    )

    extract_document_features(inv_index, x_matrix)

    return rebuild_feature_matrix(x_matrix)

