from __future__ import division
from collections import defaultdict

from magpie.base.ontology import Ontology


def evaluate_results(kw_mask, kw_vector, gt_answers):
    """
    Compute basic model evaluation metrics
    :param kw_mask: a vector of 0/1 that was predicted by a classifier
    :param kw_vector: a vector of (doc_id, KeywordToken) tuples
    that correspond to kw_mask
    :param gt_answers: dictionary containing ground truth answers for each doc

    :return: (precision, recall, accuracy) tuple, each element in (0,1)
    """

    # Create the dictionary with model predictions
    prediction_dict = defaultdict(set)
    prediction_vector = [kw_vector[i] for i in xrange(len(kw_vector))
                         if kw_mask[i] == 1]
    for doc_id, kw in prediction_vector:
        prediction_dict[doc_id].add(kw.get_parsed_form())

    # Create the dictionary with ground truth answers
    answers_dict = dict()
    for doc_id, keywords in gt_answers.iteritems():
        answers_dict[doc_id] = {Ontology.parse_label(kw) for kw in keywords}

    # Recall and precision computation
    avg_recall = avg_precision = 0

    for doc_id in gt_answers.keys():
        answers = answers_dict[doc_id]
        predictions = prediction_dict[doc_id]

        if not predictions or not answers:
            if not predictions:
                avg_precision += 1  # we assume 100% precision for no predictions

            if not answers:
                avg_recall += 1  # we assume 100% recall if there's no answers

            continue

        true_positives = predictions & answers

        avg_recall += len(true_positives) / len(answers)
        avg_precision += len(true_positives) / len(predictions)

    avg_recall /= len(gt_answers)
    avg_precision /= len(gt_answers)

    # Accuracy computation
    # TODO incorrect computation, skips ground truth kw
    # TODO that have not been even recognized as candidates
    # avg_accuracy = 0
    # for i, kw_vector_elem in enumerate(kw_vector):
    #     doc_id, kw = kw_vector_elem
    #     parsed_lab = kw.get_parsed_form()
    #
    #     true_pos = kw_mask[i] == 1 and parsed_lab in answers_dict[doc_id]
    #     true_neg = kw_mask[i] == 0 and parsed_lab not in answers_dict[doc_id]
    #
    #     if true_pos or true_neg:
    #         avg_accuracy += 1
    #
    # avg_accuracy /= len(kw_vector)

    return avg_precision, avg_recall  # , avg_accuracy
