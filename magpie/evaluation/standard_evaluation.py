from __future__ import division

import numpy as np

from magpie.evaluation.rank_metrics import mean_average_precision, \
    mean_reciprocal_rank, ndcg_at_k, r_precision, precision_at_k
from magpie.misc.considered_keywords import get_considered_keywords


def evaluate_results(kw_conf, kw_vector, gt_answers):
    """
    Compute basic evaluation ranking metrics and return them
    :param kw_conf: vector with confidence levels, return by the LearningModel
    :param kw_vector: vector with tuples (doc_id:int, kw:unicode)
    :param gt_answers: dictionary of the form dict(doc_id:int=kws:set(unicode))

    :return: dictionary with basic metrics
    """
    y_true, y_pred = build_result_matrices(kw_conf, kw_vector, gt_answers)

    y_pred = np.fliplr(y_pred.argsort())
    for i in xrange(len(y_true)):
        y_pred[i] = y_true[i][y_pred[i]]

    return {
        'map': mean_average_precision(y_pred),
        'mrr': mean_reciprocal_rank(y_pred),
        'ndcg': np.mean([ndcg_at_k(row, len(row)) for row in y_pred]),
        'r_prec': np.mean([r_precision(row) for row in y_pred]),
        'p_at_3': np.mean([precision_at_k(row, 3) for row in y_pred]),
        'p_at_5': np.mean([precision_at_k(row, 5) for row in y_pred]),
    }


def build_result_matrices(kw_conf, kw_vector, gt_answers):
    """
    Build result matrices from dict with answers and candidate vector.
    :param kw_conf: vector with confidence levels, return by the LearningModel
    :param kw_vector: vector with tuples (doc_id:int, kw:unicode)
    :param gt_answers: dictionary of the form dict(doc_id:int=kws:set(unicode))

    :return: y_true, y_pred numpy arrays
    """
    keywords = get_considered_keywords()
    keyword_indices = {kw: i for i, kw in enumerate(keywords)}
    min_docid = min(gt_answers.keys())

    y_true = np.zeros((len(gt_answers), len(keywords)), dtype=np.bool_)
    y_pred = np.zeros((len(gt_answers), len(keywords)))

    for doc_id, answers in gt_answers.iteritems():
        for kw in answers:
            if kw in keyword_indices:
                index = keyword_indices[kw]
                y_true[doc_id - min_docid][index] = True

    for conf, (doc_id, kw) in zip(kw_conf, kw_vector):
        if kw in keyword_indices:
            index = keyword_indices[kw]
            y_pred[doc_id - min_docid][index] = conf

    return y_true, y_pred
