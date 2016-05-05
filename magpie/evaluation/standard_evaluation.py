from __future__ import division

import heapq
from itertools import compress

import numpy as np
from sklearn.metrics import auc

from magpie.evaluation.rank_metrics import mean_average_precision, \
    mean_reciprocal_rank, ndcg_at_k, r_precision, precision_at_k
from magpie.linear_classifier.labels import get_labels


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

    return calculate_basic_metrics(y_pred)


def calculate_basic_metrics(y_pred):
    """ Calculate the basic metrics and return a dictionary with them. """

    return {
        'map': mean_average_precision(y_pred),
        'mrr': mean_reciprocal_rank(y_pred),
        'ndcg': np.mean([ndcg_at_k(row, len(row)) for row in y_pred]),
        'r_prec': np.mean([r_precision(row) for row in y_pred]),
        'p_at_3': np.mean([precision_at_k(row, 3) for row in y_pred]),
        'p_at_5': np.mean([precision_at_k(row, 5) for row in y_pred]),
    }


def descendant_pr_auc(y_true, y_pred, labels, ontology):
    """
    Compute the descendant hierarchical metric.
    :param y_true: binary matrix of ground true values (n_samples x l_labels)
    :param y_pred: non-binary matrix of label predictions (n_samples x l_labels)
    :param labels: list of l canonical labels
    :param ontology: ontology object

    :return: float
    """
    label_set = set(labels)
    label_mapping = {lab: idx for idx, lab in enumerate(labels)}

    def descendant_auc(label):
        dist = ontology.get_descendants_of_label(label, filtered_by=label_set)
        label_ids = [label_mapping[label] for label in dist.keys()]
        return label_ids

    return compute_hierarchical_metric(y_true, y_pred, labels, descendant_auc)


def ancestor_pr_auc(y_true, y_pred, labels, ontology):
    """
    Compute the ancestor hierarchical metric.
    :param y_true: binary matrix of ground true values (n_samples x l_labels)
    :param y_pred: non-binary matrix of label predictions (n_samples x l_labels)
    :param labels: list of l canonical labels
    :param ontology: ontology object

    :return: float
    """
    label_set = set(labels)
    label_mapping = {lab: idx for idx, lab in enumerate(labels)}

    def ancestor_auc(label):
        dist = ontology.get_ancestors_of_label(label, filtered_by=label_set)
        label_ids = [label_mapping[label] for label in dist.keys()]
        return label_ids

    return compute_hierarchical_metric(y_true, y_pred, labels, ancestor_auc)


def count_ones(n):
    """ Calculate the number of ones in a binary representation of a number. """
    count = 0
    while n != 0:
        n &= n - 1
        count += 1

    return count


def compute_hierarchical_metric(y_true, y_pred, labels, get_related_nodes):
    """
    Compute the area under precision/recall curve for hierarchical definitions
    of precision/recall.
    :param y_true: binary matrix of ground true values (n_samples x l_labels)
    :param y_pred: non-binary matrix of label predictions (n_samples x l_labels)
    :param labels: vector of l canonical labels
    :param get_related_nodes: function that takes a single label and returns
     a list of ids of several labels including the given one.
     The returned labels can be descendants, ancestors or similar of a given lab.

    :return: float: area under the precision/recall curve
    """
    total_samples = len(y_pred)

    # Transform the y_true matrix
    bit_sets = []
    for i in xrange(total_samples):
        bit_set = 0
        for label in compress(labels, y_true[i]):
            for label_idx in get_related_nodes(label):
                bit_set |= 1 << label_idx
        bit_sets.append(bit_set)

    y_true = bit_sets
    y_true_size = [count_ones(i) for i in y_true]

    # Fill up the priority queue
    heap = []
    for row in xrange(total_samples):
        for col in xrange(len(y_pred[0])):
            heapq.heappush(heap, (-y_pred[row][col], row, col))

    # Transform the y_pred matrix
    y_pred = [0 for _ in xrange(total_samples)]
    y_pred_size = [0 for _ in xrange(total_samples)]

    precision = np.ones(total_samples)
    recall = np.zeros(total_samples)
    precision_mean, recall_mean = 1, 0
    precision_means, recall_means = [1.0], [0.0]

    while heap:
        _, row, col = heapq.heappop(heap)

        for label_idx in get_related_nodes(labels[col]):
            new_label = 1 << label_idx
            if new_label & y_pred[row] == 0:
                y_pred[row] |= new_label
                y_pred_size[row] += 1

        intersection_size = count_ones(y_true[row] & y_pred[row])
        pred_size = y_pred_size[row]
        true_size = y_true_size[row]

        old_precision, old_recall = precision[row], recall[row]

        precision[row] = intersection_size / pred_size if pred_size else 1
        recall[row] = intersection_size / true_size if true_size else 1

        # We compute the means iteratively
        precision_mean -= (old_precision - precision[row]) / total_samples
        recall_mean -= (old_recall - recall[row]) / total_samples

        precision_means.append(precision_mean)
        recall_means.append(recall_mean)

    return auc(recall_means, precision_means)


def build_result_matrices(lab_conf, lab_vector, gt_answers):
    """
    Build result matrices from dict with answers and candidate vector.
    :param lab_conf: vector with confidence levels, return by the LearningModel
    :param lab_vector: vector with tuples (doc_id:int, lab:unicode)
    :param gt_answers: dictionary of the form dict(doc_id:int=labs:set(unicode))

    :return: y_true, y_pred numpy arrays
    """
    labels = get_labels()
    label_indices = {lab: i for i, lab in enumerate(labels)}
    min_docid = min(gt_answers.keys())

    y_true = build_y_true(gt_answers, label_indices, min_docid)

    y_pred = np.zeros((len(gt_answers), len(labels)))

    for conf, (doc_id, lab) in zip(lab_conf, lab_vector):
        if lab in label_indices:
            index = label_indices[lab]
            y_pred[doc_id - min_docid][index] = conf

    return y_true, y_pred


def build_y_true(gt_answers, label_indices, min_docid):
    """
    Built the ground truth matrix
    :param gt_answers: dictionary with answers for documents
    :param label_indices: {lab: index} dictionary
    :param min_docid: the lowest doc_id in the batch

    :return: numpy array
    """
    y_true = np.zeros((len(gt_answers), len(label_indices)), dtype=np.bool_)

    for doc_id, labels in gt_answers.iteritems():
        for lab in labels:
            if lab in label_indices:
                index = label_indices[lab]
                y_true[doc_id - min_docid][index] = True

    return y_true
