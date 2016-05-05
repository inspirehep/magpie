from __future__ import division
import os
import time
from collections import defaultdict

from magpie.linear_classifier.base.ontology import OntologyFactory
from magpie.linear_classifier.candidates import generate_keyword_candidates
from magpie.linear_classifier.config import ONTOLOGY_PATH
from magpie.linear_classifier.labels import get_keywords
from magpie.utils import get_documents, get_answers_for_doc


def get_ontology(path=ONTOLOGY_PATH, recreate=False, verbose=True):
    """
    Load or create an ontology from a given path
    :param path: path to the ontology file
    :param recreate: flag whether to enforce recreation of the ontology
    :param verbose: a flag whether to be verbose

    :return: Ontology object
    """
    tick = time.clock()
    ontology = OntologyFactory(path, recreate=recreate)
    if verbose:
        print("Ontology loading time: {0:.2f}s".format(time.clock() - tick))

    return ontology


def augment_the_training_dataset(data_dir, augment_method='subtree', filtered_by=None):
    """ Augment binary answer files with non-binary answers, that indicate
     partial keyword relevance to the document. Those partial relevance values
     are computed by analyzing the ontology relations between labels. """
    ontology = get_ontology()

    if not os.path.exists(data_dir):
        raise ValueError("The path to the dataset does not exist")

    if augment_method == 'exponential_decay':
        augment_answers = exponential_decay
    else:
        augment_answers = subtree_comparison

    files = {filename[:-4] for filename in os.listdir(data_dir)}
    format_line = lambda s: (s + "\n").encode('utf-8')
    lab_filter = filtered_by or ontology

    for f in files:
        answers = get_answers_for_doc(f + '.lab', data_dir, filtered_by=lab_filter)
        augmented_answers = augment_answers(answers, ontology)
        with open(os.path.join(data_dir, f + '.aug'), 'wb') as aug_file:
            for aug_ans, weight in augmented_answers.iteritems():
                line = aug_ans + u';' + unicode(weight)
                aug_file.write(format_line(line))


def subtree_comparison(answers, ontology, filtered_by=None):
    computed_metrics = defaultdict(lambda: 999999)

    for ans in answers:
        lab_ancestors = ontology.get_ancestors_of_label(ans,
                                                        filtered_by=filtered_by)
        lab_descendants = set(ontology.get_descendants_of_label(ans).keys())

        for anc in lab_ancestors.keys():
            anc_descendants = set(ontology.get_descendants_of_label(anc).keys())
            anc_descendants.add(ans)  # should never happen if ontology is good

            intersection = anc_descendants & lab_descendants

            precision = len(intersection) / len(anc_descendants)
            recall = len(intersection) / len(lab_descendants)
            f1 = 2 * precision * recall / (precision + recall)

            computed_metrics[anc] = min(computed_metrics[anc], f1)

    return computed_metrics


def exponential_decay(answers, ontology):
    aug_answers = dict()
    ancestors = dict()

    for ans in answers:
        lab_ancestors = ontology.get_ancestors_of_label(ans)
        for ancestor, distance in lab_ancestors.iteritems():
            if ancestor in ancestors:
                ancestors[ancestor] = min(ancestors[ancestor], distance)
            else:
                ancestors[ancestor] = distance

    for ancestor, distance in ancestors.iteritems():
        rel_value = 2 ** (-distance)
        if ancestor in aug_answers:
            aug_answers[ancestor] = max(aug_answers[ancestor], rel_value)
        else:
            aug_answers[ancestor] = rel_value

    return aug_answers


def calculate_recall_for_kw_candidates(data_dir, recreate_ontology=False, verbose=False):
    """
    Generate keyword candidates for files in a given directory
    and compute their recall in reference to ground truth answers
    :param data_dir: directory with .txt and .key files
    :param recreate_ontology: boolean flag for recreating the ontology
    :param verbose: whether to print computation times

    :return average_recall: float
    """
    average_recall = 0
    total_kw_number = 0

    ontology = get_ontology(recreate=recreate_ontology)
    docs = get_documents(data_dir)
    considered_keywords = set(get_keywords())
    total_docs = 0

    start_time = time.clock()
    for doc in docs:
        kw_candidates = {kw.get_canonical_form() for kw
                         in generate_keyword_candidates(doc, ontology)}

        answers = get_answers_for_doc(doc.filename, data_dir, filtered_by=considered_keywords)
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

        recall = 1 if not answers else len(kw_candidates & answers) / (len(answers))
        if verbose:
            print
            print("Paper: " + doc.filename)
            print("Candidates: " + str(len(kw_candidates)))
            print("Recall: " + unicode(recall * 100) + "%")

        average_recall += recall
        total_kw_number += len(kw_candidates)
        total_docs += 1

    average_recall /= total_docs

    if verbose:
        print
        print("Total # of keywords: " + str(total_kw_number))
        print("Time elapsed: " + str(time.clock() - start_time))

    return average_recall
