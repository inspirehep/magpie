from __future__ import division

import os
import random
import time
from collections import Counter, defaultdict

from magpie.base.document import Document
from magpie.base.ontology import OntologyFactory
from magpie.candidates import generate_keyword_candidates
from magpie.config import HEP_ONTOLOGY, HEP_TRAIN_PATH


def get_ontology(path=HEP_ONTOLOGY, recreate=False):
    """
    Load or create an ontology from a given path
    :param path: path to the ontology file
    :param recreate: flag whether to enforce recreation of the ontology

    :return: Ontology object
    """
    start_time = time.clock()
    ontology = OntologyFactory(path, recreate=recreate)
    print("Ontology loading time: {0:.2f}s".format(time.clock() - start_time))
    return ontology


def get_documents(data_dir=HEP_TRAIN_PATH, as_generator=True, shuffle=False):
    """
    Extract documents from *.txt files in a given directory
    :param data_dir: path to the directory with .txt files
    :param as_generator: flag whether to return a document generator or a list
    :param shuffle: flag whether to return the documents
    in a shuffled vs sorted order

    :return: generator or a list of Document objects
    """
    files = list({filename[:-4] for filename in os.listdir(data_dir)})
    files.sort()
    if shuffle:
        random.shuffle(files)

    generator = (Document(doc_id, os.path.join(data_dir, f + '.txt'))
                 for doc_id, f in enumerate(files))
    return generator if as_generator else list(generator)


def get_all_answers(data_dir):
    """
    Extract ground truth answers from *.key files in a given directory
    :param data_dir: path to the directory with .key files

    :return: dictionary of the form e.g. {'101231': set('key1', 'key2') etc.}
    """
    answers = dict()

    files = {filename[:-4] for filename in os.listdir(data_dir)}
    for f in files:
        answers[f] = get_answers_for_doc(f + '.key', data_dir)

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
        answers = {line.decode('utf-8').rstrip('\n') for line in f}

    return answers


def calculate_recall_for_kw_candidates(data_dir=HEP_TRAIN_PATH,
                                       recreate_ontology=False,
                                       verbose=False):
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


def calculate_keyword_distribution(data_dir=HEP_TRAIN_PATH):
    """
    Calculate the distribution of keywords in a directory. Function can be used
    to find the most frequent and not used keywords, so that the target
    vocabulary can be trimmed accordingly.
    :param data_dir: directory path with the .key files

    :return: list of KV pairs of the form (14, ['kw1', 'kw2']), which means
             that both kw1 and kw2 were keywords in 14 papers
    """
    ontology = get_ontology()
    answers = [kw for v in get_all_answers(data_dir).values() for kw in v]
    ont_answers = [ans for ans in answers if ontology.exact_match(ans)]
    counts = Counter(ont_answers)

    histogram = defaultdict(list)
    for kw, cnt in counts.iteritems():
        histogram[cnt].append(kw)

    parsed_answers = {ontology.parse_label(l) for l in counts.keys()}
    for node in ontology.graph:
        parsed = ontology.graph.node[node]['parsed']
        if parsed not in parsed_answers:
            histogram[0].append(ontology.graph.node[node]['canonical'])

    # return sorted([(k, len(v)) for k, v in histogram.iteritems()] +
    #               [(0, len(ontology.graph) - len(used_keywords))])
    return histogram


def get_coverage_ratio_for_keyword_subset(no_of_keywords, hist=None):
    """
    Compute fraction of the samples we would be able to predict, if we reduce
    the number of keywords to a certain subset of the size no_of_keywords.
    :param no_of_keywords: the number of keywords that we limit the ontology to
    :param hist: histogram of the samples.
                 Result of calculate_keyword_distribution function

    :return: number of keywords that we need to consider, coverage ratio
    """
    if not hist:
        hist = calculate_keyword_distribution()

    hist = sorted([(k, len(v)) for k, v in hist.iteritems()])

    total_shots = sum([x[0] * x[1] for x in hist])
    keywords_collected = 0
    hits_collected = 0
    for papers, kws in reversed(hist):
        hits_collected += papers * kws
        keywords_collected += kws
        if keywords_collected >= no_of_keywords:
            return keywords_collected, hits_collected / float(total_shots)

    return -1


def get_top_n_keywords(n, hist=None):
    """
    Return the n most popular keywords
    :param n: number of keywords to return
    :param hist: histogram, result of calculate_keyword_distribution() function

    :return: sorted list of strings
    """
    if not hist:
        hist = calculate_keyword_distribution()

    kw_list = sorted([(k, v) for k, v in hist.iteritems()], reverse=True)

    answer = []
    for _count, kws in kw_list:
        answer.extend(kws)
        if len(answer) >= n:
            break

    return answer[:n]

