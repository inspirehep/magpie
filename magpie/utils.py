from __future__ import division

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import random
from collections import Counter, defaultdict

from magpie.base.document import Document


def save_to_disk(path_to_disk, obj, overwrite=False):
    """ Pickle an object to disk """
    dirname = os.path.dirname(path_to_disk)
    if not os.path.exists(dirname):
        raise ValueError("Path " + dirname + " does not exist")

    if not overwrite and os.path.exists(path_to_disk):
        raise ValueError("File " + path_to_disk + "already exists")

    pickle.dump(obj, open(path_to_disk, 'wb'))


def load_from_disk(path_to_disk):
    """ Load a pickle from disk to memory """
    if not os.path.exists(path_to_disk):
        raise ValueError("File " + path_to_disk + " does not exist")

    return pickle.load(open(path_to_disk, 'rb'))


def get_documents(data_dir, as_generator=True, shuffle=False):
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


def get_all_answers(data_dir, filtered_by=None):
    """
    Extract ground truth answers from *.key files in a given directory
    :param data_dir: path to the directory with .key files
    :param filtered_by: whether to filter the answers. Both sets and ontologies
           can be passed as filters

    :return: dictionary of the form e.g. {'101231': set('key1', 'key2') etc.}
    """
    answers = dict()

    files = {filename[:-4] for filename in os.listdir(data_dir)}
    for f in files:
        answers[f] = get_answers_for_doc(f + '.txt',
                                         data_dir,
                                         filtered_by=filtered_by)

    return answers


def get_answers_for_doc(doc_name, data_dir, filtered_by=None):
    """
    Read ground_truth answers from a .key file corresponding to the doc_name
    :param doc_name: the name of the document, should end with .txt
    :param data_dir: directory in which the documents and answer files are
    :param filtered_by: whether to filter the answers. Both sets and ontologies
           can be passed as filters

    :return: set of unicodes containing answers for this particular document
    """
    filename = os.path.join(data_dir, doc_name[:-4] + '.lab')

    if not os.path.exists(filename):
        raise ValueError("Answer file " + filename + " does not exist")

    with open(filename, 'rb') as f:
        answers = {line.decode('utf-8').rstrip('\n') for line in f}

    if filtered_by:
        answers = {kw for kw in answers if kw in filtered_by}

    return answers


def calculate_keyword_distribution(data_dir, filtered_by=None):
    """
    Calculate the distribution of keywords in a directory. Function can be used
    to find the most frequent and not used keywords, so that the target
    vocabulary can be trimmed accordingly.
    :param data_dir: directory path with the .key files
    :param filtered_by: a set of keywords that defines the vocabulary.
                        Can also be an Ontology object

    :return: list of KV pairs of the form (14, ['kw1', 'kw2']), which means
             that both kw1 and kw2 were keywords in 14 papers
    """
    answers = [kw for v in get_all_answers(data_dir, filtered_by=filtered_by).values()
               for kw in v]
    counts = Counter(answers)

    histogram = defaultdict(list)
    for kw, cnt in counts.iteritems():
        histogram[cnt].append(kw)

    # Add terms that don't occur at all in the corpus
    # parsed_answers = {ontology.parse_label(l) for l in counts.keys()}
    # for node in ontology.graph:
    #     parsed = ontology.graph.node[node]['parsed']
    #     if parsed not in parsed_answers:
    #         histogram[0].append(ontology.graph.node[node]['canonical'])

    # return sorted([(k, len(v)) for k, v in histogram.iteritems()] +
    #               [(0, len(ontology.graph) - len(used_keywords))])
    return histogram


def calculate_number_of_keywords_distribution(data_dir, filtered_by=None):
    """ Look how many papers are there with 3 keywords, 4 keywords etc.
     Return a histogram. """
    answers = get_all_answers(data_dir, filtered_by=filtered_by).values()
    lengths = [len(ans_set) for ans_set in answers]
    return Counter(lengths).items()


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
