import time

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
