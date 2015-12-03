from __future__ import division
import os
import time
from candidates import generate_keyword_candidates
from magpie.base.document import Document
from magpie.base.ontology import OntologyFactory
from magpie.config import ONTOLOGY_DIR, ROOT_DIR

HEP_ONTOLOGY = os.path.join(ONTOLOGY_DIR, 'HEPont.rdf')


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
        # print
        # print(u"Paper: " + f)
        # print(u"Candidates: " + str(len(kw_candidates)))
        # print(u"Recall: " + unicode(recall))

        average_recall += recall
        total_kw_number += len(kw_candidates)

    average_recall /= len(files)

    print
    print(u"Total # of keywords: " + str(total_kw_number))
    print(u"Averaged recall: " + unicode(average_recall))
    end_time = time.clock()
    print(u"Time elapsed: " + str(end_time - start_time))


def get_ontology():
    return OntologyFactory(HEP_ONTOLOGY)
