from __future__ import division
import os
from candidates import generate_keyword_candidates
from magpie.base.document import Document
from magpie.base.ontology import OntologyFactory
from magpie.config import ONTOLOGY_DIR, ROOT_DIR

HEP_ONTOLOGY = os.path.join(ONTOLOGY_DIR, 'HEPontCore.rdf')


def calculate_recall_for_kw_candidates():
    average_recall = 0
    ontology = OntologyFactory(HEP_ONTOLOGY)
    hep_data_path = os.path.join(ROOT_DIR, 'data', 'hep')
    files = {fname[:-4] for fname in os.listdir(hep_data_path)}
    for f in files:
        document = Document(os.path.join(hep_data_path, f + '.txt'))
        kw_candidates = {kw.get_canonical_form() for kw
                         in generate_keyword_candidates(document, ontology)}
        with open(os.path.join(hep_data_path, f + '.key'), 'rb') as answer_file:
            answers = {line.rstrip('\n') for line in answer_file}

        print(document)

        print("Candidates:")
        for kw in kw_candidates:
            print("\t" + kw)
        print
        print("Answers:")
        for kw in answers:
            print("\t" + kw)
        print
        print("Conjunction:")
        for kw in kw_candidates & answers:
            print("\t" + kw)
        print

        recall = len(kw_candidates & answers) / (len(answers))
        print("Recall: " + str(recall))

        average_recall += recall

    average_recall /= len(files)

    print
    print("Averaged recall: " + str(average_recall))
