from candidates import generate_keyword_candidates
from magpie.base.document import Document
from magpie.base.ontology import Ontology
from os import path

ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
HEP_ONTOLOGY = path.join(ROOT_DIR, 'data', 'ontologies', 'HEPontCore.rdf')


def run():
    ontology = Ontology(HEP_ONTOLOGY)
    document = Document(path.join(ROOT_DIR, 'data', 'hep', '1003196.txt'))
    print document.text
    kw_candidates = generate_keyword_candidates(document, ontology)
    for token in kw_candidates:
        print token.get_value(),\
            len(token.get_all_occurrences()),\
            token.get_all_occurrences()
