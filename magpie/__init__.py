from os import path
from candidates import generate_keyword_candidates
from magpie.base.document import Document
from magpie.base.ontology import OntologyFactory
from magpie.config import ONTOLOGY_DIR, ROOT_DIR

HEP_ONTOLOGY = path.join(ONTOLOGY_DIR, 'HEPontCore.rdf')


def run():
    ontology = OntologyFactory(HEP_ONTOLOGY)
    document = Document(path.join(ROOT_DIR, 'data', 'hep', '1003196.txt'))
    print document.text
    kw_candidates = generate_keyword_candidates(document, ontology)
    for token in kw_candidates:
        print token.get_value(),\
            len(token.get_all_occurrences()),\
            token.get_all_occurrences()
