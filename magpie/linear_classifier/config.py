import os

from magpie.config import ROOT_DIR, CORPUS_DIR

ONTOLOGY_DIR = os.path.join(ROOT_DIR, 'data', 'ontologies')
ONTOLOGY_PATH = os.path.join(ONTOLOGY_DIR, 'HEPont.rdf')
MODEL_PATH = os.path.join(CORPUS_DIR, 'model.pickle')

NO_OF_LABELS = 10000
