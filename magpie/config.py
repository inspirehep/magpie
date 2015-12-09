from os import path

ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))

# Path where the model should be pickled
MODEL_PATH = path.join(ROOT_DIR, 'data', 'hep', 'model.pickle')

# Ontology related
ONTOLOGY_DIR = path.join(ROOT_DIR, 'data', 'ontologies')
HEP_ONTOLOGY = path.join(ONTOLOGY_DIR, 'HEPontCore.rdf')

# Train and test data directories
HEP_TRAIN_PATH = path.join(ROOT_DIR, 'data', 'hep', 'train')
HEP_TEST_PATH = path.join(ROOT_DIR, 'data', 'hep', 'test')
