import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path where the model should be pickled
MODEL_PATH = os.path.join(ROOT_DIR, 'data', 'hep', 'model.pickle')

# Ontology related
ONTOLOGY_DIR = os.path.join(ROOT_DIR, 'data', 'ontologies')
HEP_ONTOLOGY = os.path.join(ONTOLOGY_DIR, 'HEPont.rdf')

# Train and test data directories
HEP_TRAIN_PATH = os.path.join(ROOT_DIR, 'data', 'hep', 'train')
HEP_TEST_PATH = os.path.join(ROOT_DIR, 'data', 'hep', 'test')

# word2vec model path
WORD2VEC_MODELPATH = os.path.join(os.environ['HOME'], 'word2vec_gensim_model')

# Training parameters
BATCH_SIZE = 64
NB_EPOCHS = 1

# Number of top N keywords that we consider for prediction
CONSIDERED_KEYWORDS = 100
