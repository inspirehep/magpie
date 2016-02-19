import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# Models
MODEL_PATH = os.path.join(DATA_DIR, 'hep', 'model.pickle')
WORD2VEC_MODELPATH = os.path.join(os.environ['HOME'], 'word2vec_model.gensim')
NN_TRAINED_MODEL = os.path.join(DATA_DIR, 'berger_cnn.trained')

# Ontology related
ONTOLOGY_DIR = os.path.join(DATA_DIR, 'ontologies')
HEP_ONTOLOGY = os.path.join(ONTOLOGY_DIR, 'HEPont.rdf')

# Scaler
SCALER_PATH = os.path.join(DATA_DIR, 'scaler_nn.pickle')

# Train and test data directories
HEP_TRAIN_PATH = os.path.join(DATA_DIR, 'hep', 'train')
HEP_TEST_PATH = os.path.join(DATA_DIR, 'hep', 'test')

# Training parameters
BATCH_SIZE = 64
NB_EPOCHS = 1

# Number of top N keywords that we consider for prediction
CONSIDERED_KEYWORDS = 100
