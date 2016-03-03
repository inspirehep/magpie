import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# word2vec
EMBEDDING_SIZE = 100
WORD2VEC_MODELPATH = os.path.join(
    DATA_DIR,
    'w2v_models',
    'word2vec{0}_model.gensim'.format(EMBEDDING_SIZE)
)

# Models
MODEL_PATH = os.path.join(DATA_DIR, 'hep', 'model.pickle')
NN_TRAINED_MODEL = os.path.join(DATA_DIR, 'berger_cnn.trained')

# Ontology related
ONTOLOGY_DIR = os.path.join(DATA_DIR, 'ontologies')
HEP_ONTOLOGY = os.path.join(ONTOLOGY_DIR, 'HEPont.rdf')

# Scaler
SCALER_PATH = os.path.join(DATA_DIR,
                           'scalers',
                           'scaler_nn_{0}.pickle'.format(EMBEDDING_SIZE))

# Train and test data directories
HEP_TRAIN_PATH = os.path.join(DATA_DIR, 'hep', 'train')
HEP_TEST_PATH = os.path.join(DATA_DIR, 'hep', 'test')

# Training parameters
BATCH_SIZE = 64
NB_EPOCHS = 1

# Number of top N keywords that we consider for prediction
CONSIDERED_KEYWORDS = 100
