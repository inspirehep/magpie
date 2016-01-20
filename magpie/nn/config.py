import os

# Number of tokens to save from the abstract, zero padded
SAMPLE_LENGTH = 200

# Number of keywords we are trying to predict, length of the output vector
OUTPUT_UNITS = 100

# Training parameters
BATCH_SIZE = 64
NB_EPOCHS = 1000

WORD2VEC_MODELPATH = os.path.join(os.environ['HOME'], 'word2vec_gensim_model')
