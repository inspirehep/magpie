import os

# Number of tokens to save from the abstract, zero padded
SAMPLE_LENGTH = 200

# Number of keywords we are trying to predict, length of the output vector
OUTPUT_UNITS = 100

LOG_FOLDER = os.path.join(os.environ['HOME'], 'keras-results')
