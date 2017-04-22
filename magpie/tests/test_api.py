import io
import os
import unittest

# This one is hacky, but I'm too lazy to do it properly!
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'hep-categories')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

class TestAPI(unittest.TestCase):
	""" Basic integration test """
	def test_integrity(self):
		# Get them labels!
		label_file = os.path.join(PROJECT_DIR, 'data', 'hep-categories.labels')
		with io.open(label_file, 'r') as f:
			labels = {line.rstrip('\n') for line in f}

		# Run the model
		from magpie import MagpieModel
		model = MagpieModel()
		model.init_word_vectors(TRAIN_DIR, vec_dim=100)
		history = model.train(TRAIN_DIR, labels, test_dir=TEST_DIR, nb_epochs=3)
		assert history is not None

		# Do a simple prediction
		predictions = model.predict_from_text("Black holes are cool!")
		assert len(predictions) == len(labels)

		# Assert the hell out of it!
		for lab, val in predictions:
			assert lab in labels
			assert 0 <= val <= 1
