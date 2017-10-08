import io
import os
import unittest

from magpie import Magpie

# This one is hacky, but I'm too lazy to do it properly!
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'hep-categories')

class TestAPI(unittest.TestCase):
	""" Basic integration test """
	def test_cnn_train(self):
		# Get them labels!
		with io.open(DATA_DIR + '.labels', 'r') as f:
			labels = {line.rstrip('\n') for line in f}

		# Run the model
		model = Magpie()
		model.init_word_vectors(DATA_DIR, vec_dim=100)
		history = model.train(DATA_DIR, labels, nn_model='cnn', test_ratio=0.3, epochs=3)
		assert history is not None

		# Do a simple prediction
		predictions = model.predict_from_text("Black holes are cool!")
		assert len(predictions) == len(labels)

		# Assert the hell out of it!
		for lab, val in predictions:
			assert lab in labels
			assert 0 <= val <= 1

	def test_rnn_batch_train(self):
		# Get them labels!
		with io.open(DATA_DIR + '.labels', 'r') as f:
			labels = {line.rstrip('\n') for line in f}

		# Run the model
		model = Magpie()
		model.init_word_vectors(DATA_DIR, vec_dim=100)
		history = model.batch_train(DATA_DIR, labels, nn_model='rnn', epochs=3)
		assert history is not None

		# Do a simple prediction
		predictions = model.predict_from_text("Black holes are cool!")
		assert len(predictions) == len(labels)

		# Assert the hell out of it!
		for lab, val in predictions:
			assert lab in labels
			assert 0 <= val <= 1
