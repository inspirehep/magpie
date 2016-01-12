import os
import unittest

from magpie import api
from magpie.config import HEP_TEST_PATH, MODEL_PATH


def train_model_if_necessary(model_path):
    """ If the model under a given path does not exist, create one. """
    if not os.path.exists(model_path):
        api.train(model_path=model_path, verbose=False)


class TestAPI(unittest.TestCase):
    """ Basic tests for making sure that the API works """
    def test_train(self):
        api.train(verbose=False)

    def test_test(self):
        train_model_if_necessary(MODEL_PATH)
        precision, recall, f1_score, accuracy = api.test(verbose=False)
        self.assertGreater(precision, 0)
        self.assertGreater(recall, 0)
        self.assertGreater(f1_score, 0)
        self.assertGreater(accuracy, 0)

    def test_extract(self):
        train_model_if_necessary(MODEL_PATH)
        candidates = api.extract(HEP_TEST_PATH + '/524510.txt', verbose=False)
        self.assertGreaterEqual(len(candidates), 0)

    def test_candidate_recall(self):
        recall = api.calculate_recall_for_kw_candidates(verbose=False)
        self.assertGreater(recall, 0)
