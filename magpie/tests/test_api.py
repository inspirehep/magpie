# import os
# import unittest
# from magpie.linear_classifier.config import MODEL_PATH
# from magpie.linear_classifier import api
# from magpie.linear_classifier.utils import calculate_recall_for_kw_candidates
#
#
# def train_model_if_necessary(model_path):
#     """ If the model under a given path does not exist, create one. """
#     if not os.path.exists(model_path):
#         api.train(model_path=model_path, word2vec_path=None, verbose=False)
#
#
# class TestAPI(unittest.TestCase):
#     """ Basic tests for making sure that the API works """
#     def test_train(self):
#         api.train(verbose=False, word2vec_path=None)
#
#     def test_test(self):
#         train_model_if_necessary(MODEL_PATH)
#         metrics = api.test(verbose=False)
#         self.assertGreater(metrics['map'], 0)
#         self.assertGreater(metrics['mrr'], 0)
#         self.assertGreater(metrics['ndcg'], 0)
#         self.assertGreater(metrics['p_at_3'], 0)
#         self.assertGreater(metrics['p_at_5'], 0)
#
#     def test_candidate_recall(self):
#         recall = calculate_recall_for_kw_candidates(verbose=False)
#         self.assertGreater(recall, 0)
