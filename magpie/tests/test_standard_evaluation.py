from __future__ import division

from unittest import TestCase

from magpie import evaluate_results
from magpie.candidates.keyword_token import KeywordToken


class TestEvaluateResults(TestCase):
    def test_evaluate_results1(self):
        kw_mask = [1, 1, 1, 1]
        kw_vector = [
            (1, KeywordToken(None, parsed_label=u"lionel messi")),
            (2, KeywordToken(None, parsed_label=u"robert lewandowski")),
            (2, KeywordToken(None, parsed_label=u"karl-heinz rummenigge")),
            (3, KeywordToken(None, parsed_label=u"wayne rooney")),
        ]
        gt_answers = {
            1: {u"Lionel Messi", u"Arjen Robben"},
            2: {u"Robert Lewandowski"},
            3: {u"Pele", u"Diego Maradona"},
            4: {u"Not predicted"},
        }

        precision, recall, accuracy = evaluate_results(
            kw_mask,
            kw_vector,
            gt_answers
        )

        self.assertEqual(precision, 2.5 / 4)
        self.assertEqual(recall, 1.5 / 4)
        self.assertAlmostEqual(accuracy, 2 / 4)

    def test_evaluate_results2(self):
        kw_mask = [1, 1, 0, 1, 0, 1, 1]
        kw_vector = [
            (1, KeywordToken(None, parsed_label=u"lionel messi")),
            (1, KeywordToken(None, parsed_label=u"cristiano ronaldo")),
            (2, KeywordToken(None, parsed_label=u"robert lewandowski")),
            (2, KeywordToken(None, parsed_label=u"karl-heinz rummenigge")),
            (3, KeywordToken(None, parsed_label=u"wayne rooney")),
            (4, KeywordToken(None, parsed_label=u"neymar")),
            (4, KeywordToken(None, parsed_label=u"robin van persie")),
        ]
        gt_answers = {
            1: {u"Lionel Messi", u"Arjen Robben", u"Cristiano Ronaldo", u"Rivaldo"},
            2: {u"Robert Lewandowski", u"Karl-Heinz Rummenigge"},
            3: {u"Pele", u"Diego Maradona"},
            4: {u"Robin Van Persie"},
            5: {u"Not predicted Keyword1", u"Not predicted keyword2"},
        }

        precision, recall, accuracy = evaluate_results(
            kw_mask,
            kw_vector,
            gt_answers
        )

        self.assertEqual(precision, 4.5 / 5)
        self.assertEqual(recall, 2 / 5)
        self.assertAlmostEqual(accuracy, 5 / 7)
