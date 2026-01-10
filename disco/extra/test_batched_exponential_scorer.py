# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import unittest

import numpy as np

from batched_exponential_scorer import BatchedExponentialScorer
from batched_isin_scorer import BatchedIsinScorer

isin_scorer = BatchedIsinScorer([0,2,4,6,8,10])

class BatchedExponentialScorerTest(unittest.TestCase):

    def test_features_and_coefficients_match(self):
        scorer = BatchedExponentialScorer(isin_scorer, [0.5, 0.25, 0.125, 0, 1, 1])
        self.assertTrue(hasattr(scorer, "features"),
            "the exponential scorer should have a features attribute.")
        self.assertTrue(hasattr(scorer, "coefficients"),
            "the exponential scorer should have a coefficients attribute.")
        self.assertEqual(len(scorer.features), len(scorer.coefficients),
            "the length of both features and coefficients list should be equal.")

    def test_features_and_coefficients_mismatch(self):
        with self.assertRaises(ValueError) as cm:
            BatchedExponentialScorer(
                BatchedIsinScorer([0]),
                [0.5, 0.25]
            )

    def test_coefficients_as_tensor_like(self):
        with self.assertRaises(TypeError) as cm:
            BatchedExponentialScorer(
                BatchedIsinScorer([0]),
                0.5
            )
        with self.assertRaises(ValueError) as cm:
            BatchedExponentialScorer(
                BatchedIsinScorer([0,2,4,6,8]),
                {"rain": 0.5, "city": 0.25}
            )
        with self.assertRaises(ValueError) as cm:
            BatchedExponentialScorer(
                BatchedIsinScorer([0,2]),
                [0.5]
            )

    def test_score(self):
        scorer = BatchedExponentialScorer(isin_scorer, [0.5, 0.25, 0.125, 0, 1, 1])
        texts = [
                "I'm singing in the rain.",
                "What is the city but the people?",
                "The rain that fell on the city runs down the dark gutters and empties into the sea without even soaking the ground",
                "Every drop in the ocean counts."
            ]
        from collections import defaultdict
        word_to_id = defaultdict(lambda: len(word_to_id))
        for text in texts:
            for word in text.split():
                word_to_id[word]
        id_texts = [[word_to_id[word] for word in text.split()] for text in texts]
        
        
        from disco.distributions.lm_distribution import TextSample
        samples = [TextSample(id_texts[i], texts[i]) for i in range(len(texts))] # fake samples without the tokenizations
        
        scores = scorer.score(samples, None)
        print(samples)
        print(scores)
        self.assertEqual(len(samples), len(scores),
            "there should be a score for each sample.")
        log_scores = scorer.log_score(samples, None)
        print(log_scores)
        self.assertEqual(len(samples), len(log_scores),
            "there should be a (log-)score for each sample.")
        for e, s in zip([0.8750, 1.0000, 1.0000, 0.2500], log_scores):
            self.assertEqual(e, s,
                "the exponential scorer should (log-)score correctly."
            )


if __name__ == '__main__':
    unittest.main()
