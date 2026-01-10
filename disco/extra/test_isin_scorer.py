# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import unittest

import numpy as np

from batched_isin_scorer import BatchedIsinScorer
import torch

rain = lambda s, c: "rain" in s.text
city = lambda s, c: "city" in s.text

class BatchedIsinScorerTest(unittest.TestCase):

    def test_features_and_coefficients_match(self):
        scorer = BatchedIsinScorer([0,2,4,6,8,10], [0.5, 0.25, 0.125, 0, 1, 1])
        self.assertTrue(hasattr(scorer, "features"),
            "the exponential scorer should have a features attribute.")

    def test_features_and_coefficients_mismatch(self):
        with self.assertRaises(TypeError) as cm:
            BatchedIsinScorer(
                [0],
                [0.5, 0.25]
            )

    def test_coefficients_as_tensor_like(self):
        with self.assertRaises(TypeError) as cm:
            BatchedIsinScorer(
                [0],
                0.5
            )
            
        with self.assertRaises(TypeError) as cm:
            BatchedIsinScorer(
                0,
                [0.5]
            )
            
    def test_features_only(self):
        scorer = BatchedIsinScorer([0,2,4,6,8,10])
            
            
    def test_score(self):
        scorer = BatchedIsinScorer([0,2,4,6,8,10])
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
        #print(samples)
        scores = scorer.score(samples, None).T
        #print(samples)
        #print(scores)
        self.assertEqual(len(samples), len(scores),
            "there should be a score for each sample.")
        scores = scorer.log_score(samples, None).T
        self.assertEqual(len(samples), len(scores),
            "there should be a (log-)score for each sample.")
        scores = scorer.score(samples, None)
        true_scores = torch.Tensor([[1., 1., 1., 0., 0., 0.],
                                    [0., 0., 0., 1., 1., 0.],
                                    [0., 0., 0., 0., 0., 1.],
                                    [0., 1., 0., 0., 0., 0.]]).T
        #print(true_scores)
        #print(scores)
        assert torch.equal(true_scores, scores)


if __name__ == '__main__':
    unittest.main()
