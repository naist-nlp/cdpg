# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch
import numpy as np
from random import sample
from ftfy import fix_text

from .distribution import Distribution

class ContextDistribution(Distribution):
    """
    Context distribution class, fetching the contexts from a text file.
    It can be used as a template for other context distributions.
    """

    def __init__(self, path="contexts.txt", prefix = ''):
        """
        Parameters
        ----------
        path: string
            path to context file
        prefix: string
            Attach Prompt
        """

        try:
            if type(path) == list:
                 tmp = []
                 for p in path:
                     with open(p) as f:
                         tmp.extend(list(map(lambda l: fix_text(l.strip()), f.readlines())))
                 self.contexts = tmp
            else:
                 with open(path) as f:
                     self.contexts = list(map(lambda l: fix_text(l.strip()), f.readlines()))
                 
            
        except IOError:
                self.contexts = list()

        assert self.contexts, "there's an issue with the context file provided."

        self.prefix = prefix
        
    def log_score(self, contexts):
        """Computes log-probabilities of the contexts

        Parameters
        ----------
        contexts: list(str)
            list of contexts to (log-)score

        Returns
        -------
        tensor of logprobabilities
        """

        assert contexts, "there needs to be contexts to (log-)score."

        n_contexts = len(contexts)
        return torch.tensor(
                [np.log(self.contexts.count(context) / n_contexts) if context in self.contexts\
                    else -float("inf")\
                    for context in contexts
                ]
            )

    def sample(self, sampling_size=32):
        """Samples random elements from the list of contexts
        
        Parameters
        ----------
        sampling_size: int
            number of contexts to sample
        
        Returns
        -------
        tuple of (list of texts, tensor of logprobs)
        """
    
        assert len(self.contexts) >= sampling_size, "the contexts does not have enough elements to sample."
        contexts = [self.prefix + c for c in sample(self.contexts, sampling_size)]
        return (contexts, self.log_score(contexts))
