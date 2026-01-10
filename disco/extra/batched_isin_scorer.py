from disco.scorers.boolean_scorer import BooleanScorer
from disco.distributions.lm_distribution import TextSample
import torch
import numpy as np
from typing import Union, List, Literal
from disco.utils.device import get_device, to_same_device

#スコア用関数 以前のTfidfScoring。
#重みがついているので、スコアの値は0-1ではない。
def WeightedIsinScoring(features: Union[List[float], torch.Tensor, np.ndarray], weights: Union[List[float], torch.Tensor, np.ndarray], device = 'cpu'):

    if type(features) in [list, np.ndarray]:
        features = torch.tensor(features) # n
    features = features.to(device)
    if type(weights) in [list, np.ndarray]:
        weights = torch.tensor(weights).to(device) # n

    weights = weights.to(device)
    def _scoring(s, context):
        #サブワードで考えることにするので、token_idsをみている。
        token_ids = s.token_ids
        if type(token_ids) in [list, np.ndarray]:
            token_ids = torch.tensor(token_ids).to(device)
        return weights * torch.isin(features, token_ids.to(device)).float() # n feature and m sent -> n

    return _scoring


#こちらは重みがついていない。
#そのため0or1のみでしか返してくれない。boolean scoringのようなものやな。
def IsinScoring(features: Union[List[float], torch.Tensor, np.ndarray], device = 'cpu'):

    if type(features) in [list, np.ndarray]:
        features = torch.tensor(features) # n

    if torch.Tensor != type(features):
            raise TypeError("features should come in a tensor, or a tensorable structure.")
    features = features.to(device)
    
    def _scoring(s, context):
        #サブワードで考えることにするので、token_idsをみている。
        token_ids = s.token_ids
        if type(token_ids) in [list, np.ndarray]:
            token_ids = torch.tensor(token_ids).to(device)
        return torch.isin(features, token_ids.to(device)).float() # n feature and m sent -> n
    
    return _scoring


#featureにget_feature_names()、coefficientに実際の重みを入れていく。
#Booleanでいいかもしれない。
class BatchedIsinScorer(BooleanScorer):

    def __init__(self, features: Union[List[float], torch.Tensor, np.ndarray], weights: Union[List[float], torch.Tensor, np.array] = None):
        """
        Parameters
        ----------
        features: list(Scorer)
            scoring features
        weights: list(float)
            features' weights
        """
        #predicateだけ用意しておけばいいので、これ全部流用する。
        #そして１個忘れていたけど、lenでfeatureの数を返すように修正する。
        
        self.features = features
        #これはオリジナル。tensorにしたいので。
        if type(self.features) in [list, np.ndarray]:
            self.features = torch.tensor(self.features)

        if torch.Tensor != type(self.features):
            raise TypeError("features should come in a tensor, or a tensorable structure. (BatchedExponentialScorer specific requirements")

        if weights is not None:
            if len(weights) != len(self.features):
                raise TypeError("weights and features must be same lengths")
            self.scoring_function = WeightedIsinScoring(self.features, weights, get_device(self.features))
        else:
            self.scoring_function = IsinScoring(self.features, get_device(self.features))
            
        self.predicate = self.scoring_function
        

    def score(self, samples, context):
        """Computes probabilities for samples and context
        by casting the instance's predicate, ie scoring, function

        Parameters
        ----------
        samples : list(Sample)
            samples to score, as a list
        context: text
            context used for the samples

        Returns
        -------
        tensor of (0 / 1) probabilities"""
        scores = torch.stack(
            ([self.predicate(sample, context) for sample in samples]) #順序を逆にした。
        )
        return scores.T
        

    def __len__(self):
        return len(self.features)

