from disco.scorers.exponential_scorer import ExponentialScorer
import torch
import numpy as np
from typing import Union, List
from disco.utils.device import get_device, to_same_device
'''
Bassically, Copied from ExponentialScorer.
But, log_score is a bit change to batch processing even though apply many features
'''

# 方針（要件）
# 1. exponential scoreで直接指定するようにする。
# 2. featureとcoefficientの情報は外部から指定できるようにする。これによって、上位数件などの情報を活用することができる。
# 3. 内部的には、featureを指定することで、


#featureにget_feature_names()、coefficientに実際の重みを入れていく。
#ここで注意なんだけど、別にfeaturesはどんな型でも通ってしまう。
#なので、最悪str型でも長さがマッチしていればなんでもよいという欠点が明らかになった。
class BatchedExponentialScorer(ExponentialScorer):
        
    def log_score(self, samples, context):
        """Log-scores the samples given the context
        using the instance's features and their coefficients

        Parameters
        ----------
        samples : list(str)
            list of samples to log-score
        context: text
            context used for the samples

        Returns
        -------
        tensor of log-scores"""

        device = get_device(self.coefficients)#ここでCPUになってない？要確認
        #self.features = self.features.to(device)

        #feature_log_scores = torch.stack(
        #    ([self.predicate(sample, context).to(device) for sample in samples]) #順序を逆にした。
        #) #[n_samples, n_features]
        feature_log_scores = self.features.score(samples, context).to(device) # [n_features, n_samples]

        ###
        #そもそも、それぞれのscoringでは文全部に対してスコア付けをしていた。
        #それを変更して、一回で、スコアを文全体につける関数に修正。
        #逆に言えば、exponential scorerでやることを重み付けのみに軽減させた。
        #Tf-idfとかの値を使いたければ、isin_scorerに重みを付けてあげればいいさ。
        #ここでやることではなかった。
        #例えば、lengthを入れてあげるなら、length用のbatchをいれてあげて、ここでの改修はstackをするのみに留める。
        ###
        
        #coefficientsを1にしたらそのままtfidfの重みを扱うことができる。
        #BREAKING CHANGE!!!!!
        #[n_samples, n_features] -> [n_features, n_samples]変更。
        #理由としてはいままでが後者だったけど、それを都合で前者に変更していた。
        #contrain周りでの改修コストを考えたらこのままの方が良いと判断した。
        weighted_log_scores = self.coefficients.repeat(len(samples), 1) * feature_log_scores.T
        #print(weighted_log_scores)
        #print(weighted_log_scores.sum(dim=1))
        return weighted_log_scores.sum(dim=1)

