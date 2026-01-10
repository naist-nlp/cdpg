from disco.distributions import LMDistribution
import torch
import math
from disco.extra.batched_exponential_scorer import BatchedExponentialScorer #変更点
from disco.distributions.single_context_distribution import SingleContextDistribution
from disco.samplers.accumulation_sampler import AccumulationSampler
from disco.utils.device import get_device
from disco.utils.helpers import batchify
from disco.utils.moving_average import MovingAverage
from tqdm.autonotebook import trange

#少し全体としての改修コストはかかるかもしれないけど、
#1.BatchedExponentialScorerはそのままにして、isinでどこまで対応できるかやってみる。
#2.結局これはexponential scorerの値を修正しているだけなので、それで問題ない。
#3.複数要素については一旦考えないこととする。
#4.batched専用なので、error判定については考慮しなくても良い。
#5.featureも何もかも全部torch.Tensor形式で受け取ることとする。
#6.いや、やっぱりBatchedIsinScorerは必要。
#7.それを受け取って、ここの内部でexponentil scorerにするので。
#8.だけどそれにはそんなにコストは掛からないはず。tfidf部分を切り出せばいいだけ。

#じゃあ最初に入力してあげるのはIsinvector。重みについては考慮しなくていい。
#それを引数とするようにexponential scorerの方でも修正してあげる。
#これでそしたらうまくいくはず。


class LearningRateScheduler:
    #とりあえず、突貫工事でスケジューラ作った。
    #更新されなくなったらlrを下げるだけなので、もっと別ないい方法がありそう。
    def __init__(self,
                 initial_learning_rate = 0.2,
                 reduction_factor = 0.99,
                 improvement_tolerance = 0.001,
                 patience_epochs = 10):
        self.learning_rate = initial_learning_rate
        self.reduction_factor = reduction_factor
        self.improvement_tolerance = improvement_tolerance
        self.patience_epochs = patience_epochs
        self.min_err = float('inf')
        self.last_improved_epoch = 0

    def update_learning_rate(self, err, current_epoch = None):
        if self.improvement_tolerance < self.min_err - err:
            self.min_err = err
            self.last_improved_epoch = current_epoch
        elif current_epoch - self.last_improved_epoch >= self.patience_epochs:
            self.learning_rate *= self.reduction_factor
            self.last_improved_epoch = current_epoch
        return self.learning_rate




class BatchedLMDistribution(LMDistribution):
    '''
    ほとんどLMDistributionと挙動は一緒だけど、
    constrainの部分に関して書き換えを行う。
    具体的にはExponentialScorerをBatchedExponentialScorerに書き換えるだけ。
    constrainのコード自体はbase_modelのほうに存在している。
    '''

    #Copied from base_distribution.py
    def constrain(self,
            features, moments=None,
            proposal=None, context_distribution=SingleContextDistribution(''), context_sampling_size=1,
            n_samples=2**9, iterations=1000, learning_rate=0.05, tolerance=1e-5, sampling_size=2**5,
            reduction_factor=0.99, improvement_tolerance=0.001, patience_epochs=10 #追記部分
        ):
        """
        Constrains features to the base according to their moments,
        so producing an EBM

        Parameters
        ----------
        features: list(feature)
            multiple features to constrain
        moments: list(float)
            moments for the features. There should be as many moments as there are features
        proposal: distribution
            distribution to sample from, if different from self
        context_distribution: distribution
            to contextualize the sampling and scoring
        context_sampling_size:
            size of the batch when sampling context
        n_samples: int
            number of samples to use to fit the coefficients
        learning_rate: float
            multipliers of the delta used when fitting the coefficients
        tolerance: float
            accepted difference between the targets and moments
        sampling_size:
            size of the batch when sampling samples

        Returns
        -------
        exponential scorer with fitted coefficients
        """
        #仕様として、やっぱりList形式で受け取る必要がある。
        #内部がtensorだとしても、一旦listで受け取っておいて、それをあとで修正するほうが味が良い。
        #つまりこの関数内部で行うことは、featureをbatched booleanに変更して、momentsをそれぞれ切り分けるってかんじかな？

        #改修がめんどいので、featuresはいままで通りlistで受け取ることにした。
        #訂正。両方受け取ることにした。
        #たぶん何も考えなくてもうまく動作するはず
        #if list != type(features):
        #    raise TypeError("features should be passed as a list.")

        if 'Batch' not in str(type(features)):
            raise TypeError("This method for dense features. If you want to use sentence by sentence method, plese use LMdistribution")
        
        #if not moments:
        ##    #ちょっとここの値が怪しいけど、一旦ペンディング。
        #    #batchにしたとき、計算大丈夫かな...ちょっと入り組んでるのでパス
        #    return Product(self, *features)

        #if list != type(moments):
        #    raise TypeError("moments should be passed as a list.")

        #ここ以外は一旦全部削除。
        #とりあえず、featureの形を変形させた。lenで数えられるならそっちを採用
        #if not sum([len(x) if hasattr(x, "__len__") else 1 for x in features]) == len(moments): #ここの部分もそれぞれ変更すればいいからまあええか。
        #    raise TypeError("there should be as many as many moments as there are features.")
        if not len(features) ==len(moments): #もう一時的にこうした。一旦混合はパス。torch.stack関連と、Productの部分を変更すればうまいこと動く。
            raise TypeError("there should be as many as many moments as there are features.") 

        #一旦個々の要素も削除。もう考えるのがめんどくさいので。
        #if all([BooleanScorer == type(f) for f in features])\
        #    and all([1.0 == float(m) for m in moments]):
        #    return Product(self, *features)
        
        if not proposal:
            proposal = self

        context_samples, context_log_scores = context_distribution.sample(context_sampling_size)

        proposal_samples = dict()
        proposal_log_scores = dict()
        joint_log_scores = dict()
        feature_scores = dict()
        for (context, log_score) in zip(context_samples, context_log_scores):
            accumulator = AccumulationSampler(proposal, total_size=n_samples)
            proposal_samples[context], proposal_log_scores[context] = accumulator.sample(
                    sampling_size=sampling_size, context=context
                )
            device = get_device(proposal_log_scores[context])
            reference_log_scores = batchify(
                    self.log_score, sampling_size, samples=proposal_samples[context], context=context
                ).to(device)
            joint_log_scores[context] = torch.tensor(log_score).repeat(n_samples).to(device) + reference_log_scores
            #ここからの部分でfeatureが使用されている。
            #ここもbatch処理になっているはず。
            #stackされる要素は１個のみかも。ここの構造どうなっているのか後で確認。
            #feature_scores[context] = torch.stack(
            #        ([f.score(proposal_samples[context], context).to(device) for f in features])
            #    )
            ##[5,2,5] -> [10,5]のように、複数featuresが帰ってきた場合に対応するため。
            #feature_scores[context] = feature_scores[context].view(-1, feature_scores[context].shape[-1])
            #大幅に変更する。
            #これだけで、[n_features, n_samples]のベクトルが格納されるはず。
            feature_scores[context] = features.score(proposal_samples[context], context).to(device)

        #ここの部分でmomentsを計算するけど、ちょっとややこしくなるので、注意深く見ていく。
        coefficients = torch.tensor(0.0).repeat(len(features)).to(device)
        targets = torch.tensor(moments).to(device)

        #ここ追記部分。
        scheduler = LearningRateScheduler(initial_learning_rate=learning_rate, 
                                          reduction_factor= reduction_factor, 
                                          improvement_tolerance=improvement_tolerance, 
                                          patience_epochs=patience_epochs)
        
        with trange(iterations, desc='fitting exponential scorer') as t:
            for i in t:
                scorer = BatchedExponentialScorer(features, coefficients) #ここも変更点
                numerator = torch.tensor(0.0).repeat(len(features)).to(device)
                denominator = torch.tensor(0.0).repeat(len(features)).to(device)
                for context in context_samples:
                    target_log_scores = joint_log_scores[context] + scorer.log_score(
                            proposal_samples[context], context
                        ).to(device)
                    importance_ratios = torch.exp(target_log_scores - proposal_log_scores[context])
                    numerator += (importance_ratios * feature_scores[context]).sum(dim=1)
                    denominator += importance_ratios.sum()
                
                moments = numerator / denominator
                grad_coefficients = moments - targets
                err = grad_coefficients.abs().max().item()
                t.set_postfix(lr = learning_rate, err=err)
                if tolerance > err:
                    t.total_size = i
                    t.refresh()
                    break
                coefficients -= learning_rate * grad_coefficients
                #print(coefficients)OD
                #print(err)
                #突貫工事だけど、一応作った。
                #これでうまいこと学習が進まなかったらlrを下げてくれるはず...
                learning_rate = scheduler.update_learning_rate(err, i)
                if math.isnan(err):
                    raise ValueError("Detect: the coefficients is nan. That means failed to constrain. This error is to do fix. please set the other value")
        #TODO: ここの値をloggingしてplotできるようにする。
        #完全に値が収束しきらないので、そこのスコアをどうするか考える必要がある。
        #もう1個TODOなんだけど、いまlerning_rateが一定なので、ここを可変にしてあげたらすぐ収束するかも。
        #optimizerをどうやって入れるのかを考えるとき。
        print(f'The final loss is {err}') #一応、ログを吐き出しておく。
        return self * BatchedExponentialScorer(features, coefficients)
