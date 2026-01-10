from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from ftfy import fix_text
from abc import ABCMeta, abstractmethod
from collections import namedtuple
import torch
from itertools import chain
from typing import Union, Literal

Vector = namedtuple('Vector', ['words', 'weights'])

#Next, implement CountVector part
class BaseVectorizer:
    '''
    The class is only considering subwords. NOT words level vectorize.
    If you considering word level tokenization, you write tokenizer function.
    The base model is not considering about stopwords.
    '''
    def __init__(self, tokenizer = lambda xs: ' '.join(chain.from_iterable([x.split() for x in xs])), vectorizer = TfidfVectorizer()):
        self.tokenizer = tokenizer #もしwordlevelとかにして、stopwordなんかまぜるならここを頑張る。
        self.vectorizer = vectorizer
        self.vectorizer.token_pattern = r"(?u)\b\w+\b"

    def tokenize(self, batch):
        return self.tokenizer(batch)
    
    def read_doc(self, path):
        with open(path, encoding='utf-8') as f:
            doc = [fix_text(line.strip()) for line in f]
        doc = self.tokenize(doc)
        return ' '.join(doc)

    def vectorize(self, path):
        if type(path) is not list:
            docs = [self.read_doc(path)]
        else:
            docs = [self.read_doc(p) for p in path]
        #print(docs[:1000])
        #print(len(docs))
        self.X = self.vectorizer.fit_transform(docs)
        self.feat_names = self.vectorizer.get_feature_names_out()

    def get_vector(self, topk = None, return_tensors:Union[None, Literal['pt', 'np']] = 'pt'):
        if topk is None:
            topk = len(self.feat_names)
        vec: Vector = self._get_vector(topk)
        
        if return_tensors is None:
            if type(vec.weights) in [np.array, torch.Tensor, np.ndarray]:
                vec = Vector(words = vec.words, weights = vec.weights.tolist())
            assert type(vec.weights) is list, f'return type is not match {type(vec.weights)}'
        elif return_tensors == 'pt':
            if type(vec.weights) in [np.array, list, np.ndarray]:
                vec = Vector(words = vec.words, weights = torch.tensor(vec.weights))
            assert type(vec.weights) is torch.Tensor, f'return type is not match {type(vec.weights)}'
        else:
            if type(vec.weights) is torch.Tensor:
                vec = Vector(words = vec.words, weights = vec.weights.numpy())
            else:
                vec = Vector(words = vec.words, weights = np.array(vec.weights))
                
            assert type(vec.weights) in [np.array, np.ndarray], f'return type is not match {type(vec.weights)}'
        #出力を１文につき１個にする。
        return_vec = []
        for i in range(len(vec.words)):
            return_vec.append(Vector(words = vec.words[i], weights = vec.weights[i]))
            
        return return_vec
    
    @abstractmethod
    def _get_vector(self, topk):
        pass
        
class TFIDFVectorizer(BaseVectorizer):

    def __init__(self, tokenizer = lambda xs: [' '.join(x.split()) for x in xs], vectorizer = TfidfVectorizer(), norm = False, stopwords=[], customize_pattern=r"(?u)\b\w\w+\b"):
        super().__init__(tokenizer, vectorizer)
        self.norm = norm
        self.stop_words = stopwords
        self.vectorizer.token_pattern = customize_pattern
    
    def _get_vector(self, topk):
        # TF-IDFスコアの降順でインデックスをソート
        sorted_indices = self.X.toarray().argsort(axis=1)[:, ::-1]
        #print(sorted_indices)
        # 上位Kの単語を取得
        top_words = [self.feat_names[idx] for idx in sorted_indices[:, :topk]]
        if self.norm:
            total = self.X.toarray().sum(axis=1, keepdims=True)
            weights = np.take_along_axis((self.X.toarray() / total), sorted_indices, axis=1)[:, :topk]
        else:
            weights = np.take_along_axis(self.X.toarray(), sorted_indices, axis=1)[:, :topk]
        return Vector(words = top_words, weights = weights)
    
class IDFVectorizer(BaseVectorizer):
    def __init__(self, tokenizer = lambda xs: [' '.join(x.split()) for x in xs], vectorizer = TfidfVectorizer()):
        super().__init__(tokenizer, vectorizer)
        
    def _get_vector(self, topk):
        idf = self.vectorizer.idf_
        sorted_indices = np.argsort(idf)#[::-1]
        top_words = [self.feat_names[idx] for idx in sorted_indices[:topk]]
        weights = idf[sorted_indices[:topk]]
        return Vector(words = top_words, weights = weights)

class CountVectorizer(BaseVectorizer):
    def __init__(self, tokenizer = lambda xs: [' '.join(x.split()) for x in xs], vectorizer = CountVectorizer(), norm = False, stopwords=[], customize_pattern=r"(?u)\b\w\w+\b"):
        super().__init__(tokenizer, vectorizer)
        self.norm = norm
        self.stop_words = stopwords
        self.vectorizer.token_pattern = customize_pattern
        
    def _get_vector(self, topk):
        sorted_indices = self.X.toarray().argsort(axis=1)[:, ::-1]
        top_words = [self.feat_names[idx] for idx in sorted_indices[:, :topk]]
        if self.norm:
            total = self.X.toarray().sum(axis=1, keepdims=True)
            weights = np.take_along_axis((self.X.toarray() / total), sorted_indices, axis=1)[:, :topk]
        else:
            weights = np.take_along_axis(self.X.toarray(), sorted_indices, axis=1)[:, :topk]
        return Vector(words = top_words, weights = weights)
    
if __name__ == '__main__':
    vectorizer = CountVectorizer(norm = False)
