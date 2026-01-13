# Domain Translation with Monolingual Lexical Distribution

**Authors**:  
Yusuke Sakai*, Zhi Qu*, Hidetaka Kamigaito, Taro Watanabe, Xiaojiang Liu  

This repository accompanies our paper:  
**Domain Translation with Monolingual Lexical Distribution**, which has been accepted by **TMLR**.

You can find our paper by this [URL](https://openreview.net/forum?id=UKLBobrFCR).

---

## Introduction

This repository serves as the public hub for releasing the **code and data** used in our work.

The fine-tuning approach proposed in the paper is a variant of energy-based models (EBMs), built upon Conditional Distributional Policy Gradients (CDPG).


## data
The task in our paper is domain adaptation in machine translation.

### English-German
The English-German data is publicly available.  
In this repository, we only provide the subsets that are used in our experiments.

If you need the full dataset, please refer to the original [project](https://github.com/roeeaharoni/unsupervised-domain-clusters).

> Note:  
> The test sets have been cleaned and corrected as described in our paper.  
> Results may therefore differ from those obtained using the original raw data.

### English-Chinese
The English-Chinese data is subject to licensing restrictions, so, please obtain it from [UM-Corpus](http://nlp2ct.cis.umac.mo/um-corpus/)

### Training Data (WMT)
As described in the paper, the monolingual data used during training is derived from the **WMT Translation Tasks**.  
The monolingual corpora used in our experiments are provided in this repository.

You can find the raw data from [WMT23](https://www2.statmt.org/wmt23/).

## Codes

### Core Library Versions
The experiments were conducted with the following core dependencies:

```text
Python        3.10
datasets      3.0.0
torch         2.5.1+cu121
transformers  4.35.0
```

### Toolkit of CDPG
Our tech is based on the toolkit released by NAVER, named disco.
We have a lot of updates, including but not limited to
```text
./disco/extra/
./disco/distributions/lm_distribution.py
...
```

You can explore the original project [disco](https://github.com/naver/disco).

> Note:  
> Please be careful on ./disco/distributions/lm_distribution.py,  
> the codes related to setting pad_token_id is fixed in original version.  
> Given there are various models are usable, we follow the original logic to add and fix codes for translation models.

you can install the toolkit by
```
cd ./disco
pip install -e .
```

### Usage
You can run below script as a minimal use case, which shows a CDPG with fixed hyperparameters.

```
bash run.sh
```

Dynamic CDPG is coming soon.


