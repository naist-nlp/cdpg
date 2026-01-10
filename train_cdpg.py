import os
import random
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer

from disco.distributions import LMDistribution, ContextDistribution
from disco.tuners import CDPGTuner
from disco.extra.batched_isin_scorer import BatchedIsinScorer
from disco.extra.batched_lm_distributed import BatchedLMDistribution
from disco.extra.vectorizer import CountVectorizer

# all elements regarding wandb is disable.
# please set up it by yourself.
os.environ["WANDB_DISABLED"] = "True"
# an example
# from disco.tuners.loggers.wandb import WandBLogger
# wandb_logger = WandBLogger(tuner, project_name, entity=YOUR_ENTITY)

def build_customize_pattern(pattern: int) -> str:
    # e.g. pattern=3 -> (?u)\b\w\w\w+\b
    return r"(?u)\b" + (r"\w" * pattern) + r"+\b"


def list_training_files(training_dir: str, src_lang: str):
    """List all files under training_dir ending with .{src_lang} (sorted)."""
    suffix = f".{src_lang}"
    files = [
        os.path.join(training_dir, fn)
        for fn in os.listdir(training_dir)
        if fn.endswith(suffix) and os.path.isfile(os.path.join(training_dir, fn))
    ]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No training files ending with '{suffix}' under: {training_dir}")
    return files


def get_distribution(dev_file: str, tokenizer, customize_pattern: str):
    """
    From (avoid tokens + vectorizer on dev_file) -> (features, moments).
    Keeps your original token filtering behavior.
    """
    avoid = []
    avoid.append(tokenizer.convert_tokens_to_ids(tokenizer.eos_token))
    avoid.append(tokenizer.convert_tokens_to_ids(tokenizer.unk_token))
    avoid.append(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
    avoid.extend(tokenizer.convert_tokens_to_ids(tokenizer.additional_special_tokens))

    with tokenizer.as_target_tokenizer():
        avoid.extend(tokenizer.convert_tokens_to_ids([
            "<unk>", '▁“', '”', '、', ',', ';', ".", ',', '▁', '?', '▁"', '▁(', ')',
            '。', ')。', '”。', ')。', '。”', '-', '!"', '!', '▁!', '▁.',
        ]))

    def tokenize(x):
        with tokenizer.as_target_tokenizer():
            input_ids = tokenizer(x, add_special_tokens=False).input_ids
        return [' '.join([str(s)]) for line in input_ids for s in line if s not in avoid]

    # If you use word probability, norm=True
    vectorizer = CountVectorizer(tokenizer=tokenize, norm=True, customize_pattern=customize_pattern)
    vectorizer.vectorize(dev_file)
    word_count = vectorizer.get_vector(topk=None)[0]

    features = torch.tensor([int(s) for s in word_count.words])
    moments = word_count.weights
    return features, moments


def set_seeds(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def parse_args():
    p = argparse.ArgumentParser("CDPG tuning launcher (disco)")

    p.add_argument("--domain", required=True, type=str)
    p.add_argument("--src_lang", required=True, type=str)
    p.add_argument("--tgt_lang", required=True, type=str)
    p.add_argument("--top_p", type=float, required=True)
    p.add_argument("--temperature", type=float, required=True)

    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dev0", type=str, default="cuda:0")
    p.add_argument("--dev1", type=str, default="cuda:1")

    # Paths
    p.add_argument("--dataset_path", type=str, default="./data/en-de")
    p.add_argument("--training_dir", type=str, default="./data/en-de/dev")
    p.add_argument("--save_dir", type=str, default=None)

    # Pattern for vectorizer
    p.add_argument("--pattern", type=int, default=3)

    return p.parse_args()


def main():
    args = parse_args()
    
    set_seeds(args.seed)

    model_name = f"Helsinki-NLP/opus-mt-{args.src_lang}-{args.tgt_lang}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # save_dir simplified
    save_dir = args.save_dir or f"./runs/{args.src_lang}-{args.tgt_lang}/{args.domain}"
    os.makedirs(save_dir, exist_ok=True)

    # distribution file
    dev_file = f"{args.dataset_path}/{args.domain}/dev.{args.tgt_lang}"
    if not os.path.exists(dev_file):
        raise FileNotFoundError(f"dev_file not found: {dev_file}")

    # training files: enumerate from a folder
    training_dir = args.training_dir or f"{args.dataset_path}/dev"
    if not os.path.isdir(training_dir):
        raise NotADirectoryError(f"training_dir not found: {training_dir}")
    training_files_path = list_training_files(training_dir, args.src_lang)

    customize_pattern = build_customize_pattern(args.pattern)
    features, moments = get_distribution(dev_file, tokenizer, customize_pattern)

    print("number of features:")
    print(len(features))

    scorers = BatchedIsinScorer(features)

    base = BatchedLMDistribution(
        network=model_name,
        tokenizer=model_name,
        nature="seq2seq",
        device=args.dev0,
        top_p=args.top_p,
        temperature=args.temperature,
    )

    dataset = ContextDistribution(path=training_files_path, prefix="")

    target = base.constrain(
        scorers,
        moments,
        n_samples=2**3,
        context_distribution=dataset,
        context_sampling_size=2**3,
        learning_rate=0.05,
        sampling_size=2**3,
    )

    model = LMDistribution(
        network=model_name,
        tokenizer=model_name,
        nature="seq2seq",
        freeze=False,
        device=args.dev1,
        top_p=args.top_p,
        temperature=args.temperature,
    )

    tuner = CDPGTuner(
        model, target,
        context_distribution=dataset,
        learning_rate=0.00002,
        context_sampling_size=2**7,
        save_checkpoint_every=1,
        save_dir=save_dir,
        n_gradient_steps=10,
        n_samples_per_step=2**7,
        sampling_size=2**7,
        scoring_size=2**7,
        dynamic_mode=False,
    )

    tuner.tune()

if __name__ == "__main__":
    main()
