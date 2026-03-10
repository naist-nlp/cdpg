import os
import random
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from disco.distributions import LMDistribution, ContextDistribution
from disco.tuners import CDPGTuner
from disco.extra.batched_isin_scorer import BatchedIsinScorer
from disco.extra.batched_lm_distributed import BatchedLMDistribution
from disco.extra.vectorizer import CountVectorizer
from translation import evaluate_model

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


def default_model_name(src_lang: str, tgt_lang: str) -> str:
    return f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"


def first_existing_path(*paths: str) -> str | None:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def resolve_distribution_file(args) -> str:
    if args.distribution_file:
        if not os.path.exists(args.distribution_file):
            raise FileNotFoundError(f"distribution_file not found: {args.distribution_file}")
        return args.distribution_file

    candidate = first_existing_path(
        os.path.join(args.dataset_path, args.domain, f"dev.{args.tgt_lang}"),
        os.path.join(args.dataset_path, args.domain, f"test.{args.tgt_lang}"),
    )
    if candidate is None:
        raise FileNotFoundError(
            "Could not resolve a distribution file. "
            "Please pass --distribution_file explicitly."
        )
    return candidate


def resolve_supervision_files(args) -> tuple[str, str]:
    src_file = args.supervision_src_file
    tgt_file = args.supervision_tgt_file

    if src_file or tgt_file:
        if not src_file or not tgt_file:
            raise ValueError("Both --supervision_src_file and --supervision_tgt_file must be provided together.")
        if not os.path.exists(src_file):
            raise FileNotFoundError(f"supervision_src_file not found: {src_file}")
        if not os.path.exists(tgt_file):
            raise FileNotFoundError(f"supervision_tgt_file not found: {tgt_file}")
        return src_file, tgt_file

    candidate_src = first_existing_path(
        os.path.join(args.dataset_path, args.domain, f"dev.{args.src_lang}"),
        os.path.join(args.dataset_path, args.domain, f"test.{args.src_lang}"),
    )
    candidate_tgt = first_existing_path(
        os.path.join(args.dataset_path, args.domain, f"dev.{args.tgt_lang}"),
        os.path.join(args.dataset_path, args.domain, f"test.{args.tgt_lang}"),
    )
    if candidate_src is None or candidate_tgt is None:
        raise FileNotFoundError(
            "Could not resolve supervision files. "
            "Please pass --supervision_src_file and --supervision_tgt_file explicitly."
        )
    return candidate_src, candidate_tgt


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
    p.add_argument("--model_name_or_path", type=str, default=None)

    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dev0", type=str, default="cuda:0")
    p.add_argument("--dev1", type=str, default="cuda:1")

    # Paths
    p.add_argument("--dataset_path", type=str, default="./data/en-de")
    p.add_argument("--training_dir", type=str, default="./data/en-de/dev")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--distribution_file", type=str, default=None)
    p.add_argument("--supervision_src_file", type=str, default=None)
    p.add_argument("--supervision_tgt_file", type=str, default=None)

    # Pattern for vectorizer
    p.add_argument("--pattern", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=0.00002)
    p.add_argument("--n_gradient_steps", type=int, default=10)

    p.add_argument("--dynamic_mode", action="store_true")
    p.add_argument("--experiment_id", type=str, default="default")
    p.add_argument("--phase", type=int, default=0)
    p.add_argument("--threshold_update_num", type=int, default=2)
    p.add_argument("--log_file_path", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    
    set_seeds(args.seed)

    model_name = args.model_name_or_path or default_model_name(args.src_lang, args.tgt_lang)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # save_dir simplified
    save_dir = args.save_dir or f"./runs/{args.src_lang}-{args.tgt_lang}/{args.domain}"
    os.makedirs(save_dir, exist_ok=True)

    # distribution file
    distribution_file = resolve_distribution_file(args)

    # training files: enumerate from a folder
    training_dir = args.training_dir or f"{args.dataset_path}/dev"
    if not os.path.isdir(training_dir):
        raise NotADirectoryError(f"training_dir not found: {training_dir}")
    training_files_path = list_training_files(training_dir, args.src_lang)

    customize_pattern = build_customize_pattern(args.pattern)
    features, moments = get_distribution(distribution_file, tokenizer, customize_pattern)

    print("number of features:")
    print(len(features))

    base_bleu = 0.0
    base_f1 = 0.0
    supervising_file_path_dict = {}
    log_file_path = args.log_file_path or os.path.join(save_dir, "dynamic.log")

    if args.dynamic_mode:
        supervision_src_file, supervision_tgt_file = resolve_supervision_files(args)
        supervising_file_path_dict = {
            "src_path": supervision_src_file,
            "tgt_path": supervision_tgt_file,
        }
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        base_model.to(args.dev1)
        base_metrics = evaluate_model(
            model=base_model,
            tokenizer=tokenizer,
            src_file=supervision_src_file,
            tgt_file=supervision_tgt_file,
            tgt_lang=args.tgt_lang,
            device=args.dev1,
        )
        base_bleu = base_metrics["bleu"]
        base_f1 = base_metrics["f1"]
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        learning_rate=args.learning_rate,
        context_sampling_size=2**7,
        save_checkpoint_every=1,
        save_dir=save_dir,
        n_gradient_steps=args.n_gradient_steps,
        n_samples_per_step=2**7,
        sampling_size=2**7,
        scoring_size=2**7,
        dynamic_mode=args.dynamic_mode,
        domain=args.domain,
        experiment_id=args.experiment_id,
        current_phase=args.phase,
        base_bleu=base_bleu,
        base_f1=base_f1,
        threshold_update_num=args.threshold_update_num,
        max_epoch=args.n_gradient_steps,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        log_file_path=log_file_path,
        supervising_file_path_dict=supervising_file_path_dict,
    )

    tuner.tune()

if __name__ == "__main__":
    main()
