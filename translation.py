import argparse
import os

import evaluate
import torch
from datasets import Dataset
from ftfy import fix_text
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def read_text_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as file:
        return [fix_text(line.strip()) for line in file.readlines()]


def first_existing_path(*paths: str) -> str | None:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def default_split_files(dataset_path: str, domain: str, src_lang: str, tgt_lang: str, split: str) -> tuple[str, str]:
    src_file = first_existing_path(
        os.path.join(dataset_path, domain, f"{split}.{src_lang}"),
        os.path.join(dataset_path, f"{split}.{src_lang}"),
    )
    tgt_file = first_existing_path(
        os.path.join(dataset_path, domain, f"{split}.{tgt_lang}"),
        os.path.join(dataset_path, f"{split}.{tgt_lang}"),
    )
    if src_file is None or tgt_file is None:
        raise FileNotFoundError(
            f"Could not resolve {split} files under dataset_path={dataset_path} for domain={domain}. "
            "Please pass --src_file and --tgt_file explicitly."
        )
    return src_file, tgt_file


def resolve_eval_files(args) -> tuple[str, str]:
    if args.src_file or args.tgt_file:
        if not args.src_file or not args.tgt_file:
            raise ValueError("Both --src_file and --tgt_file must be provided together.")
        return args.src_file, args.tgt_file
    return default_split_files(args.dataset_path, args.domain, args.src_lang, args.tgt_lang, args.split)


def translate_batch(example, model, device, tokenizer):
    with torch.no_grad():
        input_ids = example["input_ids"].to(device)
        attention_mask = example["attention_mask"].to(device)
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            output_scores=True,
        )
    confidences = output.sequences_scores.exp().detach().cpu()
    hypos = tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)
    return {"hypos": hypos, "confidences": confidences}


def evaluate_model(
    model,
    tokenizer,
    src_file: str,
    tgt_file: str,
    tgt_lang: str,
    device: str = "cuda:0",
    batch_size: int = 20,
    max_length: int = 128,
    metric_namespace: str = "translation",
):
    src_lines = read_text_file(src_file)
    tgt_lines = read_text_file(tgt_file)

    dataset = Dataset.from_dict({"src": src_lines, "tgt": tgt_lines})
    tokenized_dataset = dataset.map(
        lambda samples: tokenizer(
            samples["src"],
            text_target=samples["tgt"],
            max_length=max_length,
            truncation=True,
            padding=True,
        ),
        batched=True,
    )
    tokenized_dataset.set_format("torch", columns=["labels", "input_ids", "attention_mask"])

    was_training = model.training
    model.to(device)
    model.eval()
    outputs = tokenized_dataset.map(
        lambda batch: translate_batch(batch, model=model, device=device, tokenizer=tokenizer),
        batched=True,
        batch_size=batch_size,
    )
    if was_training:
        model.train()

    hypos = outputs.data["hypos"].to_pylist()
    confidences = outputs.data["confidences"].to_pylist()

    bleu_refs = [[ref] for ref in tgt_lines]
    bleu_metric = evaluate.load("sacrebleu", experiment_id=f"{metric_namespace}-{tgt_lang}")
    if tgt_lang == "zh":
        bleu = bleu_metric.compute(references=bleu_refs, predictions=hypos, tokenize="zh")
    else:
        bleu = bleu_metric.compute(references=bleu_refs, predictions=hypos)

    bertscore_metric = evaluate.load("bertscore", experiment_id=f"{metric_namespace}-{tgt_lang}")
    bertscore = bertscore_metric.compute(references=tgt_lines, predictions=hypos, lang=tgt_lang)
    precision = sum(bertscore["precision"]) / len(bertscore["precision"])
    recall = sum(bertscore["recall"]) / len(bertscore["recall"])
    f1 = sum(bertscore["f1"]) / len(bertscore["f1"])

    return {
        "confidences": float(sum(confidences) / len(confidences)),
        "bleu": round(float(bleu["score"]), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "hypos": hypos,
    }


def save_predictions(path: str, hypos: list[str]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        if hypos:
            file.write("\n".join(hypos))


def save_metrics(path: str, metrics: dict[str, float]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for key in ("confidences", "bleu", "precision", "recall", "f1"):
            file.write(f"{key}\t{metrics[key]}\n")


def parse_args():
    parser = argparse.ArgumentParser("Translate and evaluate a checkpoint on dev/test data.")
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--domain", required=True, type=str)
    parser.add_argument("--src_lang", required=True, type=str)
    parser.add_argument("--tgt_lang", required=True, type=str)
    parser.add_argument("--dataset_path", type=str, default="./data/en-de")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--src_file", type=str, default=None)
    parser.add_argument("--tgt_file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--predictions_file", type=str, default=None)
    parser.add_argument("--metrics_file", type=str, default=None)
    parser.add_argument("--metric_namespace", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    src_file, tgt_file = resolve_eval_files(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        src_file=src_file,
        tgt_file=tgt_file,
        tgt_lang=args.tgt_lang,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        metric_namespace=args.metric_namespace or f"{args.domain}-{args.split}-{args.src_lang}-{args.tgt_lang}",
    )

    print(f"split: {args.split}")
    print(f"src_file: {src_file}")
    print(f"tgt_file: {tgt_file}")
    print(f"confidences: {metrics['confidences']}")
    print(f"bleu: {metrics['bleu']}")
    print(f"precision: {metrics['precision']}")
    print(f"recall: {metrics['recall']}")
    print(f"f1: {metrics['f1']}")

    if args.predictions_file:
        save_predictions(args.predictions_file, metrics["hypos"])
    if args.metrics_file:
        save_metrics(args.metrics_file, metrics)


if __name__ == "__main__":
    main()
