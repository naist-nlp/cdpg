from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
from ftfy import fix_text
import evaluate
import os, sys
import numpy as np

# translation by the specified model
def _read_text_(url):
    file = open(url, 'r', encoding='utf-8')
    lines = file.readlines()
    file.close()
    return [fix_text(lines[i].strip()) for i in range(len(lines))]


# params
# domain = "Science"
# experiment_id = "2"
# epoch_num = "5"
# pattern = 3
# average = 0 off; 1 on
# epoch 
device = f"cuda:0"
domain = sys.argv[1]
experiment_id = sys.argv[2]
epoch_num = sys.argv[3]
src_lang = sys.argv[4]
tgt_lang = sys.argv[5]
if experiment_id != "finetuning" and experiment_id != "pretrained" : 
    model_name = 'models_normal/models_{}_{}/{}-{}/{}-epoch'.format(src_lang, tgt_lang, domain, experiment_id, epoch_num)
elif experiment_id == "finetuning":
    model_name = f'models_finetuning/models_{src_lang}_{tgt_lang}/{domain}/checkpoint-{epoch_num}'
elif experiment_id == "pretrained":
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'


parent_path = "datasets_en_de"
if tgt_lang == "zh" or src_lang == "zh":
    parent_path = "datasets_en_zh"
src_file = "{}/{}/test.{}".format(parent_path, domain, src_lang)
tgt_file = "{}/{}/test.{}".format(parent_path, domain, tgt_lang)

data = {"src": _read_text_(src_file), "tgt": _read_text_(tgt_file)}
dataset = Dataset.from_dict(data)
# init model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to(device)


# preprocess
tokenized_dataset = dataset.map(lambda samples: tokenizer([sample for sample in samples['src']],
                                                           text_target=[sample for sample in samples['tgt']],
                                                           max_length=128,
                                                           truncation=True,
                                                           padding=True),
                                batched=True)

tokenized_dataset.set_format("torch", columns=["labels", "input_ids", "attention_mask"])
# translate
def translate(example):
    input_ids = example['input_ids'].to(device)
    attention_mask = example['attention_mask'].to(device)
    if len(input_ids.shape)==1:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
    output = model.generate(input_ids = input_ids, attention_mask = attention_mask, return_dict_in_generate=True, output_scores=True)
    confidences = output.sequences_scores.exp()
    hypos = tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)
    return {"hypos": hypos, "confidences": confidences}
outputs = tokenized_dataset.map(translate, batched=True, batch_size=20)


hypos = outputs.data["hypos"].to_pylist()
# confidence
confidences = outputs.data["confidences"].to_pylist()
print("confidences: {}".format(sum(confidences) / len(confidences)))
refs = outputs.data["tgt"].to_pylist()
for i in range(len(refs)):
    refs[i] = [refs[i]]
# sacrebleu
metrics = evaluate.load("sacrebleu", experiment_id=f"{domain}-{experiment_id}-{src_lang}-{tgt_lang}")
if tgt_lang == "zh":
    bleu = metrics.compute(references=refs, predictions=hypos, tokenize="zh")
else:
    bleu = metrics.compute(references=refs, predictions=hypos)
print("bleu:", bleu["score"])
# bertscore
metrics = evaluate.load("bertscore", experiment_id=f"{domain}-{experiment_id}-{src_lang}-{tgt_lang}")
results = metrics.compute(references=refs, predictions=hypos, lang=tgt_lang)
precision = sum(results["precision"]) / len(results["precision"])
recall = sum(results["recall"]) / len(results["recall"])
f1 = sum(results["f1"]) / len(results["f1"])
print("precision:", precision)
print("recall:", recall)
print("f1:", f1)



results = [line + '\n' for line in hypos]
results[-1] = results[-1].strip()
if experiment_id != "finetuning" and experiment_id != "pretrained" : 
    if not os.path.exists(f'outputs_normal/outputs_{src_lang}_{tgt_lang}/{domain}'):
        os.makedirs(f'outputs_normal/outputs_{src_lang}_{tgt_lang}/{domain}')
    output_file = f'outputs_normal/outputs_{src_lang}_{tgt_lang}/{domain}/{experiment_id}-{epoch_num}.{tgt_lang}'
elif experiment_id == "finetuning":
    if not os.path.exists(f'outputs_finetuning/outputs_{src_lang}_{tgt_lang}/{domain}'):
        os.makedirs(f'outputs_finetuning/outputs_{src_lang}_{tgt_lang}/{domain}')
    output_file = f'outputs_finetuning/outputs_{src_lang}_{tgt_lang}/{domain}/{experiment_id}-{epoch_num}.{tgt_lang}'
elif experiment_id == "pretrained":
    if not os.path.exists(f'gdc_results/{src_lang}-{tgt_lang}/pretrained'):
        os.makedirs(f'gdc_results/{src_lang}-{tgt_lang}/pretrained')
    output_file = f'gdc_results/{src_lang}-{tgt_lang}/pretrained/{domain}-{experiment_id}.{tgt_lang}'
tmp = open(output_file, 'w', encoding='utf-8')
tmp.writelines(results)
tmp.close()


# comet
# import subprocess
# bash_command = "comet-score -s datasets_en_de/{}/test.en -t {} --model Unbabel/wmt20-comet-qe-da".format(domain, output_file)
# process = subprocess.Popen(bash_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# output, error = process.communicate()
# last_line = output.decode().strip()[-7:]
# print("COMET:" + last_line)