#!/usr/bin/env bash
set -euo pipefail

# This is a minimal use case of fine-tuning with CDPG
# You should check other hyperparameters in the training file
SEED="0"

DOMAIN="it"
SRC_LANG="en"
TGT_LANG="de"
TOP_P="0.5"
TEMPERATURE="1.0"

DATASET_PATH="data/en-de"
TRAINING_DIR="${DATASET_PATH}/dev"

DEV0="cuda:0"
DEV1="cuda:1"


python train_cdpg.py \
  --domain "${DOMAIN}" \
  --src_lang "${SRC_LANG}" \
  --tgt_lang "${TGT_LANG}" \
  --top_p "${TOP_P}" \
  --temperature "${TEMPERATURE}" \
  --seed "${SEED}" \
  --dataset_path "${DATASET_PATH}" \
  --training_dir "${TRAINING_DIR}" \
  --dev0 "${DEV0}" \
  --dev1 "${DEV1}" \