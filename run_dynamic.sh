#!/usr/bin/env bash
set -euo pipefail

# Dynamic CDPG search:
# 1. train one phase with a (temperature, top_p) pair
# 2. accept the phase only if dev performance improves often enough
# 3. resume from the accepted best checkpoint, otherwise keep the previous one

EXPERIMENT_ID="0"
DOMAIN="it"
SRC_LANG="en"
TGT_LANG="de"
SEED="0"

N_GRADIENT_STEPS="10"
LEARNING_RATE="0.00002"
THRESHOLD_UPDATE_NUM="3"
LR_DESCENT_RATIO="sqrt($N_GRADIENT_STEPS)"

DATASET_PATH="data/en-de"
TRAINING_DIR="${DATASET_PATH}/dev"

# Dynamic mode needs an explicit supervision set.
# Adjust these two paths to the dev pair you want to trust for accept/reject.
SUPERVISION_SRC_FILE="${DATASET_PATH}/dev/wmttest2023.${SRC_LANG}"
SUPERVISION_TGT_FILE="${DATASET_PATH}/dev/wmttest2023.${TGT_LANG}"

DEV0="cuda:0"
DEV1="cuda:1"
EVAL_DEVICE="cuda:0"

# Keep this enabled if you still want per-checkpoint outputs for notebook-based comparison.
EVALUATE_ALL_CHECKPOINTS="false"

LOWER_GRID=(0.5 0.4 0.3 0.2 0.1 0.0)
UPPER_GRID=(0.6 0.7 0.8 0.9 1.0 1.1)

RUN_ROOT="runs_dynamic/${SRC_LANG}-${TGT_LANG}/${DOMAIN}"
PHASE_DIR="${RUN_ROOT}/phases"
CHECK_DIR="${RUN_ROOT}/checks"
OUTPUT_DIR="${RUN_ROOT}/translations"
SUMMARY_FILE="${RUN_ROOT}/summary.tsv"
LOG_FILE="${CHECK_DIR}/check_${DOMAIN}_${EXPERIMENT_ID}.log"

mkdir -p "${PHASE_DIR}" "${CHECK_DIR}" "${OUTPUT_DIR}"

if [[ ! -f "${SUMMARY_FILE}" ]]; then
  printf "phase\ttop_p\ttemperature\tlearning_rate\tdecision\tresume_from\tbest_phase\tbest_epoch\tdev_bleu\tdev_f1\ttest_bleu\ttest_f1\n" > "${SUMMARY_FILE}"
fi

read_metric() {
  local key="$1"
  local file="$2"
  awk -F $'\t' -v key="${key}" '$1 == key { print $2 }' "${file}"
}

evaluate_checkpoint() {
  local model_path="$1"
  local split="$2"
  local prefix="$3"
  local src_file="${4:-}"
  local tgt_file="${5:-}"

  local predictions_file="${OUTPUT_DIR}/${prefix}.${TGT_LANG}"
  local metrics_file="${OUTPUT_DIR}/${prefix}.metrics.tsv"

  if [[ -n "${src_file}" && -n "${tgt_file}" ]]; then
    python translation.py \
      --model_path "${model_path}" \
      --domain "${DOMAIN}" \
      --src_lang "${SRC_LANG}" \
      --tgt_lang "${TGT_LANG}" \
      --dataset_path "${DATASET_PATH}" \
      --split "${split}" \
      --src_file "${src_file}" \
      --tgt_file "${tgt_file}" \
      --device "${EVAL_DEVICE}" \
      --predictions_file "${predictions_file}" \
      --metrics_file "${metrics_file}"
  else
    python translation.py \
      --model_path "${model_path}" \
      --domain "${DOMAIN}" \
      --src_lang "${SRC_LANG}" \
      --tgt_lang "${TGT_LANG}" \
      --dataset_path "${DATASET_PATH}" \
      --split "${split}" \
      --device "${EVAL_DEVICE}" \
      --predictions_file "${predictions_file}" \
      --metrics_file "${metrics_file}"
  fi
}

append_summary() {
  local phase="$1"
  local top_p="$2"
  local temperature="$3"
  local learning_rate="$4"
  local decision="$5"
  local resume_from="$6"
  local best_phase="$7"
  local best_epoch="$8"
  local dev_metrics_file="$9"
  local test_metrics_file="${10}"

  local dev_bleu=""
  local dev_f1=""
  local test_bleu=""
  local test_f1=""

  if [[ -f "${dev_metrics_file}" ]]; then
    dev_bleu=$(read_metric "bleu" "${dev_metrics_file}")
    dev_f1=$(read_metric "f1" "${dev_metrics_file}")
  fi
  if [[ -f "${test_metrics_file}" ]]; then
    test_bleu=$(read_metric "bleu" "${test_metrics_file}")
    test_f1=$(read_metric "f1" "${test_metrics_file}")
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${phase}" "${top_p}" "${temperature}" "${learning_rate}" "${decision}" "${resume_from}" \
    "${best_phase}" "${best_epoch}" "${dev_bleu}" "${dev_f1}" "${test_bleu}" "${test_f1}" \
    >> "${SUMMARY_FILE}"
}

lower_idx=0
upper_idx=-1
use_lower=true
temperature="${LOWER_GRID[${lower_idx}]}"
top_p="${LOWER_GRID[${lower_idx}]}"

phase=0
best_phase=-1
best_epoch=0
model_name_or_path="Helsinki-NLP/opus-mt-${SRC_LANG}-${TGT_LANG}"

: > "${LOG_FILE}"

while true; do
  phase_save_dir="${PHASE_DIR}/${DOMAIN}-${EXPERIMENT_ID}-${phase}"
  phase_learning_rate="${LEARNING_RATE}"

  echo "logging: phase=${phase} top_p=${top_p} temperature=${temperature} learning_rate=${phase_learning_rate}" >> "${LOG_FILE}"

  python train_cdpg.py \
    --domain "${DOMAIN}" \
    --src_lang "${SRC_LANG}" \
    --tgt_lang "${TGT_LANG}" \
    --top_p "${top_p}" \
    --temperature "${temperature}" \
    --model_name_or_path "${model_name_or_path}" \
    --seed "${SEED}" \
    --dataset_path "${DATASET_PATH}" \
    --training_dir "${TRAINING_DIR}" \
    --save_dir "${phase_save_dir}" \
    --learning_rate "${phase_learning_rate}" \
    --n_gradient_steps "${N_GRADIENT_STEPS}" \
    --dynamic_mode \
    --experiment_id "${EXPERIMENT_ID}" \
    --phase "${phase}" \
    --threshold_update_num "${THRESHOLD_UPDATE_NUM}" \
    --log_file_path "${LOG_FILE}" \
    --supervision_src_file "${SUPERVISION_SRC_FILE}" \
    --supervision_tgt_file "${SUPERVISION_TGT_FILE}" \
    --dev0 "${DEV0}" \
    --dev1 "${DEV1}"

  decision_line=$(tail -n 1 "${LOG_FILE}")
  decision="reject"
  current_dev_metrics_file=""
  current_test_metrics_file=""

  if [[ "${decision_line}" == Accept* ]]; then
    best_epoch=$(echo "${decision_line}" | grep -oE '[0-9]+$')
    best_phase="${phase}"
    model_name_or_path="${phase_save_dir}/${best_epoch}-epoch"
    LEARNING_RATE=$(python -c "print(${LEARNING_RATE} / ${LR_DESCENT_RATIO})")
    decision="accept"

    current_dev_prefix="${DOMAIN}_${EXPERIMENT_ID}_phase${phase}_best_dev"
    current_test_prefix="${DOMAIN}_${EXPERIMENT_ID}_phase${phase}_best_test"
    evaluate_checkpoint "${model_name_or_path}" "dev" "${current_dev_prefix}" "${SUPERVISION_SRC_FILE}" "${SUPERVISION_TGT_FILE}"
    evaluate_checkpoint "${model_name_or_path}" "test" "${current_test_prefix}"
    current_dev_metrics_file="${OUTPUT_DIR}/${current_dev_prefix}.metrics.tsv"
    current_test_metrics_file="${OUTPUT_DIR}/${current_test_prefix}.metrics.tsv"
  else
    if [[ "${use_lower}" == true ]]; then
      use_lower=false
    else
      use_lower=true
    fi
  fi

  if [[ "${EVALUATE_ALL_CHECKPOINTS}" == "true" ]]; then
    for checkpoint_dir in "${phase_save_dir}"/*-epoch; do
      if [[ -d "${checkpoint_dir}" ]]; then
        checkpoint_name=$(basename "${checkpoint_dir}")
        checkpoint_epoch="${checkpoint_name%-epoch}"
        evaluate_checkpoint \
          "${checkpoint_dir}" \
          "test" \
          "${DOMAIN}_${EXPERIMENT_ID}_phase${phase}_epoch${checkpoint_epoch}_test"
      fi
    done
  fi

  append_summary \
    "${phase}" \
    "${top_p}" \
    "${temperature}" \
    "${phase_learning_rate}" \
    "${decision}" \
    "${model_name_or_path}" \
    "${best_phase}" \
    "${best_epoch}" \
    "${current_dev_metrics_file}" \
    "${current_test_metrics_file}"

  if [[ "${use_lower}" == true ]]; then
    lower_idx=$((lower_idx + 1))
    temperature="${LOWER_GRID[${lower_idx}]}"
    top_p="${LOWER_GRID[${lower_idx}]}"
  else
    upper_idx=$((upper_idx + 1))
    temperature="${UPPER_GRID[${upper_idx}]}"
    top_p="${UPPER_GRID[${upper_idx}]}"
  fi

  if [[ "${temperature}" == "0.0" || "${temperature}" == "1.1" ]]; then
    break
  fi

  phase=$((phase + 1))
done
