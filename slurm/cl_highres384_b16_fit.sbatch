#!/bin/bash
#SBATCH --job-name=pat_hi448_b16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/muqsitamir/repos/Part-Aware-Transformer}"

PYTHON="${PYTHON:-/home/muqsitamir/miniconda3/envs/pat/bin/python}"
CHALLENGE_ROOT="${CHALLENGE_ROOT:-/home/muqsitamir/datasets/Urban2026}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-/home/muqsitamir/datasets/UAM_unified/UAM_Unified}"
PRETRAIN_DIR="${PRETRAIN_DIR:-${PWD}/models/pretrained}"
BASE_CKPT="${BASE_CKPT:-${PWD}/models/vitl_aspectpad_288x256_e6_anneal_lr1e5_e4/part_attention_vit_1.pth}"
LOG_NAME="${LOG_NAME:-vitl_aspectpad_448x400_cl_b16_hardaug_e12}"
EXPORT_EPOCHS="${EXPORT_EPOCHS:-3 6 9 12}"
TEST_BATCH="${TEST_BATCH:-32}"

mkdir -p slurm_logs models tb_log submissions track_outputs features

echo "repo=${PWD}"
echo "python=${PYTHON}"
echo "challenge=${CHALLENGE_ROOT}"
echo "external=${EXTERNAL_ROOT}"
echo "base_ckpt=${BASE_CKPT}"
echo "log_name=${LOG_NAME}"
nvidia-smi || true

"${PYTHON}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
PY

"${PYTHON}" train.py \
  --config_file config/UrbanElementsReID_train.yml \
  DATASETS.MODE challenge_plus_external \
  DATASETS.ROOT_DIR "${CHALLENGE_ROOT}" \
  DATASETS.EXTERNAL_ROOT "${EXTERNAL_ROOT}" \
  DATASETS.TRAIN "('UrbanElementsReID',)" \
  DATASETS.TEST "('UrbanElementsReID_test',)" \
  MODEL.TRANSFORMER_TYPE vit_large_patch16_224_TransReID \
  MODEL.PRETRAIN_PATH "${PRETRAIN_DIR}" \
  MODEL.PRETRAIN_CHOICE finetune \
  MODEL.FINETUNE_PATH "${BASE_CKPT}" \
  MODEL.CLASS_AWARE.ENABLED False \
  MODEL.CLASS_AWARE.LOSS_WEIGHT 0.0 \
  MODEL.CLASS_AWARE.DISTANCE_PENALTY 0.0 \
  MODEL.PC_LOSS False \
  INPUT.SIZE_TRAIN "[448,400]" \
  INPUT.SIZE_TEST "[448,400]" \
  INPUT.ASPECT_PAD.ENABLED True \
  INPUT.LGT.DO_LGT False \
  INPUT.DO_FLIP True \
  INPUT.FLIP_PROB 0.5 \
  INPUT.CJ.ENABLED True \
  INPUT.CJ.PROB 0.75 \
  INPUT.CJ.BRIGHTNESS 0.30 \
  INPUT.CJ.CONTRAST 0.30 \
  INPUT.CJ.SATURATION 0.25 \
  INPUT.CJ.HUE 0.06 \
  INPUT.DO_AUGMIX True \
  INPUT.REA.ENABLED True \
  INPUT.REA.PROB 0.35 \
  INPUT.RPT.ENABLED True \
  INPUT.RPT.PROB 0.30 \
  DATALOADER.NUM_WORKERS 8 \
  DATALOADER.NUM_INSTANCE 2 \
  SOLVER.IMS_PER_BATCH 16 \
  SOLVER.BASE_LR 0.000005 \
  SOLVER.MAX_EPOCHS 12 \
  SOLVER.SCHEDULER_EPOCHS 14 \
  SOLVER.WARMUP_EPOCHS 0 \
  SOLVER.CHECKPOINT_PERIOD 1 \
  SOLVER.DELETE_OLD_CHECKPOINTS False \
  TEST.IMS_PER_BATCH "${TEST_BATCH}" \
  TEST.SKIP_EVAL_IF_DUMMY_IDS True \
  LOG_NAME "${LOG_NAME}"

for epoch in ${EXPORT_EPOCHS}; do
  ckpt="models/${LOG_NAME}/part_attention_vit_${epoch}.pth"
  if [ ! -f "${ckpt}" ]; then
    echo "missing checkpoint ${ckpt}; skipping export"
    continue
  fi
  "${PYTHON}" tools/export_submission_fast.py \
    --config_file config/UrbanElementsReID_test.yml \
    --track "submissions/${LOG_NAME}_e${epoch}_fast.txt" \
    DATASETS.MODE challenge_only \
    DATASETS.ROOT_DIR "${CHALLENGE_ROOT}" \
    DATASETS.TEST "('UrbanElementsReID_test',)" \
    MODEL.TRANSFORMER_TYPE vit_large_patch16_224_TransReID \
    MODEL.PRETRAIN_PATH "${PRETRAIN_DIR}" \
    MODEL.CLASS_AWARE.ENABLED False \
    MODEL.CLASS_AWARE.LOSS_WEIGHT 0.0 \
    MODEL.CLASS_AWARE.DISTANCE_PENALTY 0.0 \
    INPUT.SIZE_TEST "[448,400]" \
    INPUT.ASPECT_PAD.ENABLED True \
    DATALOADER.NUM_WORKERS 8 \
    TEST.WEIGHT "${ckpt}" \
    TEST.IMS_PER_BATCH "${TEST_BATCH}" \
    TEST.QUERY_EXPANSION False \
    TEST.CLASS_POSTPROCESS off \
    TEST.CLASS_MISMATCH_PENALTY 0.0 \
    TEST.FEAT_Q_PATH "features/${LOG_NAME}_e${epoch}_qf.npy" \
    TEST.FEAT_G_PATH "features/${LOG_NAME}_e${epoch}_gf.npy" \
    LOG_NAME "submissions/${LOG_NAME}_e${epoch}_fast"
done
