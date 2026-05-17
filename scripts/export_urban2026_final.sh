#!/usr/bin/env bash
set -euo pipefail

CHALLENGE_ROOT="${CHALLENGE_ROOT:?Set CHALLENGE_ROOT to the Urban2026 dataset root}"
PRETRAIN_DIR="${PRETRAIN_DIR:?Set PRETRAIN_DIR to the directory containing ViT pretrained weights}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the downloaded final .pth checkpoint}"

PYTHON="${PYTHON:-python}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-submissions/final}"
TEST_BATCH="${TEST_BATCH:-16}"

mkdir -p "$(dirname "${OUTPUT_PREFIX}")" features

"${PYTHON}" tools/export_submission_fast.py \
  --config_file config/UrbanElementsReID_test.yml \
  --track "${OUTPUT_PREFIX}.txt" \
  DATASETS.MODE challenge_only \
  DATASETS.ROOT_DIR "${CHALLENGE_ROOT}" \
  DATASETS.TEST "('UrbanElementsReID_test',)" \
  MODEL.TRANSFORMER_TYPE vit_large_patch16_224_TransReID \
  MODEL.PRETRAIN_PATH "${PRETRAIN_DIR}" \
  MODEL.CLASS_AWARE.ENABLED False \
  MODEL.CLASS_AWARE.LOSS_WEIGHT 0.0 \
  MODEL.CLASS_AWARE.DISTANCE_PENALTY 0.0 \
  INPUT.SIZE_TRAIN "[288,256]" \
  INPUT.SIZE_TEST "[288,256]" \
  INPUT.ASPECT_PAD.ENABLED True \
  DATALOADER.NUM_WORKERS 0 \
  TEST.WEIGHT "${CHECKPOINT}" \
  TEST.IMS_PER_BATCH "${TEST_BATCH}" \
  TEST.RE_RANKING True \
  TEST.QUERY_EXPANSION False \
  TEST.CLASS_POSTPROCESS off \
  TEST.CLASS_MISMATCH_PENALTY 0.0 \
  TEST.FEAT_Q_PATH "features/final_qf.npy" \
  TEST.FEAT_G_PATH "features/final_gf.npy" \
  LOG_NAME "${OUTPUT_PREFIX}"
