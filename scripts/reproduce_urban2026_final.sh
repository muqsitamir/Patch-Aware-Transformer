#!/usr/bin/env bash
set -euo pipefail

CHALLENGE_ROOT="${CHALLENGE_ROOT:?Set CHALLENGE_ROOT to the Urban2026 dataset root}"
PRETRAIN_DIR="${PRETRAIN_DIR:?Set PRETRAIN_DIR to the directory containing ViT pretrained weights}"
BASE_CKPT="${BASE_CKPT:?Set BASE_CKPT to the checkpoint used to initialize target-camera adaptation}"

PYTHON="${PYTHON:-python}"
LOG_NAME="${LOG_NAME:-vitl_c004style_from_best_e8_lr6e6_b8}"
MAX_EPOCHS="${MAX_EPOCHS:-8}"
SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-8}"
EXPORT_EPOCHS="${EXPORT_EPOCHS:-4 5 6 8}"
IMS_PER_BATCH="${IMS_PER_BATCH:-8}"
TEST_BATCH="${TEST_BATCH:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p models submissions features track_outputs tb_log

"${PYTHON}" train.py \
  --config_file config/UrbanElementsReID_train.yml \
  DATASETS.MODE challenge_only \
  DATASETS.ROOT_DIR "${CHALLENGE_ROOT}" \
  DATASETS.TRAIN "('UrbanElementsReID',)" \
  DATASETS.TEST "('UrbanElementsReID_test',)" \
  MODEL.TRANSFORMER_TYPE vit_large_patch16_224_TransReID \
  MODEL.PRETRAIN_PATH "${PRETRAIN_DIR}" \
  MODEL.PRETRAIN_CHOICE finetune \
  MODEL.FINETUNE_PATH "${BASE_CKPT}" \
  MODEL.CLASS_AWARE.ENABLED False \
  MODEL.CLASS_AWARE.LOSS_WEIGHT 0.0 \
  MODEL.CLASS_AWARE.DISTANCE_PENALTY 0.0 \
  SOLVER.BASE_LR 0.000006 \
  SOLVER.MAX_EPOCHS "${MAX_EPOCHS}" \
  SOLVER.SCHEDULER_EPOCHS "${SCHEDULER_EPOCHS}" \
  SOLVER.WARMUP_EPOCHS 0 \
  SOLVER.CHECKPOINT_PERIOD 1 \
  SOLVER.DELETE_OLD_CHECKPOINTS False \
  SOLVER.IMS_PER_BATCH "${IMS_PER_BATCH}" \
  DATALOADER.NUM_INSTANCE 2 \
  DATALOADER.NUM_WORKERS "${NUM_WORKERS}" \
  INPUT.SIZE_TRAIN "[288,256]" \
  INPUT.SIZE_TEST "[288,256]" \
  INPUT.ASPECT_PAD.ENABLED True \
  INPUT.ASPECT_PAD.FILL 128 \
  INPUT.DO_FLIP True \
  INPUT.FLIP_PROB 0.5 \
  INPUT.LGT.DO_LGT False \
  INPUT.CJ.ENABLED True \
  INPUT.CJ.PROB 0.35 \
  INPUT.CJ.BRIGHTNESS 0.12 \
  INPUT.CJ.CONTRAST 0.12 \
  INPUT.CJ.SATURATION 0.08 \
  INPUT.CJ.HUE 0.03 \
  INPUT.REA.ENABLED False \
  INPUT.TARGET_STYLE.ENABLED True \
  INPUT.TARGET_STYLE.ROOT_DIR "${CHALLENGE_ROOT}" \
  INPUT.TARGET_STYLE.IMAGE_DIR image_query \
  INPUT.TARGET_STYLE.PROB 0.7 \
  INPUT.TARGET_STYLE.STRENGTH 0.6 \
  TEST.SKIP_EVAL_IF_DUMMY_IDS True \
  TEST.IMS_PER_BATCH "${TEST_BATCH}" \
  LOG_NAME "${LOG_NAME}"

for epoch in ${EXPORT_EPOCHS}; do
  ckpt="models/${LOG_NAME}/part_attention_vit_${epoch}.pth"
  if [[ ! -f "${ckpt}" ]]; then
    echo "missing checkpoint ${ckpt}; skipping export" >&2
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
    INPUT.SIZE_TRAIN "[288,256]" \
    INPUT.SIZE_TEST "[288,256]" \
    INPUT.ASPECT_PAD.ENABLED True \
    DATALOADER.NUM_WORKERS 0 \
    TEST.WEIGHT "${ckpt}" \
    TEST.IMS_PER_BATCH "${TEST_BATCH}" \
    TEST.RE_RANKING True \
    TEST.QUERY_EXPANSION False \
    TEST.CLASS_POSTPROCESS off \
    TEST.CLASS_MISMATCH_PENALTY 0.0 \
    TEST.FEAT_Q_PATH "features/${LOG_NAME}_e${epoch}_qf.npy" \
    TEST.FEAT_G_PATH "features/${LOG_NAME}_e${epoch}_gf.npy" \
    LOG_NAME "submissions/${LOG_NAME}_e${epoch}_fast"
done
