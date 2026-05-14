#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/rhome/mmi/projects/Patch-Aware-Transformer"
PY="/home/mmi/.conda/envs/pat/bin/python"
CHALLENGE_ROOT="/mnt/rhome/mmi/datasets/Urban2026"
EXTERNAL_ROOT="/mnt/rhome/mmi/datasets/UAM_unified/UAM_Unified"
PRETRAIN_DIR="${ROOT}/models/pretrained"
TRAIN_SET="('UrbanElementsReID',)"
TEST_SET="('UrbanElementsReID_test',)"
LOG_NAME="vitl_aspectpad_320x288_from_e6_hardaug_b6_e3"

cd "${ROOT}"
mkdir -p submissions features models/vpulab_queue_logs
LOG="models/vpulab_queue_logs/mediumres320_b6_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

echo "[$(date)] medium-res 320x288 b6 started"
nvidia-smi || true

"${PY}" train.py \
  --config_file config/UrbanElementsReID_train.yml \
  DATASETS.MODE challenge_plus_external \
  DATASETS.ROOT_DIR "${CHALLENGE_ROOT}" \
  DATASETS.EXTERNAL_ROOT "${EXTERNAL_ROOT}" \
  DATASETS.TRAIN "${TRAIN_SET}" \
  DATASETS.TEST "${TEST_SET}" \
  MODEL.NAME part_attention_vit \
  MODEL.TRANSFORMER_TYPE vit_large_patch16_224_TransReID \
  MODEL.PRETRAIN_PATH "${PRETRAIN_DIR}" \
  MODEL.PRETRAIN_CHOICE finetune \
  MODEL.FINETUNE_PATH models/vitl_aspectpad_288x256_from_e15_hardaug_e12/part_attention_vit_6.pth \
  MODEL.FREEZE_PATCH_EMBED False \
  MODEL.PC_LOSS False \
  MODEL.DROP_PATH 0.3 \
  MODEL.CLASS_AWARE.ENABLED False \
  MODEL.CLASS_AWARE.LOSS_WEIGHT 0.0 \
  MODEL.CLASS_AWARE.DISTANCE_PENALTY 0.0 \
  INPUT.SIZE_TRAIN "[320,288]" \
  INPUT.SIZE_TEST "[320,288]" \
  INPUT.ASPECT_PAD.ENABLED True \
  INPUT.ASPECT_PAD.FILL 128 \
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
  DATALOADER.NUM_WORKERS 0 \
  DATALOADER.NUM_INSTANCE 2 \
  SOLVER.OPTIMIZER_NAME Adam \
  SOLVER.IMS_PER_BATCH 6 \
  SOLVER.BASE_LR 0.000008 \
  SOLVER.LARGE_FC_LR True \
  SOLVER.MAX_EPOCHS 3 \
  SOLVER.SCHEDULER_EPOCHS 10 \
  SOLVER.WARMUP_EPOCHS 0 \
  SOLVER.WEIGHT_DECAY 0.0002 \
  SOLVER.CHECKPOINT_PERIOD 1 \
  SOLVER.DELETE_OLD_CHECKPOINTS False \
  TEST.SKIP_EVAL_IF_DUMMY_IDS True \
  TEST.IMS_PER_BATCH 32 \
  LOG_NAME "${LOG_NAME}"

for E in 1 2 3; do
  echo "[$(date)] exporting medium-res epoch ${E}"
  "${PY}" tools/export_submission_fast.py \
    --config_file config/UrbanElementsReID_test.yml \
    --track "submissions/${LOG_NAME}_e${E}_fast.txt" \
    DATASETS.MODE challenge_only \
    DATASETS.ROOT_DIR "${CHALLENGE_ROOT}" \
    DATASETS.TEST "${TEST_SET}" \
    MODEL.TRANSFORMER_TYPE vit_large_patch16_224_TransReID \
    MODEL.PRETRAIN_PATH "${PRETRAIN_DIR}" \
    MODEL.CLASS_AWARE.ENABLED False \
    MODEL.CLASS_AWARE.LOSS_WEIGHT 0.0 \
    MODEL.CLASS_AWARE.DISTANCE_PENALTY 0.0 \
    INPUT.SIZE_TRAIN "[320,288]" \
    INPUT.SIZE_TEST "[320,288]" \
    INPUT.ASPECT_PAD.ENABLED True \
    DATALOADER.NUM_WORKERS 0 \
    TEST.WEIGHT "models/${LOG_NAME}/part_attention_vit_${E}.pth" \
    TEST.IMS_PER_BATCH 32 \
    TEST.QUERY_EXPANSION False \
    TEST.CLASS_POSTPROCESS off \
    TEST.CLASS_MISMATCH_PENALTY 0.0 \
    TEST.FEAT_Q_PATH "features/${LOG_NAME}_e${E}_qf.npy" \
    TEST.FEAT_G_PATH "features/${LOG_NAME}_e${E}_gf.npy" \
    LOG_NAME "submissions/${LOG_NAME}_e${E}_fast"
done

echo "[$(date)] medium-res 320x288 b6 finished"
