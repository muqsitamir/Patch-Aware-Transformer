#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/rhome/mmi/projects/Patch-Aware-Transformer"
PY="/home/mmi/.conda/envs/pat/bin/python"
CHALLENGE_ROOT="/mnt/rhome/mmi/datasets/Urban2026"
PRETRAIN_DIR="${ROOT}/models/pretrained"
TEST_SET="('UrbanElementsReID_test',)"
LOG_NAME="vitl_aspectpad_320x288_from_e6_hardaug_b6_fit_e8"

cd "${ROOT}"
mkdir -p submissions features models/vpulab_queue_logs
LOG="models/vpulab_queue_logs/export_mediumres320_e6e7_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

for E in 6 7; do
  echo "[$(date)] exporting ${LOG_NAME} epoch ${E}"
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

echo "[$(date)] medium-res e6/e7 export finished"
