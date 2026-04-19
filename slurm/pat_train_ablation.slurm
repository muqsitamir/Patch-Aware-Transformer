#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
cd /home/muqsitamir/repos/Part-Aware-Transformer

mkdir -p logs models tb_log track_outputs

source /home/muqsitamir/miniconda3/etc/profile.d/conda.sh
conda activate pat

CONFIG=${CONFIG:-config/UrbanElementsReID_train.yml}
LOG_NAME=${LOG_NAME:?LOG_NAME is required}
CLASS_AWARE_ENABLED=${CLASS_AWARE_ENABLED:-False}
LOSS_WEIGHT=${LOSS_WEIGHT:-0.0}
DISTANCE_PENALTY=${DISTANCE_PENALTY:-0.0}
RETRIEVAL_SOURCE=${RETRIEVAL_SOURCE:-predicted}

python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

python train.py --config_file "$CONFIG" \
  LOG_NAME "$LOG_NAME" \
  MODEL.CLASS_AWARE.ENABLED "$CLASS_AWARE_ENABLED" \
  MODEL.CLASS_AWARE.LOSS_WEIGHT "$LOSS_WEIGHT" \
  MODEL.CLASS_AWARE.DISTANCE_PENALTY "$DISTANCE_PENALTY" \
  MODEL.CLASS_AWARE.RETRIEVAL_SOURCE "$RETRIEVAL_SOURCE"
