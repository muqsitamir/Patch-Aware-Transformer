#!/usr/bin/env bash

# external pretrain
sbatch --job-name=pat-uam-external \
  --partition=gpu \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=12:00:00 \
  --output=logs/%x-%j.out \
  --error=logs/%x-%j.err \
  --wrap='cd /home/muqsitamir/repos/Part-Aware-Transformer && source /home/muqsitamir/miniconda3/etc/profile.d/conda.sh && conda activate pat && python train.py --config_file config/UrbanElementsReID_train.yml DATASETS.MODE external_only DATASETS.EXTERNAL_ROOT /home/muqsitamir/datasets/UAM_unified/UAM_Unified DATASETS.TEST "('\''UrbanElementsReID_test'\'',)" LOG_NAME external_uam_pretrain'

# finetune baseline
sbatch --job-name=pat-ft-ext-base \
  --partition=gpu \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=12:00:00 \
  --output=logs/%x-%j.out \
  --error=logs/%x-%j.err \
  --wrap='cd /home/muqsitamir/repos/Part-Aware-Transformer && source /home/muqsitamir/miniconda3/etc/profile.d/conda.sh && conda activate pat && python train.py --config_file config/UrbanElementsReID_train.yml MODEL.PRETRAIN_CHOICE finetune MODEL.FINETUNE_PATH /home/muqsitamir/repos/Part-Aware-Transformer/models/external_uam_pretrain_baseline/part_attention_vit_60.pth DATASETS.MODE challenge_only DATASETS.ROOT_DIR /home/muqsitamir/datasets/Urban2026/ MODEL.CLASS_AWARE.ENABLED False MODEL.CLASS_AWARE.LOSS_WEIGHT 0.0 MODEL.CLASS_AWARE.DISTANCE_PENALTY 0.0 MODEL.CLASS_AWARE.RETRIEVAL_SOURCE predicted LOG_NAME finetune_external_baseline'

# finetune class-loss
sbatch --job-name=pat-ft-ext-class \
  --partition=gpu \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=12:00:00 \
  --output=logs/%x-%j.out \
  --error=logs/%x-%j.err \
  --wrap='cd /home/muqsitamir/repos/Part-Aware-Transformer && source /home/muqsitamir/miniconda3/etc/profile.d/conda.sh && conda activate pat && python train.py --config_file config/UrbanElementsReID_train.yml MODEL.PRETRAIN_CHOICE finetune MODEL.FINETUNE_PATH /home/muqsitamir/repos/Part-Aware-Transformer/models/external_uam_pretrain_classloss/part_attention_vit_60.pth DATASETS.MODE challenge_only DATASETS.ROOT_DIR /home/muqsitamir/datasets/Urban2026/ MODEL.CLASS_AWARE.ENABLED True MODEL.CLASS_AWARE.LOSS_WEIGHT 1.0 MODEL.CLASS_AWARE.DISTANCE_PENALTY 0.0 MODEL.CLASS_AWARE.RETRIEVAL_SOURCE predicted LOG_NAME finetune_external_classloss'
