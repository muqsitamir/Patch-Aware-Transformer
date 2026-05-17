# Urban Elements ReID Challenge 2026 Code Release

This repository contains the code used for our Urban Elements ReID Challenge 2026 submissions.

Official references:

- Challenge listing: https://2026.ieeeicip.org/grand-challenges/
- Challenge website: http://www-vpu.eps.uam.es/challenges/UrbanReIDChallenge2026/
- Kaggle competition: https://www.kaggle.com/competitions/urban-elements-re-id-challenge-2026/

## Method Summary

The solution is based on Part-Aware Transformer (PAT) with a ViT-L/16 backbone. The final recipe is a target-camera adaptation run:

- Backbone: `vit_large_patch16_224_TransReID`
- Input representation: aspect-preserving resize with center padding at `288x256`
- Initialization: ViT-L ImageNet checkpoint, then challenge fine-tuning checkpoint
- Training augmentation: horizontal flip, mild color jitter, and target-camera style transfer
- Target-camera style source: unlabeled `image_query` crops from the challenge test set
- Inference: normalized embeddings, k-reciprocal re-ranking in the fast exporter, no semantic class postprocessing, no query expansion

The target-camera style transfer is implemented in `data/transforms/transforms.py` as `TargetStyleTransfer` and is opt-in through:

```yaml
INPUT.TARGET_STYLE.ENABLED: True
INPUT.TARGET_STYLE.ROOT_DIR: <challenge-root>
INPUT.TARGET_STYLE.IMAGE_DIR: image_query
```

## External Data And Pretrained Weights

The code expects ImageNet ViT checkpoints in `MODEL.PRETRAIN_PATH`. For ViT-L, the file used by this code path is:

```text
jx_vit_large_p16_224-4ee7a4dc.pth
```

If reproducing the exact final run from a previous challenge-adapted checkpoint, set `BASE_CKPT` to that checkpoint. If reproducing from scratch, first train the baseline PAT/ViT-L model on the challenge training split, then use the resulting checkpoint as `BASE_CKPT` for target-camera adaptation.

No challenge labels from the test/query/gallery split are used. The target-style augmentation uses only unlabeled query image color statistics.

## Expected Dataset Layout

Set `CHALLENGE_ROOT` to a directory with this structure:

```text
Urban2026/
  train.csv
  query.csv
  test.csv
  image_train/
  image_query/
  image_test/
```

The CSV files are read by `data/datasets/UrbanElementsReID.py` and `data/datasets/UrbanElementsReID_test.py`.

## Environment

Recommended:

```bash
conda create -n pat python=3.10
conda activate pat
pip install -r requirements.txt
```

The original environment bootstrap remains available in `enviroments.sh`.

## Reproduction

The end-to-end target-camera adaptation script is:

```bash
bash scripts/reproduce_urban2026_final.sh
```

Required environment variables:

```bash
export CHALLENGE_ROOT=/path/to/Urban2026
export PRETRAIN_DIR=/path/to/pretrained
export BASE_CKPT=/path/to/base/part_attention_vit_*.pth
```

Optional environment variables:

```bash
export LOG_NAME=vitl_c004style_from_best_e8_lr6e6_b8
export MAX_EPOCHS=8
export EXPORT_EPOCHS="4 5 6 8"
export IMS_PER_BATCH=8
export TEST_BATCH=16
```

The script writes:

```text
models/<LOG_NAME>/part_attention_vit_<epoch>.pth
submissions/<LOG_NAME>_e<epoch>_fast_submission.csv
features/<LOG_NAME>_e<epoch>_qf.npy
features/<LOG_NAME>_e<epoch>_gf.npy
```

## Final Submission Generation Only

Given a trained checkpoint, generate a Kaggle CSV with:

```bash
python tools/export_submission_fast.py \
  --config_file config/UrbanElementsReID_test.yml \
  --track submissions/final.txt \
  DATASETS.MODE challenge_only \
  DATASETS.ROOT_DIR "$CHALLENGE_ROOT" \
  DATASETS.TEST "('UrbanElementsReID_test',)" \
  MODEL.TRANSFORMER_TYPE vit_large_patch16_224_TransReID \
  MODEL.PRETRAIN_PATH "$PRETRAIN_DIR" \
  MODEL.CLASS_AWARE.ENABLED False \
  MODEL.CLASS_AWARE.LOSS_WEIGHT 0.0 \
  MODEL.CLASS_AWARE.DISTANCE_PENALTY 0.0 \
  INPUT.SIZE_TRAIN "[288,256]" \
  INPUT.SIZE_TEST "[288,256]" \
  INPUT.ASPECT_PAD.ENABLED True \
  DATALOADER.NUM_WORKERS 0 \
  TEST.WEIGHT "$CHECKPOINT" \
  TEST.IMS_PER_BATCH 16 \
  TEST.RE_RANKING True \
  TEST.QUERY_EXPANSION False \
  TEST.CLASS_POSTPROCESS off \
  TEST.CLASS_MISMATCH_PENALTY 0.0 \
  LOG_NAME submissions/final
```

The generated file is:

```text
submissions/final_submission.csv
```

## Files Added For Challenge Reproduction

- `tools/export_submission_fast.py`: direct feature extraction and Kaggle CSV generation.
- `data/transforms/transforms.py`: `ResizePad` and `TargetStyleTransfer`.
- `data/transforms/build.py`: config wiring for aspect-pad and target-style augmentation.
- `config/defaults.py`: challenge-specific config fields.
- `scripts/reproduce_urban2026_final.sh`: final training/export workflow.
