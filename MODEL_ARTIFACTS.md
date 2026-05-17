# Model Artifacts

The competition rules require model files to be uploaded through a Google Drive link and linked from GitHub.

## Google Drive Link

```text
TODO_ADD_PUBLIC_GOOGLE_DRIVE_LINK_HERE
```

Set the Drive folder to public read access before sending the repository to the organizers.

## Files To Upload

Upload the exact files used to generate the final selected Kaggle submission:

```text
part_attention_vit_8.pth
jx_vit_large_p16_224-4ee7a4dc.pth
MANIFEST.txt
```

If the submitted model was produced through a multi-stage run, also upload the stage-1 checkpoint used as `BASE_CKPT`:

```text
stage1_base_checkpoint.pth
```

## Suggested MANIFEST.txt Format

```text
Urban Elements ReID Challenge 2026 model artifacts

final_checkpoint:
  file: part_attention_vit_8.pth
  description: final ViT-L PAT target-camera style adaptation checkpoint
  sha256: <fill after upload>

stage1_checkpoint:
  file: stage1_base_checkpoint.pth
  description: challenge-only checkpoint used as BASE_CKPT for adaptation
  sha256: <fill after upload>

pretrained_initialization:
  file: jx_vit_large_p16_224-4ee7a4dc.pth
  description: public ViT-L ImageNet initialization used by PAT/TransReID code path
  sha256: <fill after upload>

created_labels:
  none
```

## Checksum Command

```bash
sha256sum part_attention_vit_8.pth stage1_base_checkpoint.pth jx_vit_large_p16_224-4ee7a4dc.pth
```

On macOS:

```bash
shasum -a 256 part_attention_vit_8.pth stage1_base_checkpoint.pth jx_vit_large_p16_224-4ee7a4dc.pth
```
