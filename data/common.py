import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_utils import read_image
from utils.class_aware import CLASS_ID_KEY

# ---------------------------------------------------------------------------
# Preprocessing-Pipeline import helper (same pattern as vit_pytorch.py).
# The directory name contains a hyphen so it cannot be imported normally;
# we add it to sys.path and import from individual module files.
# ---------------------------------------------------------------------------
_PIPELINE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../PreprocessingPipeline')
)

def _ensure_pipeline_importable():
    if _PIPELINE_DIR not in sys.path:
        sys.path.insert(0, _PIPELINE_DIR)


class CommDataset(Dataset):
    """Image Person ReID Dataset.

    Optional mask support: when ``mask_gen`` is provided the dataset generates
    foreground-region masks for each image (via ForegroundMaskGenerator) and
    returns them alongside the image under the ``"masks"`` key.  Pass
    ``paired_transform`` (a PairedImageMaskTransform) to apply geometric
    augmentations identically to both image and masks.  If ``mask_cache`` is
    provided, generated masks are persisted to disk so they are only computed
    once across epochs.
    """

    def __init__(self, img_items, transform=None, relabel=True,
                 mask_gen=None, mask_cache=None, paired_transform=None):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel
        self.mask_gen = mask_gen
        self.mask_cache = mask_cache
        self.paired_transform = paired_transform

        self.pid_dict = {}
        if self.relabel:
            pids = list()
            for i, item in enumerate(img_items):
                if item[1] in pids: continue
                pids.append(item[1])
            self.pids = pids
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])

        class_ids = []
        for item in img_items:
            if len(item) > 3 and isinstance(item[-1], dict):
                class_id = item[-1].get(CLASS_ID_KEY, -1)
                if class_id >= 0:
                    class_ids.append(class_id)
        self.num_semantic_classes = max(class_ids) + 1 if class_ids else 0

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        if len(self.img_items[index]) > 3:
            img_path, pid, camid, others = self.img_items[index]
        else:
            img_path, pid, camid = self.img_items[index]
            others = {}
        if others == '':
            others = {}
        img = read_image(img_path)
        if self.relabel:
            pid = self.pid_dict[pid]

        if self.mask_gen is not None:
            # class_name is used by ForegroundMaskGenerator to decide whether
            # to fall back to whole-image masks (e.g. for 'crosswalk').
            cls_name = others.get('class_name', '') if isinstance(others, dict) else ''

            if self.mask_cache is not None:
                masks_np = self.mask_cache.get_or_compute(
                    img_path, lambda: self.mask_gen(img, cls=cls_name)
                )
            else:
                masks_np = self.mask_gen(img, cls=cls_name)

            if self.paired_transform is not None:
                img, masks = self.paired_transform(img, masks_np)
            else:
                if self.transform is not None:
                    img = self.transform(img)
                masks = torch.from_numpy(np.array(masks_np, dtype=np.float32))

            return {
                "images": img,
                "masks": masks,
                "targets": pid,
                "camid": camid,
                "img_path": img_path,
                "others": others,
            }
        else:
            if self.transform is not None:
                img = self.transform(img)
            return {
                "images": img,
                "targets": pid,
                "camid": camid,
                "img_path": img_path,
                "others": others,
            }

    @property
    def num_classes(self):
        return len(self.pids)
