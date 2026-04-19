import torch
from torch.utils.data import Dataset

from .data_utils import read_image
from utils.class_aware import CLASS_ID_KEY


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

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
        if self.transform is not None: img = self.transform(img)
        if self.relabel: pid = self.pid_dict[pid]
        return {
            "images": img,
            "targets": pid,
            "camid": camid,
            "img_path": img_path,
            "others": others
        }

    @property
    def num_classes(self):
        return len(self.pids)
