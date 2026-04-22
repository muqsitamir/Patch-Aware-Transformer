# encoding: utf-8

import csv
import os.path as osp

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
from utils.class_aware import add_class_metadata, build_class_to_idx, read_class_csv


class _UrbanElementsReIDLocalValBase(ImageDataset):
    train_csv = "local_train.csv"
    query_csv = "local_val_query.csv"
    gallery_csv = "local_val_gallery.csv"
    train_class_csv = "local_train_classes.csv"
    query_class_csv = "local_val_query_classes.csv"
    gallery_class_csv = "local_val_gallery_classes.csv"

    def __init__(self, root="/home/muqsitamir/datasets/Urban2026/", verbose=True, **kwargs):
        self.class_aware = kwargs.pop("class_aware", False)
        self.class_aware_cfg = kwargs.pop("class_aware_cfg", None)
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, "image_train/")
        self.query_dir = osp.join(self.dataset_dir, "image_train/")
        self.gallery_dir = osp.join(self.dataset_dir, "image_train/")

        self._check_before_run()
        self.class_to_idx = self._build_class_to_idx()
        self.num_semantic_classes = len(self.class_to_idx)

        train = self._process_csv(
            self.train_dir,
            self.train_csv,
            self._class_cfg_value("TRAIN_CSV", self.train_class_csv),
            relabel=True,
        )
        query = self._process_csv(
            self.query_dir,
            self.query_csv,
            self._class_cfg_value("QUERY_CSV", self.query_class_csv),
            relabel=False,
        )
        gallery = self._process_csv(
            self.gallery_dir,
            self.gallery_csv,
            self._class_cfg_value("TEST_CSV", self.gallery_class_csv),
            relabel=False,
        )

        self.train = train
        self.query = query
        self.gallery = gallery

        super(_UrbanElementsReIDLocalValBase, self).__init__(
            self.train, self.query, self.gallery, verbose=verbose, **kwargs
        )

    def _check_before_run(self):
        required = (
            self.dataset_dir,
            self.train_dir,
            osp.join(self.dataset_dir, self.train_csv),
            osp.join(self.dataset_dir, self.query_csv),
            osp.join(self.dataset_dir, self.gallery_csv),
        )
        for path in required:
            if not osp.exists(path):
                raise RuntimeError("'{}' is not available".format(path))

    def _class_cfg_value(self, key, default):
        if self.class_aware_cfg is None:
            return default
        value = getattr(self.class_aware_cfg, key, default)
        challenge_defaults = {
            "TRAIN_CSV": "train_classes.csv",
            "QUERY_CSV": "query_classes.csv",
            "TEST_CSV": "test_classes.csv",
        }
        if value in ("", None, challenge_defaults.get(key)):
            return default
        return value

    def _build_class_to_idx(self):
        if not self.class_aware:
            return {}
        train_csv = self._class_cfg_value("TRAIN_CSV", self.train_class_csv)
        return build_class_to_idx(self.dataset_dir, train_csv)

    def _read_class_map(self, csv_name):
        if not self.class_aware:
            return {}
        return read_class_csv(osp.join(self.dataset_dir, csv_name))

    def _read_reid_csv(self, csv_name):
        csv_path = osp.join(self.dataset_dir, csv_name)
        rows = []
        with open(csv_path, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if not reader.fieldnames:
                raise ValueError("{} has no header".format(csv_path))
            columns = {name.strip(): name for name in reader.fieldnames}
            camera_col = columns.get("cameraID") or columns.get("camera_id")
            image_col = columns.get("imageName") or columns.get("image_name")
            pid_col = (
                columns.get("Corresponding Indexes")
                or columns.get("objectID")
                or columns.get("object")
            )
            if camera_col is None or image_col is None or pid_col is None:
                raise ValueError("{} is missing cameraID/imageName/identity columns".format(csv_path))

            for row in reader:
                camera_id = str(row.get(camera_col, "")).strip()
                image_name = str(row.get(image_col, "")).strip()
                pid = int(str(row.get(pid_col, "")).strip())
                rows.append((camera_id, image_name, pid))
        return rows

    def _process_csv(self, dir_path, csv_name, class_csv_name, relabel=False):
        csv_rows = self._read_reid_csv(csv_name)
        image_to_class = self._read_class_map(class_csv_name)

        pid_container = sorted({pid for _, _, pid in csv_rows if pid != -1})
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for camera_id, image_name, pid in csv_rows:
            if pid == -1:
                continue
            camid = int(camera_id[1:]) if camera_id.startswith("c") else int(camera_id)
            if relabel:
                pid = pid2label[pid]
            item = (osp.join(dir_path, image_name), pid, camid)
            if self.class_aware:
                item = add_class_metadata(item, image_name, image_to_class, self.class_to_idx)
            dataset.append(item)
        return dataset


@DATASET_REGISTRY.register()
class UrbanElementsReID_localval_train(_UrbanElementsReIDLocalValBase):
    pass


@DATASET_REGISTRY.register()
class UrbanElementsReID_localval_test(_UrbanElementsReIDLocalValBase):
    pass
